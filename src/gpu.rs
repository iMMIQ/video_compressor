use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::process::Command;
use std::time::{Duration, Instant};

pub const MIN_CONCURRENT: usize = 1;
pub const DEFAULT_MAX_CONCURRENT: usize = 8;
pub const DEFAULT_JETSON_HW_CONCURRENT: usize = 2;
const MARGINAL_GAIN_THRESHOLD: u8 = 5;
const UTILIZATION_HIGH: u8 = 90;
const MAX_HISTORY_LEN: usize = 64;
const DECREASE_COOLDOWN_TICKS: u8 = 4;

/// Desktop NVIDIA NVENC utilization monitor (multi-GPU aware).
pub struct GpuMonitor {
    last_check: Instant,
    cache_duration: Duration,
    cached_utilization: u8,
}

impl GpuMonitor {
    pub fn new() -> Self {
        Self {
            last_check: Instant::now() - Duration::from_secs(10),
            cache_duration: Duration::from_secs(2),
            cached_utilization: 0,
        }
    }

    /// Get average NVENC encoder utilization across GPUs (with 2s cache).
    pub fn get_utilization(&mut self) -> Result<u8> {
        if self.last_check.elapsed() < self.cache_duration {
            return Ok(self.cached_utilization);
        }

        let util = self.read_nvidia_smi_load()?;
        self.cached_utilization = util;
        self.last_check = Instant::now();
        Ok(util)
    }

    /// Read NVENC encoder utilization from nvidia-smi, falling back to GPU utilization
    /// if the driver reports `[N/A]` for encoder utilization.
    fn read_nvidia_smi_load(&self) -> Result<u8> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.encoder,utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .context("无法执行 nvidia-smi")?;

        if !output.status.success() {
            anyhow::bail!("nvidia-smi 执行失败");
        }

        let util_str = String::from_utf8_lossy(&output.stdout);
        Ok(parse_nvidia_smi_utilization(&util_str))
    }
}

/// Parse `nvidia-smi --query-gpu=utilization.encoder,utilization.gpu --format=csv,noheader,nounits`.
/// Averages across all GPUs on multi-GPU hosts; prefers encoder utilization and falls back to GPU utilization.
pub fn parse_nvidia_smi_utilization(stdout: &str) -> u8 {
    let mut sum: u32 = 0;
    let mut count: u32 = 0;
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let mut parts = line.split(',').map(|s| s.trim());
        let enc_util = parts.next().and_then(|s| s.parse::<u8>().ok());
        let gpu_util = parts.next().and_then(|s| s.parse::<u8>().ok());
        if let Some(u) = enc_util.or(gpu_util) {
            sum += u.min(100) as u32;
            count += 1;
        }
    }
    sum.checked_div(count).unwrap_or(0).min(100) as u8
}

/// Concurrency adjustment action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyAction {
    Increase,
    Decrease,
    Maintain,
}

/// Dynamic concurrency controller with bounded history, zero-util I/O filtering,
/// anti-oscillation cooldown, and dynamic hard-cap for consumer NVENC session limits.
pub struct ConcurrencyController {
    current_max: usize,
    max_limit: usize,
    history: VecDeque<(usize, u8)>, // (concurrent jobs, utilization)
    cooldown: u8,
}

impl ConcurrencyController {
    pub fn new(initial: usize, max_limit: usize) -> Self {
        let clamped_limit = max_limit.max(MIN_CONCURRENT);
        Self {
            current_max: initial.clamp(MIN_CONCURRENT, clamped_limit),
            max_limit: clamped_limit,
            history: VecDeque::with_capacity(MAX_HISTORY_LEN),
            cooldown: 0,
        }
    }

    /// Permanently cap `max_limit` at runtime (e.g. when a consumer GeForce driver hits its
    /// NVENC session limit and we back off and retry cleanly).
    pub fn cap_max_limit(&mut self, new_cap: usize) {
        let clamped = new_cap.max(MIN_CONCURRENT);
        if clamped < self.max_limit {
            self.max_limit = clamped;
        }
        if self.current_max > self.max_limit {
            self.current_max = self.max_limit;
        }
        self.cooldown = DECREASE_COOLDOWN_TICKS;
    }

    /// Decide next action based on current encoder utilization and running count.
    pub fn decide(&mut self, current_util: u8, current_running: usize) -> ConcurrencyAction {
        if current_running < self.current_max {
            return ConcurrencyAction::Maintain;
        }

        // Ignore 0% encoder utilization samples (workers are in ffprobe/extract/XPSNR I/O phase);
        // recording (N, 0%) would cause false marginal_gain=0 lock-in at low concurrency.
        if current_util == 0 {
            return ConcurrencyAction::Maintain;
        }

        if self.history.len() == MAX_HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back((current_running, current_util));

        if current_util > UTILIZATION_HIGH {
            if self.current_max > MIN_CONCURRENT {
                self.current_max -= 1;
                self.cooldown = DECREASE_COOLDOWN_TICKS;
                return ConcurrencyAction::Decrease;
            }
            return ConcurrencyAction::Maintain;
        }

        if self.cooldown > 0 {
            self.cooldown -= 1;
            return ConcurrencyAction::Maintain;
        }

        let marginal_gain = self.calculate_marginal_gain();
        if marginal_gain < MARGINAL_GAIN_THRESHOLD {
            return ConcurrencyAction::Maintain;
        }

        if self.current_max < self.max_limit {
            self.current_max += 1;
            return ConcurrencyAction::Increase;
        }

        ConcurrencyAction::Maintain
    }

    fn calculate_marginal_gain(&self) -> u8 {
        if self.history.len() < 2 {
            return 100;
        }

        let last = *self.history.back().unwrap();
        let prev = self
            .history
            .iter()
            .rev()
            .skip(1)
            .find(|(jobs, _)| *jobs < last.0);

        let Some(&(prev_jobs, prev_util)) = prev else {
            return 100;
        };

        if last.0 <= prev_jobs {
            return 100;
        }

        last.1.saturating_sub(prev_util)
    }

    pub fn current_max(&self) -> usize {
        self.current_max
    }

    pub fn max_limit(&self) -> usize {
        self.max_limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_smi_prefers_encoder_util_and_averages_multi_gpu() {
        assert_eq!(parse_nvidia_smi_utilization("68, 12\n"), 68);
        assert_eq!(parse_nvidia_smi_utilization("[N/A], 45\n"), 45);
        assert_eq!(parse_nvidia_smi_utilization("80, 10\n40, 20\n"), 60);
        assert_eq!(parse_nvidia_smi_utilization(""), 0);
    }

    #[test]
    fn controller_ignores_zero_util_avoids_oscillation_and_respects_session_cap() {
        let mut ctrl = ConcurrencyController::new(2, 6);
        assert_eq!(ctrl.current_max(), 2);

        // Zero util during I/O phase must not stall or corrupt history
        assert_eq!(ctrl.decide(0, 2), ConcurrencyAction::Maintain);
        assert_eq!(ctrl.decide(0, 2), ConcurrencyAction::Maintain);

        // Active at 40% -> increase to 3
        assert_eq!(ctrl.decide(40, 2), ConcurrencyAction::Increase);
        assert_eq!(ctrl.current_max(), 3);

        // Active at 75% (+35% gain) -> increase to 4
        assert_eq!(ctrl.decide(75, 3), ConcurrencyAction::Increase);
        assert_eq!(ctrl.current_max(), 4);

        // Overload at 95% -> decrease to 3 and enter cooldown (no immediate bounce!)
        assert_eq!(ctrl.decide(95, 4), ConcurrencyAction::Decrease);
        assert_eq!(ctrl.current_max(), 3);
        assert_eq!(ctrl.decide(70, 3), ConcurrencyAction::Maintain);

        // Session limit hit -> hard cap max_limit to 2
        ctrl.cap_max_limit(2);
        assert_eq!(ctrl.max_limit(), 2);
        assert_eq!(ctrl.current_max(), 2);
    }
}
