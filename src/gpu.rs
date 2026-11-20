use crate::config::EncoderType;
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::fs;
use std::process::Command;
use std::time::{Duration, Instant};

pub const MIN_CONCURRENT: usize = 1;
pub const DEFAULT_MAX_CONCURRENT: usize = 8;
const MARGINAL_GAIN_THRESHOLD: u8 = 5;
const UTILIZATION_HIGH: u8 = 90;
const MAX_HISTORY_LEN: usize = 64;

/// Jetson GPU load sysfs paths (tried in order)
const JETSON_GPU_LOAD_PATHS: &[&str] = &[
    "/sys/devices/platform/17000000.gpu/load",
    "/sys/devices/platform/gpu.0/load",
];

/// GPU utilization monitor
pub struct GpuMonitor {
    last_check: Instant,
    cache_duration: Duration,
    cached_utilization: u8,
    is_jetson: bool,
}

impl GpuMonitor {
    pub fn new_for_encoder(encoder_type: EncoderType) -> Self {
        Self {
            last_check: Instant::now() - Duration::from_secs(10),
            cache_duration: Duration::from_secs(2),
            cached_utilization: 0,
            is_jetson: matches!(encoder_type, EncoderType::Jetson),
        }
    }

    /// Get GPU encoder utilization (with 2s cache).
    pub fn get_utilization(&mut self) -> Result<u8> {
        if self.last_check.elapsed() < self.cache_duration {
            return Ok(self.cached_utilization);
        }

        let util = if self.is_jetson {
            self.read_jetson_gpu_load()?
        } else {
            self.read_nvidia_smi_load()?
        };

        self.cached_utilization = util;
        self.last_check = Instant::now();
        Ok(util)
    }

    /// Read GPU load from Jetson sysfs (value 0-1000, maps to 0-100%).
    fn read_jetson_gpu_load(&self) -> Result<u8> {
        for path in JETSON_GPU_LOAD_PATHS {
            if let Ok(content) = fs::read_to_string(path) {
                let val: u32 = content.trim().parse().unwrap_or(0);
                return Ok((val / 10).min(100) as u8);
            }
        }
        Ok(50)
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
/// Prefers encoder utilization; falls back to GPU utilization when encoder metric is unavailable.
pub fn parse_nvidia_smi_utilization(stdout: &str) -> u8 {
    let Some(first_line) = stdout.lines().find(|l| !l.trim().is_empty()) else {
        return 0;
    };
    let mut parts = first_line.split(',').map(|s| s.trim());
    let enc_util = parts.next().and_then(|s| s.parse::<u8>().ok());
    let gpu_util = parts.next().and_then(|s| s.parse::<u8>().ok());

    enc_util.or(gpu_util).unwrap_or(0).min(100)
}

/// Concurrency adjustment action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyAction {
    Increase,
    Decrease,
    Maintain,
}

/// Dynamic concurrency controller with bounded history.
pub struct ConcurrencyController {
    current_max: usize,
    max_limit: usize,
    history: VecDeque<(usize, u8)>, // (concurrent jobs, utilization)
}

impl ConcurrencyController {
    pub fn new(initial: usize, max_limit: usize) -> Self {
        let clamped_limit = max_limit.max(MIN_CONCURRENT);
        Self {
            current_max: initial.clamp(MIN_CONCURRENT, clamped_limit),
            max_limit: clamped_limit,
            history: VecDeque::with_capacity(MAX_HISTORY_LEN),
        }
    }

    /// Decide next action based on current encoder utilization and running count.
    pub fn decide(&mut self, current_util: u8, current_running: usize) -> ConcurrencyAction {
        if self.history.len() == MAX_HISTORY_LEN {
            self.history.pop_front();
        }
        self.history.push_back((current_running, current_util));

        // Wait until running tasks fill current_max before scaling up/down.
        if current_running < self.current_max {
            return ConcurrencyAction::Maintain;
        }

        if current_util > UTILIZATION_HIGH {
            if self.current_max > MIN_CONCURRENT {
                self.current_max -= 1;
                return ConcurrencyAction::Decrease;
            }
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
            .find(|(jobs, _)| *jobs != last.0);

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
    fn parse_nvidia_smi_prefers_encoder_util_and_falls_back() {
        assert_eq!(parse_nvidia_smi_utilization("68, 12\n"), 68);
        assert_eq!(parse_nvidia_smi_utilization("[N/A], 45\n"), 45);
        assert_eq!(parse_nvidia_smi_utilization(""), 0);
    }

    #[test]
    fn controller_ramps_up_backs_off_and_bounds_history() {
        let mut ctrl = ConcurrencyController::new(1, 4);
        assert_eq!(ctrl.current_max(), 1);

        // Running 1 job at 30% -> should increase to 2
        assert_eq!(ctrl.decide(30, 1), ConcurrencyAction::Increase);
        assert_eq!(ctrl.current_max(), 2);

        // Running 1 < current_max(2) -> maintain while filling
        assert_eq!(ctrl.decide(35, 1), ConcurrencyAction::Maintain);

        // Running 2 jobs at 65% (gain +30%) -> increase to 3
        assert_eq!(ctrl.decide(65, 2), ConcurrencyAction::Increase);
        assert_eq!(ctrl.current_max(), 3);

        // Overloaded at 95% -> decrease to 2
        assert_eq!(ctrl.decide(95, 3), ConcurrencyAction::Decrease);
        assert_eq!(ctrl.current_max(), 2);

        // Verify history stays bounded
        for _ in 0..200 {
            ctrl.decide(60, 2);
        }
        assert!(ctrl.history.len() <= MAX_HISTORY_LEN);
    }
}
