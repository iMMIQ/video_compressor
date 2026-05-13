use crate::config::{EncoderConfig, EncoderType};
use crate::encoder::preview_encode;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

const MIN_CONCURRENT: usize = 1;
const MAX_CONCURRENT: usize = 8;
const MARGINAL_GAIN_THRESHOLD: u8 = 5;
const UTILIZATION_HIGH: u8 = 95;

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

    /// Get GPU utilization (with cache)
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

    /// Read GPU load from Jetson sysfs (value 0-1000, maps to 0-100%)
    fn read_jetson_gpu_load(&self) -> Result<u8> {
        for path in JETSON_GPU_LOAD_PATHS {
            if let Ok(content) = fs::read_to_string(path) {
                let val: u32 = content.trim().parse().unwrap_or(0);
                // sysfs load is 0-1000 (permille), convert to 0-100
                return Ok((val / 10).min(100) as u8);
            }
        }
        // Fallback: if sysfs not readable, return 50% as safe default
        Ok(50)
    }

    /// Read GPU load from nvidia-smi (desktop GPUs)
    fn read_nvidia_smi_load(&self) -> Result<u8> {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
            .output()
            .context("无法执行 nvidia-smi")?;

        if !output.status.success() {
            anyhow::bail!("nvidia-smi 执行失败");
        }

        let util_str = String::from_utf8_lossy(&output.stdout);
        Ok(util_str.trim().parse::<u8>().unwrap_or(0))
    }

    /// Force refresh cache and get latest value
    pub fn force_refresh(&mut self) -> Result<u8> {
        self.last_check = Instant::now() - Duration::from_secs(10);
        self.get_utilization()
    }
}

/// Concurrency adjustment action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcurrencyAction {
    Increase,
    Decrease,
    Maintain,
}

/// Concurrency controller
pub struct ConcurrencyController {
    current_max: usize,
    history: Vec<(usize, u8)>, // (concurrent jobs, GPU utilization)
}

impl ConcurrencyController {
    pub fn new(initial: usize) -> Self {
        Self {
            current_max: initial.clamp(MIN_CONCURRENT, MAX_CONCURRENT),
            history: Vec::new(),
        }
    }

    /// Decide next action based on current GPU utilization
    pub fn decide(&mut self, current_gpu_util: u8, current_running: usize) -> ConcurrencyAction {
        self.history.push((current_running, current_gpu_util));

        // If running tasks haven't reached current max, fill up first
        if current_running < self.current_max {
            return ConcurrencyAction::Maintain;
        }

        // GPU overloaded, reduce concurrency
        if current_gpu_util > UTILIZATION_HIGH {
            if self.current_max > MIN_CONCURRENT {
                self.current_max -= 1;
            }
            return ConcurrencyAction::Decrease;
        }

        // Calculate marginal gain
        let marginal_gain = self.calculate_marginal_gain();

        if marginal_gain < MARGINAL_GAIN_THRESHOLD {
            // Marginal gain too small, maintain status quo
            return ConcurrencyAction::Maintain;
        }

        // Still room for improvement, try increasing
        if self.current_max < MAX_CONCURRENT {
            self.current_max += 1;
            return ConcurrencyAction::Increase;
        }

        ConcurrencyAction::Maintain
    }

    /// Calculate marginal gain from last two records
    fn calculate_marginal_gain(&self) -> u8 {
        if self.history.len() < 2 {
            return 100; // Insufficient data, assume room for improvement
        }

        let len = self.history.len();
        // Find last two records with different concurrency
        let mut prev_idx = len - 1;
        while prev_idx > 0 && self.history[prev_idx].0 == self.history[len - 1].0 {
            prev_idx -= 1;
        }

        if prev_idx == 0 && self.history[0].0 == self.history[len - 1].0 {
            return 100; // No comparable data
        }

        let (prev_jobs, prev_util) = self.history[prev_idx];
        let (curr_jobs, curr_util) = self.history[len - 1];

        if curr_jobs <= prev_jobs {
            return 100;
        }

        // Gain per additional task
        let gain = curr_util.saturating_sub(prev_util);
        gain
    }

    pub fn current_max(&self) -> usize {
        self.current_max
    }
}

/// Probe result
pub struct ProbeResult {
    #[allow(dead_code)]
    pub gpu_impact: u8, // Single task's impact on GPU utilization
}

/// Probe first video to evaluate single task GPU usage
pub fn probe_gpu_impact(
    first_video: &Path,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
) -> Result<ProbeResult> {
    let mut monitor = GpuMonitor::new_for_encoder(encoder_config.encoder_type);

    // Record baseline
    let baseline_util = monitor.force_refresh()?;
    println!("  探测: GPU 基线利用率 {}%", baseline_util);

    // Execute a brief preview encode
    let start = Instant::now();
    let _ = preview_encode(first_video, 23, encoder_config, audio_bitrate, 5, 0.0, None, None);
    let elapsed = start.elapsed();

    // Get GPU utilization during encoding
    let peak_util = monitor.force_refresh()?;
    let gpu_impact = peak_util.saturating_sub(baseline_util).max(10); // Assume at least 10%

    println!("  探测完成: 耗时 {:?}, GPU 影响 {}%", elapsed, gpu_impact);

    Ok(ProbeResult { gpu_impact })
}

/// Estimate initial concurrency based on probe result
pub fn estimate_initial_concurrency(_probe: &ProbeResult) -> usize {
    // Start with 1 thread, let concurrency controller adjust based on GPU utilization
    1
}
