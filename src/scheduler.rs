use crate::config::{EncoderType, VideoConfig};
use crate::encoder::{HwSlotPool, is_nvenc_session_limit_error};
use crate::logging::{log_error, log_info};
use crate::model::{FileStats, MediaStats, ProcessResult};
use crate::signal::is_shutdown_requested;
use crate::state::StateCache;
use crate::utils::format_size;
use crate::video::process_video;
use anyhow::Result;
use crossbeam_channel::{Receiver, unbounded};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

#[derive(Debug)]
enum TaskResult {
    Completed { path: PathBuf, stats: FileStats },
    Skipped { path: PathBuf, reason: String },
    Failed { path: PathBuf, error: String },
}

/// Bundled options for the unified video processing pool.
#[derive(Clone)]
pub struct VideoJobOptions {
    pub video: VideoConfig,
    pub max_concurrency: usize,
    pub serial: bool,
    pub state_cache: Option<Arc<StateCache>>,
}

/// Unified video worker pool supporting all backends:
/// - **Hardware (`!serial`)**: decoupled `HwSlotPool` (`hw_slots` active hardware encodes + extra
///   prefetch/XPSNR workers to keep hardware silicon continuously fed with zero I/O bubbles).
/// - **Desktop NVENC**: round-robin multi-GPU dispatch + automatic session-limit slot shrinking
///   and retry (up to 2 attempts).
/// - **CPU / `--serial`**: direct fixed-size worker pool without slot decoupling.
pub fn run_video_pool(
    video_rx: Receiver<PathBuf>,
    discovered_count: Arc<AtomicUsize>,
    options: &VideoJobOptions,
    video_stats: &mut MediaStats,
) -> Result<()> {
    let encoder_type = options.video.encoder.encoder_type;
    let is_nvenc = encoder_type == EncoderType::Gpu;
    let use_hw_slots = encoder_type.is_hardware() && !options.serial;

    let hw_slots_count = if options.serial {
        1
    } else {
        options.max_concurrency.max(1)
    };
    let workers_count = if use_hw_slots {
        (hw_slots_count + 2).min(hw_slots_count.saturating_mul(2).max(2))
    } else {
        hw_slots_count
    };

    let gpu_count = options.video.encoder.capabilities.nvenc_gpu_count.max(1);
    if use_hw_slots {
        if is_nvenc {
            log_info(&format!(
                "=== 开始 GPU 硬件解耦流水线 (GPU数: {} | 硬件编码槽: {} | 工作线程: {}) ===\n",
                gpu_count, hw_slots_count, workers_count
            ));
        } else {
            log_info(&format!(
                "=== 开始硬件解耦流水线 (硬件编码槽: {} | 预取/评测工作线程: {}) ===\n",
                hw_slots_count, workers_count
            ));
        }
    }

    let hw_slots = use_hw_slots.then(|| Arc::new(HwSlotPool::new(hw_slots_count)));
    let next_gpu = Arc::new(AtomicUsize::new(0));
    let (result_tx, result_rx) = unbounded::<TaskResult>();

    let mut workers = Vec::with_capacity(workers_count);
    for _ in 0..workers_count {
        let rx = video_rx.clone();
        let tx = result_tx.clone();
        let opts = options.clone();
        let slots = hw_slots.clone();
        let gpu_rr = next_gpu.clone();

        workers.push(thread::spawn(move || {
            while let Ok(video_path) = rx.recv() {
                if is_shutdown_requested() {
                    break;
                }
                let mut retries = 0u8;
                let task_result = loop {
                    let observed_limit = slots.as_deref().map(HwSlotPool::limit).unwrap_or(1);
                    let gpu_idx = is_nvenc
                        .then(|| gpu_rr.fetch_add(1, Ordering::Relaxed) % gpu_count);
                    match process_video(
                        &video_path,
                        &opts.video,
                        gpu_idx,
                        slots.as_deref(),
                        opts.state_cache.as_deref(),
                    ) {
                        Ok(ProcessResult::Converted(stats)) => {
                            break TaskResult::Completed {
                                path: video_path,
                                stats,
                            };
                        }
                        Ok(ProcessResult::Skipped(reason)) => {
                            break TaskResult::Skipped {
                                path: video_path,
                                reason,
                            };
                        }
                        Err(e) => {
                            let msg = e.to_string();
                            if is_nvenc
                                && retries < 2
                                && !is_shutdown_requested()
                                && is_nvenc_session_limit_error(&msg)
                            {
                                retries += 1;
                                if let Some(pool) = slots.as_deref() {
                                    let new_cap = pool.shrink_below(observed_limit);
                                    log_info(&format!(
                                        "  [GPU调度] 检测到NVENC会话上限，自动降并发至 {} 并重试 ({}/2): {}",
                                        new_cap,
                                        retries,
                                        video_path.display()
                                    ));
                                }
                                continue;
                            }
                            break TaskResult::Failed {
                                path: video_path,
                                error: msg,
                            };
                        }
                    }
                };
                let _ = tx.send(task_result);
            }
        }));
    }
    drop(video_rx);
    drop(result_tx);

    let mut completed = 0usize;
    while let Ok(task_result) = result_rx.recv() {
        completed += 1;
        let total = discovered_count.load(Ordering::Relaxed);
        match task_result {
            TaskResult::Completed { path, stats } => {
                video_stats.record_converted(stats);
                log_info(&format!(
                    "[视频 {}/{}] ✓ {} -> {} ({:.1}% 节省)",
                    completed,
                    total,
                    path.display(),
                    format_size(stats.output_size),
                    stats.savings_percent()
                ));
            }
            TaskResult::Skipped { path, reason } => {
                video_stats.record_skipped();
                log_info(&format!(
                    "[视频 {}/{}] - 跳过: {} ({})",
                    completed,
                    total,
                    path.display(),
                    reason
                ));
            }
            TaskResult::Failed { path, error } => {
                video_stats.record_failed();
                log_error(&format!(
                    "[视频 {}/{}] ✗ 失败: {} ({})",
                    completed,
                    total,
                    path.display(),
                    error
                ));
            }
        }
    }

    for worker in workers {
        let _ = worker.join();
    }
    Ok(())
}
