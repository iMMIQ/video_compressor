use crate::config::EncoderConfig;
use crate::encoder::{HwSlotPool, is_nvenc_session_limit_error};
use crate::gpu::{ConcurrencyAction, ConcurrencyController, GpuMonitor};
use crate::logging::{log_error, log_info};
use crate::model::{FileStats, MediaStats, ProcessResult};
use crate::signal::is_shutdown_requested;
use crate::state::StateCache;
use crate::utils::format_size;
use crate::video::process_video;
use anyhow::Result;
use crossbeam_channel::{Receiver, TryRecvError, bounded, unbounded};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

/// Result emitted by a video worker.
#[derive(Debug)]
enum TaskResult {
    Completed { path: PathBuf, stats: FileStats },
    Skipped { path: PathBuf, reason: String },
    SessionLimitRetry { path: PathBuf },
    Failed { path: PathBuf, error: String },
}

fn record_and_log_video_result(
    completed_count: usize,
    total_discovered: usize,
    task_result: TaskResult,
    stats: &mut MediaStats,
) {
    match task_result {
        TaskResult::Completed {
            path,
            stats: file_stats,
        } => {
            stats.record_converted(file_stats);
            let ratio = if file_stats.input_size > 0 {
                (1.0 - file_stats.output_size as f64 / file_stats.input_size as f64) * 100.0
            } else {
                0.0
            };
            log_info(&format!(
                "[视频 {}/{}] ✓ {} -> {} ({:.1}% 节省)",
                completed_count,
                total_discovered,
                path.display(),
                format_size(file_stats.output_size),
                ratio
            ));
        }
        TaskResult::Skipped { path, reason } => {
            stats.record_skipped();
            log_info(&format!(
                "[视频 {}/{}] - 跳过: {} ({})",
                completed_count,
                total_discovered,
                path.display(),
                reason
            ));
        }
        TaskResult::Failed { path, error } => {
            stats.record_failed();
            log_error(&format!(
                "[视频 {}/{}] ✗ 失败: {} ({})",
                completed_count,
                total_discovered,
                path.display(),
                error
            ));
        }
        TaskResult::SessionLimitRetry { .. } => {}
    }
}

/// Bundled options for video processing workers.
#[derive(Clone)]
pub struct VideoJobOptions {
    pub encoder_config: EncoderConfig,
    pub max_crf: u8,
    pub min_compression_ratio: u8,
    pub audio_bitrate: u32,
    pub max_concurrency: usize,
    pub state_cache: Option<Arc<StateCache>>,
}

/// Desktop NVIDIA NVENC dynamic-concurrency scheduler:
/// - Starts at concurrency 2 (capped by max_concurrency)
/// - Distributes work round-robin across multiple GPUs when present
/// - Detects consumer NVENC session-limit errors, backs off concurrency, and retries cleanly
pub fn run_gpu_parallel(
    video_rx: Receiver<PathBuf>,
    discovered_count: Arc<AtomicUsize>,
    options: &VideoJobOptions,
    video_stats: &mut MediaStats,
) -> Result<()> {
    let gpu_count = options.encoder_config.capabilities.nvenc_gpu_count.max(1);
    let initial = 2usize.min(options.max_concurrency).max(1);
    let mut controller = ConcurrencyController::new(initial, options.max_concurrency);
    let mut monitor = GpuMonitor::new();
    let pool_size = controller.max_limit();

    log_info(&format!(
        "=== 开始 GPU 并行视频处理 (GPU数: {} | 初始并发: {}, 最大并发: {}) ===\n",
        gpu_count,
        controller.current_max(),
        pool_size
    ));

    let (dispatch_tx, dispatch_rx) = bounded::<(PathBuf, usize, u8)>(pool_size);
    let (result_tx, result_rx) = unbounded::<(TaskResult, u8)>();

    let mut workers = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let rx = dispatch_rx.clone();
        let tx = result_tx.clone();
        let opts = options.clone();
        workers.push(thread::spawn(move || {
            while let Ok((video_path, gpu_idx, retries)) = rx.recv() {
                if is_shutdown_requested() {
                    let _ = tx.send((
                        TaskResult::Skipped {
                            path: video_path,
                            reason: "收到中断信号".to_string(),
                        },
                        retries,
                    ));
                    continue;
                }
                let result = process_video(
                    &video_path,
                    &opts.encoder_config,
                    opts.max_crf,
                    opts.min_compression_ratio,
                    opts.audio_bitrate,
                    Some(gpu_idx),
                    None,
                    opts.state_cache.as_deref(),
                );
                let task_result = match result {
                    Ok(ProcessResult::Converted(stats)) => TaskResult::Completed {
                        path: video_path,
                        stats,
                    },
                    Ok(ProcessResult::Skipped(reason)) => TaskResult::Skipped {
                        path: video_path,
                        reason,
                    },
                    Err(e) => {
                        let msg = e.to_string();
                        if retries < 2 && is_nvenc_session_limit_error(&msg) {
                            TaskResult::SessionLimitRetry { path: video_path }
                        } else {
                            TaskResult::Failed {
                                path: video_path,
                                error: msg,
                            }
                        }
                    }
                };
                let _ = tx.send((task_result, retries));
            }
        }));
    }
    drop(dispatch_rx);
    drop(result_tx);

    let mut pending: VecDeque<(PathBuf, u8)> = VecDeque::new();
    let mut scanner_closed = false;
    let mut in_flight: usize = 0;
    let mut completed_count: usize = 0;
    let mut dispatch_seq: usize = 0;

    let handle_result = |task_result: TaskResult,
                         retries: u8,
                         in_flight: &mut usize,
                         completed_count: &mut usize,
                         controller: &mut ConcurrencyController,
                         pending: &mut VecDeque<(PathBuf, u8)>,
                         video_stats: &mut MediaStats| {
        *in_flight = in_flight.saturating_sub(1);
        match task_result {
            TaskResult::SessionLimitRetry { path } => {
                let new_cap = (*in_flight).max(1);
                controller.cap_max_limit(new_cap);
                log_info(&format!(
                    "  [GPU调度] 检测到NVENC会话上限，自动降并发至 {} 并重试: {}",
                    controller.current_max(),
                    path.display()
                ));
                pending.push_front((path, retries + 1));
            }
            other => {
                *completed_count += 1;
                record_and_log_video_result(
                    *completed_count,
                    discovered_count.load(Ordering::Relaxed),
                    other,
                    video_stats,
                );
            }
        }
    };

    loop {
        if !scanner_closed && !is_shutdown_requested() {
            while pending.len() < 64 {
                match video_rx.try_recv() {
                    Ok(path) => pending.push_back((path, 0)),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        scanner_closed = true;
                        break;
                    }
                }
            }
        }

        if !is_shutdown_requested() {
            while in_flight < controller.current_max() {
                let Some((path, retries)) = pending.pop_front() else {
                    break;
                };
                let gpu_idx = dispatch_seq % gpu_count;
                dispatch_seq = dispatch_seq.wrapping_add(1);
                if dispatch_tx.send((path, gpu_idx, retries)).is_ok() {
                    in_flight += 1;
                }
            }
        } else {
            pending.clear();
        }

        if in_flight == 0 && (is_shutdown_requested() || (scanner_closed && pending.is_empty())) {
            break;
        }

        match result_rx.recv_timeout(Duration::from_millis(250)) {
            Ok((first_result, retries)) => {
                handle_result(
                    first_result,
                    retries,
                    &mut in_flight,
                    &mut completed_count,
                    &mut controller,
                    &mut pending,
                    video_stats,
                );
                while let Ok((res, r)) = result_rx.try_recv() {
                    handle_result(
                        res,
                        r,
                        &mut in_flight,
                        &mut completed_count,
                        &mut controller,
                        &mut pending,
                        video_stats,
                    );
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }

        if !is_shutdown_requested() && in_flight > 0 {
            let gpu_util = monitor.get_utilization().unwrap_or(0);
            let action = controller.decide(gpu_util, in_flight);
            if action != ConcurrencyAction::Maintain {
                let total = discovered_count.load(Ordering::Relaxed);
                log_info(&format!(
                    "  [GPU调度] 运行: {} | 并发上限: {} | 编码器利用率: {}% | 已发现: {}",
                    in_flight,
                    controller.current_max(),
                    gpu_util,
                    total
                ));
            }
        }
    }

    drop(dispatch_tx);
    for worker in workers {
        let _ = worker.join();
    }

    Ok(())
}

/// Hardware-slot decoupled worker pool for Jetson (official GStreamer nvv4l2), QSV, VAAPI, AMF, VideoToolbox.
/// Spawns extra I/O+XPSNR workers (`hw_slots + 2`) while bounding concurrent hardware encodes strictly
/// to `hw_slots` via `HwSlotPool`, keeping the hardware encoder continuously fed with zero I/O bubbles!
pub fn run_hw_decoupled_pool(
    video_rx: Receiver<PathBuf>,
    discovered_count: Arc<AtomicUsize>,
    options: &VideoJobOptions,
    serial: bool,
    video_stats: &mut MediaStats,
) -> Result<()> {
    let hw_slots_count = if serial {
        1
    } else {
        options.max_concurrency.max(1)
    };
    let workers_count = if serial {
        1
    } else {
        (hw_slots_count + 2).min(hw_slots_count.saturating_mul(2).max(2))
    };

    log_info(&format!(
        "=== 开始硬件解耦流水线 (硬件编码槽: {} | 预取/评测工作线程: {}) ===\n",
        hw_slots_count, workers_count
    ));

    let hw_slots = Arc::new(HwSlotPool::new(hw_slots_count));
    let (result_tx, result_rx) = unbounded::<TaskResult>();

    let mut workers = Vec::with_capacity(workers_count);
    for _ in 0..workers_count {
        let rx = video_rx.clone();
        let tx = result_tx.clone();
        let opts = options.clone();
        let slots = hw_slots.clone();
        workers.push(thread::spawn(move || {
            while let Ok(video_path) = rx.recv() {
                if is_shutdown_requested() {
                    break;
                }
                let result = process_video(
                    &video_path,
                    &opts.encoder_config,
                    opts.max_crf,
                    opts.min_compression_ratio,
                    opts.audio_bitrate,
                    None,
                    Some(&slots),
                    opts.state_cache.as_deref(),
                );
                let task_result = match result {
                    Ok(ProcessResult::Converted(stats)) => TaskResult::Completed {
                        path: video_path,
                        stats,
                    },
                    Ok(ProcessResult::Skipped(reason)) => TaskResult::Skipped {
                        path: video_path,
                        reason,
                    },
                    Err(e) => TaskResult::Failed {
                        path: video_path,
                        error: e.to_string(),
                    },
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
        record_and_log_video_result(
            completed,
            discovered_count.load(Ordering::Relaxed),
            task_result,
            video_stats,
        );
    }

    for worker in workers {
        let _ = worker.join();
    }

    Ok(())
}

/// Fixed-size worker pool for CPU / legacy NVMPI / serial video processing.
pub fn run_video_pool(
    video_rx: Receiver<PathBuf>,
    discovered_count: Arc<AtomicUsize>,
    options: &VideoJobOptions,
    video_stats: &mut MediaStats,
) -> Result<()> {
    let workers_count = options.max_concurrency.max(1);
    let (result_tx, result_rx) = unbounded::<TaskResult>();

    let mut workers = Vec::with_capacity(workers_count);
    for _ in 0..workers_count {
        let rx = video_rx.clone();
        let tx = result_tx.clone();
        let opts = options.clone();
        workers.push(thread::spawn(move || {
            while let Ok(video_path) = rx.recv() {
                if is_shutdown_requested() {
                    break;
                }
                let result = process_video(
                    &video_path,
                    &opts.encoder_config,
                    opts.max_crf,
                    opts.min_compression_ratio,
                    opts.audio_bitrate,
                    None,
                    None,
                    opts.state_cache.as_deref(),
                );
                let task_result = match result {
                    Ok(ProcessResult::Converted(stats)) => TaskResult::Completed {
                        path: video_path,
                        stats,
                    },
                    Ok(ProcessResult::Skipped(reason)) => TaskResult::Skipped {
                        path: video_path,
                        reason,
                    },
                    Err(e) => TaskResult::Failed {
                        path: video_path,
                        error: e.to_string(),
                    },
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
        record_and_log_video_result(
            completed,
            discovered_count.load(Ordering::Relaxed),
            task_result,
            video_stats,
        );
    }

    for worker in workers {
        let _ = worker.join();
    }

    Ok(())
}
