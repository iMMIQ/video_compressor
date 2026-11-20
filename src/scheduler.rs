use crate::config::EncoderConfig;
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
                "[视频 {}/{}] ✓ {} -> {} ({:.1}% 压缩)",
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

/// GPU dynamic-concurrency scheduler backed by a fixed worker pool and event-driven loop.
/// Replaces unbounded `Vec::remove(0)` + per-task `thread::spawn` + `sleep(500ms)` polling.
pub fn run_gpu_parallel(
    video_rx: Receiver<PathBuf>,
    discovered_count: Arc<AtomicUsize>,
    options: &VideoJobOptions,
    video_stats: &mut MediaStats,
) -> Result<()> {
    let mut controller = ConcurrencyController::new(1, options.max_concurrency);
    let mut monitor = GpuMonitor::new_for_encoder(options.encoder_config.encoder_type);
    let pool_size = controller.max_limit();

    log_info(&format!(
        "=== 开始 GPU 并行视频处理 (初始并发: {}, 最大并发: {}) ===\n",
        controller.current_max(),
        pool_size
    ));

    let (dispatch_tx, dispatch_rx) = bounded::<PathBuf>(pool_size);
    let (result_tx, result_rx) = unbounded::<TaskResult>();

    let mut workers = Vec::with_capacity(pool_size);
    for _ in 0..pool_size {
        let rx = dispatch_rx.clone();
        let tx = result_tx.clone();
        let opts = options.clone();
        workers.push(thread::spawn(move || {
            while let Ok(video_path) = rx.recv() {
                if is_shutdown_requested() {
                    let _ = tx.send(TaskResult::Skipped {
                        path: video_path,
                        reason: "收到中断信号".to_string(),
                    });
                    continue;
                }
                let result = process_video(
                    &video_path,
                    &opts.encoder_config,
                    opts.max_crf,
                    opts.min_compression_ratio,
                    opts.audio_bitrate,
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
    drop(dispatch_rx);
    drop(result_tx);

    let mut pending: VecDeque<PathBuf> = VecDeque::new();
    let mut scanner_closed = false;
    let mut in_flight: usize = 0;
    let mut completed_count: usize = 0;

    loop {
        // 1. Pull newly discovered videos into local VecDeque (bounded buffer for backpressure).
        if !scanner_closed && !is_shutdown_requested() {
            while pending.len() < 64 {
                match video_rx.try_recv() {
                    Ok(path) => pending.push_back(path),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        scanner_closed = true;
                        break;
                    }
                }
            }
        }

        // 2. Dispatch pending tasks up to current dynamic concurrency limit.
        if !is_shutdown_requested() {
            while in_flight < controller.current_max() {
                let Some(path) = pending.pop_front() else {
                    break;
                };
                if dispatch_tx.send(path).is_ok() {
                    in_flight += 1;
                }
            }
        } else {
            pending.clear();
        }

        // 3. Check termination condition.
        if in_flight == 0 && (is_shutdown_requested() || (scanner_closed && pending.is_empty())) {
            break;
        }

        // 4. Event-driven wait: wake immediately on task completion, or tick every 250ms
        //    to evaluate encoder utilization and adjust concurrency.
        match result_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(first_result) => {
                in_flight = in_flight.saturating_sub(1);
                completed_count += 1;
                record_and_log_video_result(
                    completed_count,
                    discovered_count.load(Ordering::Relaxed),
                    first_result,
                    video_stats,
                );
                // Drain any additional ready results without waiting.
                while let Ok(res) = result_rx.try_recv() {
                    in_flight = in_flight.saturating_sub(1);
                    completed_count += 1;
                    record_and_log_video_result(
                        completed_count,
                        discovered_count.load(Ordering::Relaxed),
                        res,
                        video_stats,
                    );
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }

        // 5. Update concurrency controller from live GPU encoder utilization.
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

/// Fixed-size worker pool for CPU / Jetson / serial video processing.
/// Respects `--jobs` (1 when serial or default CPU/Jetson).
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
