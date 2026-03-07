use crate::config::EncoderConfig;
use crate::gpu::{estimate_initial_concurrency, probe_gpu_impact, ConcurrencyAction, ConcurrencyController, GpuMonitor};
use crate::scanner::scan_videos_streaming;
use crate::utils::format_size;
use crate::video::{process_video, ConversionStats, FileStats, ProcessResult};
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Task execution result
#[derive(Debug)]
pub enum TaskResult {
    Completed { path: PathBuf, stats: FileStats },
    Skipped { path: PathBuf, reason: String },
    Failed { path: PathBuf, error: String },
}

/// Task scheduler (supports dynamic task addition)
pub struct TaskScheduler {
    pending: Mutex<Vec<PathBuf>>,
    running_count: Arc<AtomicUsize>,
    result_sender: Sender<TaskResult>,
    result_receiver: Receiver<TaskResult>,
    encoder_config: EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
    scanning_done: Arc<AtomicBool>,
}

impl TaskScheduler {
    pub fn new(
        encoder_config: EncoderConfig,
        max_crf: u8,
        min_compression_ratio: u8,
        audio_bitrate: u32,
    ) -> Self {
        let (sender, receiver) = unbounded();
        Self {
            pending: Mutex::new(Vec::new()),
            running_count: Arc::new(AtomicUsize::new(0)),
            result_sender: sender,
            result_receiver: receiver,
            encoder_config,
            max_crf,
            min_compression_ratio,
            audio_bitrate,
            scanning_done: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Mark scanning complete
    pub fn mark_scanning_done(&self) {
        self.scanning_done.store(true, Ordering::SeqCst);
    }

    /// Dynamically add task
    pub fn add_task(&self, video_path: PathBuf) {
        self.pending.lock().unwrap().push(video_path);
    }

    /// Get current running task count
    pub fn running_count(&self) -> usize {
        self.running_count.load(Ordering::SeqCst)
    }

    /// Check if there's pending or running work
    pub fn has_pending_or_running(&self) -> bool {
        let pending_empty = self.pending.lock().unwrap().is_empty();
        let running = self.running_count() > 0;
        let done = self.scanning_done.load(Ordering::SeqCst);

        // If scanning not complete, always return true (may have new tasks)
        // If scanning complete, check if there are pending or running tasks
        !done || !pending_empty || running
    }

    /// Try to spawn new tasks up to specified concurrency
    pub fn spawn_up_to(&self, max_concurrent: usize) {
        loop {
            let running = self.running_count();
            if running >= max_concurrent {
                break;
            }

            let video_path = {
                let mut pending = self.pending.lock().unwrap();
                if pending.is_empty() {
                    break;
                }
                pending.remove(0)
            };

            let sender = self.result_sender.clone();
            let running_count = self.running_count.clone();
            let config = self.encoder_config.clone();
            let max_crf = self.max_crf;
            let min_ratio = self.min_compression_ratio;
            let audio_bitrate = self.audio_bitrate;

            running_count.fetch_add(1, Ordering::SeqCst);

            thread::spawn(move || {
                let result = process_video(&video_path, &config, max_crf, min_ratio, audio_bitrate);

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

                running_count.fetch_sub(1, Ordering::SeqCst);
                let _ = sender.send(task_result);
            });
        }
    }

    /// Try to receive completed task result (non-blocking)
    pub fn try_recv_result(&self) -> Option<TaskResult> {
        self.result_receiver.try_recv().ok()
    }

    /// Remaining pending count
    #[allow(dead_code)]
    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }
}

/// GPU parallel processing main function (streaming scan)
pub fn run_gpu_parallel(
    input_dir: &Path,
    encoder_config: &EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
    results: &mut ConversionStats,
) -> Result<()> {
    // Create scheduler and controller
    let scheduler = Arc::new(TaskScheduler::new(
        encoder_config.clone(),
        max_crf,
        min_compression_ratio,
        audio_bitrate,
    ));

    // Start scanning thread
    let scheduler_clone = scheduler.clone();
    let input_dir_clone = input_dir.to_path_buf();
    let video_count = Arc::new(AtomicUsize::new(0));
    let video_count_clone = video_count.clone();

    let scanner_thread = thread::spawn(move || {
        scan_videos_streaming(&input_dir_clone, |path| {
            scheduler_clone.add_task(path);
            video_count_clone.fetch_add(1, Ordering::SeqCst);
        });
        scheduler_clone.mark_scanning_done();
    });

    // Wait for first video for probing
    while scheduler.pending_count() == 0 && !scheduler.scanning_done.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(50));
    }

    // Probe phase: evaluate GPU impact with first video
    let first_video = {
        let pending = scheduler.pending.lock().unwrap();
        pending.first().cloned()
    };

    if first_video.is_none() {
        scanner_thread.join().ok();
        return Ok(());
    }

    println!("=== GPU 并行探测阶段 ===");
    let probe_result = probe_gpu_impact(first_video.as_ref().unwrap(), encoder_config, audio_bitrate)
        .unwrap_or(crate::gpu::ProbeResult { gpu_impact: 30 });

    let initial_concurrency = estimate_initial_concurrency(&probe_result);
    println!("初始并发数: {}", initial_concurrency);
    println!("=== 开始并行处理 ===\n");

    let mut controller = ConcurrencyController::new(initial_concurrency);
    let mut monitor = GpuMonitor::new_for_encoder(encoder_config.encoder_type);

    let mut completed_count = 0;

    // Main loop
    while scheduler.has_pending_or_running() {
        // 1. Collect completed results
        while let Some(task_result) = scheduler.try_recv_result() {
            completed_count += 1;
            let total = video_count.load(Ordering::SeqCst);
            match task_result {
                TaskResult::Completed { path, stats } => {
                    results.successful += 1;
                    results.total_input_size += stats.input_size;
                    results.total_output_size += stats.output_size;
                    println!(
                        "[{}/{}] ✓ {} -> {} ({:.1}% 压缩)",
                        completed_count,
                        total,
                        path.display(),
                        format_size(stats.output_size),
                        (1.0 - stats.output_size as f64 / stats.input_size as f64) * 100.0
                    );
                }
                TaskResult::Skipped { path, reason } => {
                    results.skipped += 1;
                    println!(
                        "[{}/{}] - 跳过: {} ({})",
                        completed_count, total, path.display(), reason
                    );
                }
                TaskResult::Failed { path, error } => {
                    results.failed += 1;
                    println!(
                        "[{}/{}] ✗ 失败: {} ({})",
                        completed_count, total, path.display(), error
                    );
                }
            }
        }

        // 2. Get GPU status and decide concurrency adjustment
        let gpu_util = monitor.get_utilization().unwrap_or(0);
        let running = scheduler.running_count();
        let action = controller.decide(gpu_util, running);

        // 3. Spawn tasks based on decision
        let max_concurrent = controller.current_max();
        scheduler.spawn_up_to(max_concurrent);

        // Print status (only when adjusting)
        if action != ConcurrencyAction::Maintain || running == 0 {
            let total = video_count.load(Ordering::SeqCst);
            println!(
                "  [状态] 运行: {} | 并发上限: {} | GPU: {}% | 已发现: {}",
                running, max_concurrent, gpu_util, total
            );
        }

        // 4. Short sleep to avoid busy waiting
        thread::sleep(Duration::from_millis(500));
    }

    // Ensure scanner thread ends
    scanner_thread.join().ok();

    // Process remaining results (ensure all tasks are counted)
    while let Some(task_result) = scheduler.try_recv_result() {
        completed_count += 1;
        let total = video_count.load(Ordering::SeqCst);
        match task_result {
            TaskResult::Completed { path, stats } => {
                results.successful += 1;
                results.total_input_size += stats.input_size;
                results.total_output_size += stats.output_size;
                println!(
                    "[{}/{}] ✓ {} -> {} ({:.1}% 压缩)",
                    completed_count,
                    total,
                    path.display(),
                    format_size(stats.output_size),
                    (1.0 - stats.output_size as f64 / stats.input_size as f64) * 100.0
                );
            }
            TaskResult::Skipped { path, reason } => {
                results.skipped += 1;
                println!(
                    "[{}/{}] - 跳过: {} ({})",
                    completed_count, total, path.display(), reason
                );
            }
            TaskResult::Failed { path, error } => {
                results.failed += 1;
                println!(
                    "[{}/{}] ✗ 失败: {} ({})",
                    completed_count, total, path.display(), error
                );
            }
        }
    }

    Ok(())
}
