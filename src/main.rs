mod config;
mod encoder;
mod ffmpeg;
mod gpu;
mod image;
mod logging;
mod model;
mod scanner;
mod scheduler;
mod signal;
mod state;
mod utils;
mod video;
mod webp_mux;

use anyhow::Result;
use clap::Parser;
use config::{EncoderConfig, EncoderType};
use crossbeam_channel::{Receiver, bounded, unbounded};
use ffmpeg::{check_ffmpeg, detect_available_encoders, resolve_encoder_type};
use gpu::DEFAULT_MAX_CONCURRENT;
use logging::{LogLevel, log_always, log_error, log_info, set_log_level};
use model::{ConversionStats, MediaItem, MediaStats, ProcessResult};
use scanner::{scan_media_streaming, scan_webp_streaming};
use scheduler::{VideoJobOptions, run_gpu_parallel, run_video_pool};
use signal::{install_signal_handlers, is_shutdown_requested};
use state::StateCache;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use utils::{format_size, get_image_output_path, get_output_path};

/// Video & image compression tool - recursively convert videos to H.265 and images to WebP
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory or file path
    #[arg(short, long)]
    input: PathBuf,

    /// Maximum CRF value (quality floor), when CRF=23 can't reach compression target,
    /// this is the max CRF limit for extrapolation
    #[arg(long, default_value = "35")]
    max_crf: u8,

    /// Minimum compression ratio (percentage), conversion only proceeds if this target is met
    #[arg(long, default_value = "30")]
    min_compression_ratio: u8,

    /// Video encoder preset (e.g., fast, medium, slow, or NVENC p1-p7)
    #[arg(long, default_value = "medium")]
    preset: String,

    /// Video encoder type (cpu, gpu, jetson, auto)
    #[arg(long, default_value = "auto")]
    encoder: String,

    /// Audio encoding bitrate (kbps), set to 0 to keep original audio
    #[arg(short, long, default_value = "128")]
    audio_bitrate: u32,

    /// Scan only without executing conversion
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// Max concurrent jobs (0 = auto: GPU up to 8, CPU/Jetson 1, images min(cpus, 8))
    #[arg(short, long, default_value = "0")]
    jobs: usize,

    /// Force serial processing (disable GPU parallel encoding & concurrent image pipeline)
    #[arg(long, default_value = "false")]
    serial: bool,

    /// Repair WebP files produced by the old EXIF-first writer, then exit
    #[arg(long, default_value = "false")]
    repair_webp: bool,

    /// Log verbosity level (quiet, info, debug)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Optional persistent skip-cache file to avoid re-probing unchanged skipped files
    #[arg(long)]
    state_file: Option<PathBuf>,
}

/// Default cap on image-processing concurrency (NFS I/O-bound sweet spot).
const DEFAULT_MAX_IMAGE_WORKERS: usize = 8;

/// Bounded queue capacity for backpressure between scanner and workers.
const CHANNEL_CAPACITY: usize = 256;

fn resolve_image_workers(jobs_arg: usize, serial: bool) -> usize {
    if serial {
        return 1;
    }
    if jobs_arg > 0 {
        return jobs_arg;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(DEFAULT_MAX_IMAGE_WORKERS)
}

fn resolve_gpu_max_jobs(jobs_arg: usize) -> usize {
    if jobs_arg > 0 {
        jobs_arg
    } else {
        DEFAULT_MAX_CONCURRENT
    }
}

fn resolve_cpu_video_jobs(jobs_arg: usize, serial: bool) -> usize {
    if serial || jobs_arg == 0 {
        1
    } else {
        jobs_arg
    }
}

fn repair_webp_files(input: &Path, dry_run: bool) -> Result<()> {
    log_always("WebP 容器恢复模式");
    log_always(&format!("输入路径: {}", input.display()));
    log_always(&format!(
        "执行方式: {}",
        if dry_run { "仅检测" } else { "原位修复" }
    ));

    let mut scanned = 0usize;
    let mut affected = 0usize;
    let mut failed = 0usize;
    scan_webp_streaming(input, |path| {
        if is_shutdown_requested() {
            return;
        }
        scanned += 1;
        match webp_mux::repair_file(&path, dry_run) {
            Ok(webp_mux::RepairResult::NotAffected) => {}
            Ok(webp_mux::RepairResult::WouldRepair { old_size, new_size }) => {
                affected += 1;
                log_info(&format!(
                    "[待修复] {} ({} -> {})",
                    path.display(),
                    format_size(old_size),
                    format_size(new_size)
                ));
            }
            Ok(webp_mux::RepairResult::Repaired { old_size, new_size }) => {
                affected += 1;
                log_info(&format!(
                    "[已修复] {} ({} -> {})",
                    path.display(),
                    format_size(old_size),
                    format_size(new_size)
                ));
            }
            Err(error) => {
                failed += 1;
                log_error(&format!("[失败] {}: {:#}", path.display(), error));
            }
        }
    });

    log_always(&format!(
        "扫描完成: {} 个 WebP，{} {}，{} 个失败",
        scanned,
        affected,
        if dry_run { "个待修复" } else { "个已修复" },
        failed
    ));
    if failed > 0 {
        anyhow::bail!("有 {} 个 WebP 修复失败", failed);
    }
    Ok(())
}

/// Process images from a streaming channel using a fixed worker pool.
fn run_image_pool(
    work_rx: Receiver<PathBuf>,
    num_workers: usize,
    state_cache: Option<Arc<StateCache>>,
) -> MediaStats {
    let mut stats = MediaStats::default();
    let workers_count = num_workers.max(1);
    log_info(&format!("图片处理线程池启动 ({} 线程)", workers_count));

    let (result_tx, result_rx) = unbounded::<(PathBuf, anyhow::Result<ProcessResult>)>();

    let workers: Vec<_> = (0..workers_count)
        .map(|_| {
            let rx = work_rx.clone();
            let tx = result_tx.clone();
            let cache = state_cache.clone();
            thread::spawn(move || {
                while let Ok(path) = rx.recv() {
                    if is_shutdown_requested() {
                        break;
                    }
                    let result = image::process_image(&path, cache.as_deref());
                    let _ = tx.send((path, result));
                }
            })
        })
        .collect();

    drop(work_rx);
    drop(result_tx);

    let mut count = 0usize;
    while let Ok((image_path, process_result)) = result_rx.recv() {
        count += 1;
        match process_result {
            Ok(ProcessResult::Converted(file_stats)) => {
                stats.record_converted(file_stats);
                let ratio = if file_stats.input_size > 0 {
                    (1.0 - file_stats.output_size as f64 / file_stats.input_size as f64) * 100.0
                } else {
                    0.0
                };
                log_info(&format!(
                    "[图片 {}] ✓ {} ({} -> {}, {:.1}% 压缩)",
                    count,
                    image_path.display(),
                    format_size(file_stats.input_size),
                    format_size(file_stats.output_size),
                    ratio
                ));
            }
            Ok(ProcessResult::Skipped(reason)) => {
                stats.record_skipped();
                log_info(&format!(
                    "[图片 {}] - 跳过: {} ({})",
                    count,
                    image_path.display(),
                    reason
                ));
            }
            Err(e) => {
                stats.record_failed();
                log_error(&format!(
                    "[图片 {}] ✗ 失败: {} ({})",
                    count,
                    image_path.display(),
                    e
                ));
            }
        }
    }

    for worker in workers {
        let _ = worker.join();
    }
    stats
}

fn print_category_stats(label: &str, stats: &MediaStats) {
    if stats.total_processed() == 0 {
        return;
    }
    log_always(&format!(
        "{}: 成功 {} | 跳过 {} | 失败 {}",
        label, stats.successful, stats.skipped, stats.failed
    ));
    if stats.total_input_size > 0 {
        log_always(&format!(
            "  体积: {} -> {} ({:.1}% 节省)",
            format_size(stats.total_input_size),
            format_size(stats.total_output_size),
            stats.savings_percent()
        ));
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    set_log_level(LogLevel::parse(&args.log_level)?);
    install_signal_handlers();

    if !args.input.exists() {
        anyhow::bail!("输入目录不存在: {}", args.input.display());
    }

    if args.repair_webp {
        return repair_webp_files(&args.input, args.dry_run);
    }

    let state_cache = match &args.state_file {
        Some(path) => {
            let cache = StateCache::open(path)?;
            log_info(&format!(
                "已加载状态缓存: {} ({} 条记录)",
                path.display(),
                cache.len()
            ));
            Some(Arc::new(cache))
        }
        None => None,
    };

    // Single-call encoder detection
    let available_encoders = detect_available_encoders()?;
    let encoder_type = resolve_encoder_type(&args.encoder, available_encoders)?;
    let encoder_config = EncoderConfig {
        encoder_type,
        preset: args.preset.clone(),
    };
    let encoder_name = match encoder_config.encoder_type {
        EncoderType::Cpu => "CPU (libx265)",
        EncoderType::Gpu => "GPU (hevc_nvenc)",
        EncoderType::Jetson => "Jetson GPU (hevc_nvmpi)",
    };

    log_info("视频/图片压缩工具 (智能压缩模式)");
    log_info(&format!("输入目录: {}", args.input.display()));
    log_info("输出模式: 原子原地替换（视频->.mp4 / 图片->.webp）");
    log_info(&format!("编码器: {}", encoder_name));
    log_info(&format!("预览CRF: 23 | 最大CRF: {}", args.max_crf));
    log_info(&format!("最小压缩比: {}%", args.min_compression_ratio));
    log_info(&format!("编码预设: {}", args.preset));
    log_info(&format!("音频比特率: {}kbps", args.audio_bitrate));

    check_ffmpeg(encoder_config.encoder_type, available_encoders)?;

    let mut results = ConversionStats::default();

    if args.dry_run {
        // Single-pass scan for dry-run
        log_info("单次遍历扫描媒体文件中...\n");
        let mut video_count = 0usize;
        let mut image_count = 0usize;
        scan_media_streaming(&args.input, |item| match item {
            MediaItem::Video(path) => {
                video_count += 1;
                let output_path = get_output_path(&path);
                log_info(&format!(
                    "[视频 {}] {} -> (dry run) {}",
                    video_count,
                    path.display(),
                    output_path.display()
                ));
            }
            MediaItem::Image(path) => {
                image_count += 1;
                let output_path = get_image_output_path(&path);
                log_info(&format!(
                    "[图片 {}] {} -> (dry run) {}",
                    image_count,
                    path.display(),
                    output_path.display()
                ));
            }
        });
        log_always(&format!(
            "\n扫描完成: 共找到 {} 个视频文件, {} 个图片文件",
            video_count, image_count
        ));
        return Ok(());
    }

    let image_workers = resolve_image_workers(args.jobs, args.serial);

    if matches!(encoder_config.encoder_type, EncoderType::Gpu) && !args.serial {
        // GPU parallel mode: single-pass scanner feeds both GPU video scheduler
        // and CPU image worker pool concurrently (heterogeneous hardware utilization).
        log_info("开始单次流式扫描，视频(GPU)与图片(CPU)并行处理...\n");
        let (video_tx, video_rx) = bounded::<PathBuf>(CHANNEL_CAPACITY);
        let (image_tx, image_rx) = bounded::<PathBuf>(CHANNEL_CAPACITY);
        let video_discovered = Arc::new(AtomicUsize::new(0));
        let video_discovered_scan = video_discovered.clone();

        let input_clone = args.input.clone();
        let scan_handle = thread::spawn(move || {
            scan_media_streaming(&input_clone, |item| {
                if is_shutdown_requested() {
                    return;
                }
                match item {
                    MediaItem::Video(path) => {
                        video_discovered_scan.fetch_add(1, Ordering::Relaxed);
                        let _ = video_tx.send(path);
                    }
                    MediaItem::Image(path) => {
                        let _ = image_tx.send(path);
                    }
                }
            });
        });

        let img_cache = state_cache.clone();
        let image_handle =
            thread::spawn(move || run_image_pool(image_rx, image_workers, img_cache));

        let video_options = VideoJobOptions {
            encoder_config,
            max_crf: args.max_crf,
            min_compression_ratio: args.min_compression_ratio,
            audio_bitrate: args.audio_bitrate,
            max_concurrency: resolve_gpu_max_jobs(args.jobs),
            state_cache,
        };
        run_gpu_parallel(
            video_rx,
            video_discovered,
            &video_options,
            &mut results.video,
        )?;

        let _ = scan_handle.join();
        if let Ok(image_stats) = image_handle.join() {
            results.image = image_stats;
        }
    } else {
        // CPU / Jetson / serial mode: single-pass scanner streams videos immediately
        // and collects image paths (avoiding a second WalkDir traversal and avoiding
        // CPU contention between libx265 and libwebp).
        let video_jobs = resolve_cpu_video_jobs(args.jobs, args.serial);
        log_info(&format!(
            "开始单次流式扫描和处理 (视频并发: {}, 图片并发: {})...\n",
            video_jobs, image_workers
        ));

        let (video_tx, video_rx) = bounded::<PathBuf>(CHANNEL_CAPACITY);
        let video_discovered = Arc::new(AtomicUsize::new(0));
        let video_discovered_scan = video_discovered.clone();

        let input_clone = args.input.clone();
        let scan_handle = thread::spawn(move || {
            let mut deferred_images = Vec::new();
            scan_media_streaming(&input_clone, |item| {
                if is_shutdown_requested() {
                    return;
                }
                match item {
                    MediaItem::Video(path) => {
                        video_discovered_scan.fetch_add(1, Ordering::Relaxed);
                        let _ = video_tx.send(path);
                    }
                    MediaItem::Image(path) => {
                        deferred_images.push(path);
                    }
                }
            });
            deferred_images
        });

        let video_options = VideoJobOptions {
            encoder_config,
            max_crf: args.max_crf,
            min_compression_ratio: args.min_compression_ratio,
            audio_bitrate: args.audio_bitrate,
            max_concurrency: video_jobs,
            state_cache: state_cache.clone(),
        };
        run_video_pool(
            video_rx,
            video_discovered,
            &video_options,
            &mut results.video,
        )?;

        let deferred_images = scan_handle.join().unwrap_or_default();
        if !deferred_images.is_empty() && !is_shutdown_requested() {
            log_info(&format!(
                "\n=== 图片转换 ({} 个文件) ===",
                deferred_images.len()
            ));
            let (img_tx, img_rx) = bounded::<PathBuf>(CHANNEL_CAPACITY);
            let feeder = thread::spawn(move || {
                for path in deferred_images {
                    if is_shutdown_requested() || img_tx.send(path).is_err() {
                        break;
                    }
                }
            });
            results.image = run_image_pool(img_rx, image_workers, state_cache);
            let _ = feeder.join();
        }
    }

    if is_shutdown_requested() {
        log_always("\n收到中断信号，已优雅停止。");
    }

    // Output separated and combined statistics
    let combined = results.combined();
    log_always(&format!("\n{}", "=".repeat(50)));
    log_always("转换完成!");
    print_category_stats("视频", &results.video);
    print_category_stats("图片", &results.image);
    log_always(&format!(
        "总计: 成功 {} | 跳过 {} | 失败 {}",
        combined.successful, combined.skipped, combined.failed
    ));
    if combined.total_input_size > 0 {
        log_always(&format!(
            "总体积: {} -> {} ({:.1}% 节省)",
            format_size(combined.total_input_size),
            format_size(combined.total_output_size),
            combined.savings_percent()
        ));
    }
    log_always(&"=".repeat(50));

    Ok(())
}
