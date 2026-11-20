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
use ffmpeg::{check_ffmpeg, detect_hardware_capabilities, resolve_encoder_type};
use gpu::{DEFAULT_JETSON_HW_CONCURRENT, DEFAULT_MAX_CONCURRENT};
use logging::{LogLevel, log_always, log_error, log_info, set_log_level};
use model::{ConversionStats, MediaItem, MediaStats, ProcessResult};
use scanner::{scan_media_streaming, scan_webp_streaming};
use scheduler::{VideoJobOptions, run_gpu_parallel, run_hw_decoupled_pool, run_video_pool};
use signal::{install_signal_handlers, is_shutdown_requested};
use state::StateCache;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use utils::{format_size, get_image_output_path, get_output_path};

/// Video & image compression tool - hardware-accelerated perceptual H.265 compression & WebP conversion
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory or file path
    #[arg(short, long)]
    input: PathBuf,

    /// Maximum quality knob (CRF/CQ/QP cap) during perceptual quality search
    #[arg(long, default_value = "34")]
    max_crf: u8,

    /// Minimum net size saving percentage required to replace original (output is ALWAYS strictly smaller than input)
    #[arg(long, default_value = "15")]
    min_compression_ratio: u8,

    /// Target perceptual XPSNR quality threshold in dB (higher = closer to visually lossless, recommended 41.0-43.5)
    #[arg(long, default_value = "42.0")]
    target_xpsnr: f64,

    /// Video encoder preset (e.g., fast, medium, slow, or NVENC p1-p7)
    #[arg(long, default_value = "medium")]
    preset: String,

    /// Video encoder type (auto, gpu, jetson, jetson-gst, nvmpi, qsv, vaapi, amf, videotoolbox, cpu)
    #[arg(long, default_value = "auto")]
    encoder: String,

    /// Audio encoding target bitrate (kbps); set to 0 to always copy original audio.
    /// Compatible audio already at or below this bitrate is copied losslessly automatically.
    #[arg(short, long, default_value = "128")]
    audio_bitrate: u32,

    /// Scan only without executing conversion
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// Max concurrent jobs (0 = auto: GPU dynamic up to 8, Jetson/HW 2, CPU 1, images min(cpus, 8))
    #[arg(short, long, default_value = "0")]
    jobs: usize,

    /// Force serial processing (disable parallel encoding & concurrent image pipeline)
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

    /// Force re-encoding without pre-skipping already HEVC/AV1/VP9 or low-BPP videos
    /// (still strictly guarantees output is smaller than input)
    #[arg(short = 'f', long = "force", alias = "reencode-hevc", default_value = "false")]
    force: bool,
}

/// Default cap on image-processing concurrency (NFS I/O-bound sweet spot).
const DEFAULT_MAX_IMAGE_WORKERS: usize = 8;

/// Bounded queue capacity for backpressure between scanner and workers.
const CHANNEL_CAPACITY: usize = 256;

fn resolve_image_workers(jobs_arg: usize, serial: bool, concurrent_with_video: bool) -> usize {
    if serial {
        return 1;
    }
    if jobs_arg > 0 {
        return jobs_arg;
    }
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if concurrent_with_video {
        cpus.saturating_sub(2).clamp(1, DEFAULT_MAX_IMAGE_WORKERS)
    } else {
        cpus.min(DEFAULT_MAX_IMAGE_WORKERS)
    }
}

fn resolve_gpu_max_jobs(jobs_arg: usize) -> usize {
    if jobs_arg > 0 {
        jobs_arg
    } else {
        DEFAULT_MAX_CONCURRENT
    }
}

fn resolve_hw_jobs(jobs_arg: usize, serial: bool) -> usize {
    if serial {
        1
    } else if jobs_arg > 0 {
        jobs_arg
    } else {
        DEFAULT_JETSON_HW_CONCURRENT
    }
}

fn resolve_cpu_video_jobs(jobs_arg: usize, serial: bool) -> usize {
    if serial || jobs_arg == 0 {
        1
    } else {
        jobs_arg
    }
}

fn build_cache_fingerprint(encoder_type: EncoderType, args: &Args) -> String {
    format!(
        "v2|enc={}|xpsnr={:.1}|min={}|maxq={}|preset={}|audio={}|force={}",
        encoder_type.short_id(),
        args.target_xpsnr,
        args.min_compression_ratio,
        args.max_crf,
        args.preset,
        args.audio_bitrate,
        args.force as u8
    )
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

    // Real runtime hardware capability probe (once at startup).
    let capabilities = detect_hardware_capabilities()?;
    let encoder_type = resolve_encoder_type(&args.encoder, &capabilities)?;
    let encoder_config = EncoderConfig {
        encoder_type,
        preset: args.preset.clone(),
        capabilities,
        target_xpsnr: args.target_xpsnr,
    };

    let cache_fingerprint = build_cache_fingerprint(encoder_type, &args);
    let state_cache = match &args.state_file {
        Some(path) => {
            let cache = StateCache::open_with_fingerprint(path, cache_fingerprint)?;
            log_info(&format!(
                "已加载状态缓存: {} ({} 条记录)",
                path.display(),
                cache.len()
            ));
            Some(Arc::new(cache))
        }
        None => None,
    };

    log_info("视频/图片压缩工具 (硬件感知无损压缩模式)");
    log_info(&format!("输入路径: {}", args.input.display()));
    log_info("输出模式: 原子原地替换（视频->10-bit H.265 .mp4 / 图片->.webp，严格不大于原片）");
    log_info(&format!("编码器: {}", encoder_type.display_name()));
    log_info(&format!(
        "感知目标: XPSNR >= {:.1} dB | 质量上限Q: {} | 最小净节省: {}%{}",
        args.target_xpsnr,
        args.max_crf,
        args.min_compression_ratio,
        if args.force { " | 强制不跳过(已HEVC/低BPP也重编码评估)" } else { "" }
    ));
    log_info(&format!(
        "编码预设: {} | 音频目标: {} (兼容低码率自动直拷)",
        args.preset,
        if args.audio_bitrate == 0 {
            "强制直拷".to_string()
        } else {
            format!("{}kbps AAC", args.audio_bitrate)
        }
    ));

    check_ffmpeg(encoder_config.encoder_type, &encoder_config.capabilities)?;

    let mut results = ConversionStats::default();

    if args.dry_run {
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

    let concurrent_images = encoder_type.allows_concurrent_images() && !args.serial;
    let image_workers = resolve_image_workers(args.jobs, args.serial, concurrent_images);

    if concurrent_images {
        log_info("开始单次流式扫描，视频(专用硬件)与图片(CPU)异构并行处理...\n");
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

        if encoder_type == EncoderType::Gpu {
            let video_options = VideoJobOptions {
                encoder_config,
                max_crf: args.max_crf,
                min_compression_ratio: args.min_compression_ratio,
                audio_bitrate: args.audio_bitrate,
                force: args.force,
                max_concurrency: resolve_gpu_max_jobs(args.jobs),
                state_cache,
            };
            run_gpu_parallel(
                video_rx,
                video_discovered,
                &video_options,
                &mut results.video,
            )?;
        } else {
            let video_options = VideoJobOptions {
                encoder_config,
                max_crf: args.max_crf,
                min_compression_ratio: args.min_compression_ratio,
                audio_bitrate: args.audio_bitrate,
                force: args.force,
                max_concurrency: if encoder_type == EncoderType::JetsonNvmpi {
                    1
                } else {
                    resolve_hw_jobs(args.jobs, args.serial)
                },
                state_cache,
            };
            run_hw_decoupled_pool(
                video_rx,
                video_discovered,
                &video_options,
                args.serial,
                &mut results.video,
            )?;
        }

        let _ = scan_handle.join();
        if let Ok(image_stats) = image_handle.join() {
            results.image = image_stats;
        }
    } else {
        let video_jobs = if encoder_type.is_hardware() {
            1
        } else {
            resolve_cpu_video_jobs(args.jobs, args.serial)
        };
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
            force: args.force,
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
