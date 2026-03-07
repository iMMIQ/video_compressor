mod config;
mod encoder;
mod ffmpeg;
mod gpu;
mod scheduler;
mod scanner;
mod utils;
mod video;

use anyhow::Result;
use clap::Parser;
use config::{EncoderConfig, EncoderType};
use crossbeam_channel::unbounded;
use ffmpeg::check_ffmpeg;
use scheduler::run_gpu_parallel;
use scanner::scan_videos_streaming;
use std::path::PathBuf;
use std::thread;
use utils::{format_size, get_output_path};
use video::{process_video, ConversionStats, ProcessResult};

/// Video compression tool - recursively convert videos to H.265 format
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory path
    #[arg(short, long)]
    input: PathBuf,

    /// Maximum CRF value (quality floor), when CRF=23 can't reach compression target,
    /// this is the max CRF limit for extrapolation
    #[arg(long, default_value = "35")]
    max_crf: u8,

    /// Minimum compression ratio (percentage), conversion only proceeds if this target is met
    #[arg(long, default_value = "30")]
    min_compression_ratio: u8,

    /// Video encoder preset
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

    /// Concurrent processing jobs
    #[arg(short, long, default_value = "1")]
    jobs: usize,

    /// Force serial processing (disable GPU parallel encoding)
    #[arg(long, default_value = "false")]
    serial: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate input directory
    if !args.input.exists() {
        anyhow::bail!("输入目录不存在: {}", args.input.display());
    }

    // Parse and resolve encoder type
    let encoder_type = ffmpeg::resolve_encoder_type(&args.encoder)?;
    let encoder_config = EncoderConfig {
        encoder_type,
        preset: args.preset.clone(),
        target_bitrate_kbps: None, // Will be set dynamically for Jetson
        use_2pass: false,
    };
    let encoder_name = match encoder_config.encoder_type {
        EncoderType::Cpu => "CPU (libx265)",
        EncoderType::Gpu => "GPU (hevc_nvenc)",
        EncoderType::Jetson => "Jetson GPU (hevc_nvmpi)",
    };

    println!("视频压缩工具 (智能压缩模式)");
    println!("输入目录: {}", args.input.display());
    println!("输出模式: 原地替换（后缀名改为.mp4）");
    println!("编码器: {}", encoder_name);
    println!("预览CRF: 23 | 最大CRF: {}", args.max_crf);
    println!("最小压缩比: {}%", args.min_compression_ratio);
    println!("编码预设: {}", args.preset);
    println!("音频比特率: {}kbps", args.audio_bitrate);

    // Check FFmpeg availability
    check_ffmpeg(encoder_config.encoder_type)?;

    // Execute conversion
    let mut results = ConversionStats::default();

    if args.dry_run {
        // Dry run mode: stream scan, display only
        println!("扫描视频中...\n");
        let mut count = 0;
        scan_videos_streaming(&args.input, |path| {
            count += 1;
            println!("[{}] 处理: {}", count, path.display());
            let output_path = get_output_path(&path);
            println!("  -> (dry run) {}", output_path.display());
        });
        println!("\n共找到 {} 个视频文件", count);
    } else if matches!(encoder_config.encoder_type, EncoderType::Gpu | EncoderType::Jetson) && !args.serial {
        // GPU mode: use parallel scheduler (streaming scan)
        println!("开始流式扫描和处理...\n");
        run_gpu_parallel(
            &args.input,
            &encoder_config,
            args.max_crf,
            args.min_compression_ratio,
            args.audio_bitrate,
            &mut results,
        )?;
    } else {
        // CPU mode or GPU serial mode: streaming sequential processing
        println!("开始流式扫描和处理...\n");
        let mut count = 0;
        let (sender, receiver) = unbounded::<PathBuf>();

        // Scanner thread
        let input_clone = args.input.clone();
        let scan_thread = thread::spawn(move || {
            scan_videos_streaming(&input_clone, |path| {
                let _ = sender.send(path);
            });
        });

        // Main thread processing
        while let Ok(video_path) = receiver.recv() {
            count += 1;
            println!("[{}] 处理: {}", count, video_path.display());

            match process_video(
                &video_path,
                &encoder_config,
                args.max_crf,
                args.min_compression_ratio,
                args.audio_bitrate,
            ) {
                Ok(ProcessResult::Converted(stats)) => {
                    results.successful += 1;
                    results.total_input_size += stats.input_size;
                    results.total_output_size += stats.output_size;
                    println!(
                        "  ✓ {} -> {} ({:.1}% 压缩)",
                        format_size(stats.input_size),
                        format_size(stats.output_size),
                        (1.0 - stats.output_size as f64 / stats.input_size as f64) * 100.0
                    );
                }
                Ok(ProcessResult::Skipped(reason)) => {
                    results.skipped += 1;
                    println!("  - 跳过: {}", reason);
                }
                Err(e) => {
                    results.failed += 1;
                    println!("  ✗ 失败: {}", e);
                }
            }
        }

        scan_thread.join().ok();
    }

    // Output statistics
    println!("\n{}", "=".repeat(50));
    println!("转换完成!");
    println!(
        "成功: {} | 跳过: {} | 失败: {}",
        results.successful, results.skipped, results.failed
    );
    if results.total_input_size > 0 {
        println!(
            "总大小: {} -> {} ({:.1}% 节省)",
            format_size(results.total_input_size),
            format_size(results.total_output_size),
            (1.0 - results.total_output_size as f64 / results.total_input_size as f64) * 100.0
        );
    }
    println!("{}", "=".repeat(50));

    Ok(())
}
