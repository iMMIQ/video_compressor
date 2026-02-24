use anyhow::{Context, Result};
use clap::Parser;
use filetime::{set_file_times, FileTime};
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::NamedTempFile;
use walkdir::WalkDir;

/// Encoder type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderType {
    Cpu,
    Gpu,
    Auto,
}

/// Encoder configuration
#[derive(Debug, Clone)]
struct EncoderConfig {
    encoder_type: EncoderType,
    preset: String,
}

impl EncoderType {
    /// Parse from string
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            _ => anyhow::bail!("Invalid encoder type: {}. Use: cpu, gpu, or auto", s),
        }
    }

    /// Resolve Auto to Cpu or Gpu based on NVENC availability
    fn resolve(self) -> Result<Self> {
        match self {
            Self::Auto => {
                if check_nvenc_available()? {
                    Ok(Self::Gpu)
                } else {
                    Ok(Self::Cpu)
                }
            }
            other => Ok(other),
        }
    }
}

/// 视频压缩工具 - 将视频递归转换为H.265格式以减小体积
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 输入目录路径
    #[arg(short, long)]
    input: PathBuf,

    /// 最大CRF值（质量下限），当CRF=23无法达到压缩目标时，外推的最大CRF限制
    #[arg(long, default_value = "35")]
    max_crf: u8,

    /// 最小压缩比（百分比），必须达到此压缩比才会执行转换
    #[arg(long, default_value = "30")]
    min_compression_ratio: u8,

    /// 视频编码器预设
    #[arg(long, default_value = "medium")]
    preset: String,

    /// 视频编码器类型 (cpu, gpu, auto)
    #[arg(long, default_value = "auto")]
    encoder: String,

    /// 音频编码比特率（kbps），设为0保持原音频
    #[arg(short, long, default_value = "128")]
    audio_bitrate: u32,

    /// 仅扫描不执行转换
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// 并发处理任务数
    #[arg(short, long, default_value = "1")]
    jobs: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 验证输入目录
    if !args.input.exists() {
        anyhow::bail!("输入目录不存在: {}", args.input.display());
    }

    // 解析编码器类型
    let encoder_type = EncoderType::from_str(&args.encoder)?;
    let encoder_type = encoder_type.resolve()?;
    let encoder_config = EncoderConfig {
        encoder_type,
        preset: args.preset.clone(),
    };
    let encoder_name = match encoder_config.encoder_type {
        EncoderType::Cpu => "CPU (libx265)",
        EncoderType::Gpu => "GPU (hevc_nvenc)",
        EncoderType::Auto => unreachable!("Auto should be resolved"),
    };

    println!("视频压缩工具 (智能压缩模式)");
    println!("输入目录: {}", args.input.display());
    println!("输出模式: 原地替换（后缀名改为.mp4）");
    println!("编码器: {}", encoder_name);
    println!("预览CRF: 23 | 最大CRF: {}", args.max_crf);
    println!("最小压缩比: {}%", args.min_compression_ratio);
    println!("编码预设: {}", args.preset);
    println!("音频比特率: {}kbps", args.audio_bitrate);
    println!("扫描视频中...\n");

    // 检查FFmpeg是否可用
    check_ffmpeg(encoder_config.encoder_type)?;

    // 扫描视频文件
    let video_files = scan_videos(&args.input)?;
    println!("找到 {} 个视频文件\n", video_files.len());

    if video_files.is_empty() {
        println!("未找到任何视频文件。");
        return Ok(());
    }

    // 执行转换
    let mut results = ConversionStats::default();
    for (i, video_path) in video_files.iter().enumerate() {
        println!("[{}/{}] 处理: {}", i + 1, video_files.len(), video_path.display());

        if args.dry_run {
            // 生成输出路径用于显示
            let output_path = get_output_path(video_path);
            println!("  -> (dry run) {}", output_path.display());
            continue;
        }

        match process_video(
            video_path,
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

    // 输出统计信息
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

#[derive(Debug)]
enum ProcessResult {
    Converted(FileStats),
    Skipped(String),
}

#[derive(Debug, Default)]
struct FileStats {
    input_size: u64,
    output_size: u64,
    #[allow(dead_code)]
    used_crf: u8,
}

#[derive(Debug, Default)]
struct ConversionStats {
    successful: usize,
    skipped: usize,
    failed: usize,
    total_input_size: u64,
    total_output_size: u64,
}

/// 生成输出路径：与输入文件同目录，后缀名改为.mp4
fn get_output_path(input_path: &Path) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or(std::ffi::OsStr::new(""));
    let parent = input_path.parent().unwrap_or(Path::new(""));
    parent.join(format!("{}.mp4", stem.to_string_lossy()))
}

/// 跨文件系统安全的文件移动
/// 先尝试rename，失败则使用copy+remove
fn safe_move_file(from: &Path, to: &Path) -> Result<()> {
    // 先尝试直接重命名（同文件系统最快）
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }

    // rename失败（可能是跨文件系统），使用copy+remove
    std::fs::copy(from, to)
        .context(format!("无法复制文件从 {} 到 {}", from.display(), to.display()))?;
    std::fs::remove_file(from)
        .context(format!("无法删除源文件 {}", from.display()))?;

    Ok(())
}

fn process_video(
    input_path: &Path,
    max_crf: u8,
    min_compression_ratio: u8,
    preset: &str,
    audio_bitrate: u32,
) -> Result<ProcessResult> {
    const PREVIEW_CRF: u8 = 23;  // 固定预览CRF
    const PREVIEW_RATIO: f64 = 0.10;  // 预览时长为视频的10%
    const MIN_PREVIEW_SECONDS: u32 = 5;  // 最短预览5秒
    const MAX_PREVIEW_SECONDS: u32 = 30; // 最长预览30秒

    // 获取输入文件元数据（用于后续恢复）
    let original_metadata = std::fs::metadata(input_path)
        .context("无法读取输入文件元数据")?;
    let original_atime = FileTime::from_last_access_time(&original_metadata);
    let original_mtime = FileTime::from_last_modification_time(&original_metadata);
    let original_permissions = original_metadata.permissions();

    // 获取输入文件大小
    let input_size = original_metadata.len();

    // 获取视频信息
    let video_info = get_video_info(input_path)?;
    let duration = video_info.duration;

    // 判断源视频编码类型
    let is_hevc = video_info
        .codec_name
        .as_deref()
        .map(|c| c.to_lowercase().contains("hevc") || c.to_lowercase().contains("h265"))
        .unwrap_or(false);

    println!(
        "  源编码: {} | 分辨率: {}x{} | 时长: {:.1}s",
        video_info.codec_name.as_deref().unwrap_or("unknown"),
        video_info.width,
        video_info.height,
        duration
    );

    // 如果已经是HEVC，直接跳过
    if is_hevc {
        return Ok(ProcessResult::Skipped("已是H.265编码".to_string()));
    }

    // 计算预览时长（10%时长，限制在5-30秒之间）
    let preview_duration = if duration > 0.0 {
        let preview_from_ratio = (duration * PREVIEW_RATIO) as u32;
        preview_from_ratio.max(MIN_PREVIEW_SECONDS).min(MAX_PREVIEW_SECONDS).min(duration as u32)
    } else {
        MIN_PREVIEW_SECONDS
    };

    println!("  预览编码 ({}秒, CRF={})...", preview_duration, PREVIEW_CRF);

    // 执行预览编码
    let preview_result = preview_encode(
        input_path,
        PREVIEW_CRF,
        preset,
        audio_bitrate,
        preview_duration,
        duration,
    )?;

    // 计算预估压缩比
    // 使用预览输出的每秒大小推算完整视频大小
    // 应用修正因子（预览编码通常比完整编码小约1.8倍）
    const CORRECTION_FACTOR: f64 = 1.8;
    let expected_output_size = preview_result.output_size_per_second * duration * CORRECTION_FACTOR;
    let predicted_ratio = if input_size > 0 {
        1.0 - expected_output_size / input_size as f64
    } else {
        0.0
    };

    let target_ratio = min_compression_ratio as f64 / 100.0;

    println!("  预估压缩比: {:.1}% (目标: {}%)", predicted_ratio * 100.0, min_compression_ratio);

    // 决定使用的CRF
    let final_crf = if predicted_ratio >= target_ratio {
        // 预估压缩比达标，直接使用 CRF=23
        PREVIEW_CRF
    } else {
        // 预估压缩比不达标，外推更高的CRF
        // 使用指数公式: CRF +6 ≈ 比特率减半
        // new_output_ratio = old_output_ratio * 2^((old_crf - new_crf) / 6)
        let predicted_output_ratio = 1.0 - predicted_ratio;  // 输出/输入 比例
        let target_output_ratio = 1.0 - target_ratio;

        if predicted_output_ratio <= 0.0 || target_output_ratio <= 0.0 {
            return Ok(ProcessResult::Skipped("无法计算有效CRF".to_string()));
        }

        // target_output_ratio = predicted_output_ratio * 2^((PREVIEW_CRF - new_crf) / 6)
        // 2^((PREVIEW_CRF - new_crf) / 6) = target_output_ratio / predicted_output_ratio
        // new_crf = PREVIEW_CRF - 6 * log2(target_output_ratio / predicted_output_ratio)
        let ratio = target_output_ratio / predicted_output_ratio;
        let log2_ratio = ratio.log2();
        let crf_delta = 6.0 * log2_ratio;
        let extrapolated_crf = (PREVIEW_CRF as f64 - crf_delta).round() as i32;

        println!("  外推CRF: {} (需要减少输出至 {:.1}%)", extrapolated_crf, target_output_ratio * 100.0);

        // 检查是否超出最大CRF限制
        if extrapolated_crf > max_crf as i32 {
            return Ok(ProcessResult::Skipped(format!(
                "需要CRF={} 超出限制({})",
                extrapolated_crf, max_crf
            )));
        }

        // 确保CRF在有效范围内
        extrapolated_crf.max(PREVIEW_CRF as i32) as u8
    };

    println!("  使用 CRF={} 进行完整编码...", final_crf);

    // 创建临时文件用于编码
    let temp_file = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;
    let temp_path = temp_file.path();

    // 执行完整编码
    let output_size = full_encode(input_path, temp_path, final_crf, preset, audio_bitrate)?;

    let actual_ratio = 1.0 - output_size as f64 / input_size as f64;
    println!(
        "  实际压缩: {} -> {} ({:.1}%)",
        format_size(input_size),
        format_size(output_size),
        actual_ratio * 100.0
    );

    // 生成最终输出路径
    let output_path = get_output_path(input_path);

    // 删除原文件
    std::fs::remove_file(input_path)
        .context("无法删除原文件")?;

    // 将临时文件移动到输出路径
    safe_move_file(temp_path, &output_path)
        .context("无法移动输出文件")?;

    // 恢复时间戳和权限
    set_file_times(&output_path, original_atime, original_mtime)
        .context("无法设置文件时间戳")?;
    std::fs::set_permissions(&output_path, original_permissions)
        .context("无法设置文件权限")?;

    // 消费temp_file避免它被删除
    let _ = temp_file;

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: final_crf,
    }))
}

struct PreviewResult {
    output_size_per_second: f64,   // 预览编码后的每秒大小（字节/秒）
}

fn preview_encode(
    input_path: &Path,
    crf: u8,
    preset: &str,
    audio_bitrate: u32,
    duration: u32,
    video_duration: f64,
) -> Result<PreviewResult> {
    // 创建临时文件
    let temp_output = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;
    let temp_output_path = temp_output.path();

    let segment_paths: Vec<PathBuf>;

    if video_duration <= duration as f64 * 1.5 {
        // 视频太短，从开头直接提取原始片段
        let temp_segment = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;

        let extract_cmd = Command::new("ffmpeg")
            .arg("-y")
            .arg("-i")
            .arg(input_path)
            .arg("-t")
            .arg(duration.to_string())
            .arg("-c")
            .arg("copy")
            .arg(temp_segment.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg提取预览片段失败")?;

        if !extract_cmd.status.success() {
            drop(temp_segment);
            anyhow::bail!("提取预览片段失败");
        }

        segment_paths = vec![temp_segment.path().to_path_buf()];
        // 保持 temp_segment 存活直到编码完成
        let _keep = temp_segment;

        // 对原始片段进行H.265编码（包含音频处理，与完整编码一致）
        let mut encode_cmd = Command::new("ffmpeg");
        encode_cmd
            .arg("-y")
            .arg("-i")
            .arg(&segment_paths[0])
            .arg("-c:v")
            .arg("libx265")
            .arg("-crf")
            .arg(crf.to_string())
            .arg("-preset")
            .arg(preset)
            .arg("-x265-params")
            .arg("fast=1:log-level=error")
            .arg("-f")
            .arg("mp4");

        // 音频处理（与完整编码保持一致）
        if audio_bitrate > 0 {
            encode_cmd.arg("-c:a").arg("aac").arg("-b:a").arg(format!("{}k", audio_bitrate));
        } else {
            encode_cmd.arg("-c:a").arg("copy");
        }

        encode_cmd.arg(temp_output_path);

        let output = encode_cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg预览编码失败")?;

        if !output.status.success() {
            anyhow::bail!("预览编码失败");
        }

        drop(_keep);
    } else {
        // 多段采样：开头(0%)、中间(50%)、结尾(90%)
        let segment_duration = duration / 3;
        let remaining = duration % 3;

        let seg1_start = 0.0;
        let seg2_start = video_duration * 0.5 - segment_duration as f64 / 2.0;
        let seg3_start = video_duration * 0.9;

        let seg2_start = seg2_start.max(0.0);
        let seg3_start = seg3_start.max(0.0);

        let seg1_dur = segment_duration;
        let seg2_dur = segment_duration;
        let seg3_dur = segment_duration + remaining;

        // 提取三个原始片段到临时文件
        let temp_seg1 = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件1")?;
        let temp_seg2 = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件2")?;
        let temp_seg3 = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件3")?;

        // 提取第一段（保留音频）
        let extract1 = Command::new("ffmpeg")
            .arg("-y")
            .arg("-ss")
            .arg(format!("{:.1}", seg1_start))
            .arg("-i")
            .arg(input_path)
            .arg("-t")
            .arg(seg1_dur.to_string())
            .arg("-c")
            .arg("copy")
            .arg(temp_seg1.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("提取片段1失败")?;

        // 提取第二段（保留音频）
        let extract2 = Command::new("ffmpeg")
            .arg("-y")
            .arg("-ss")
            .arg(format!("{:.1}", seg2_start))
            .arg("-i")
            .arg(input_path)
            .arg("-t")
            .arg(seg2_dur.to_string())
            .arg("-c")
            .arg("copy")
            .arg(temp_seg2.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("提取片段2失败")?;

        // 提取第三段（保留音频）
        let extract3 = Command::new("ffmpeg")
            .arg("-y")
            .arg("-ss")
            .arg(format!("{:.1}", seg3_start))
            .arg("-i")
            .arg(input_path)
            .arg("-t")
            .arg(seg3_dur.to_string())
            .arg("-c")
            .arg("copy")
            .arg(temp_seg3.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("提取片段3失败")?;

        if !extract1.status.success() || !extract2.status.success() || !extract3.status.success() {
            drop(temp_seg1);
            drop(temp_seg2);
            drop(temp_seg3);
            anyhow::bail!("提取多段预览片段失败");
        }

        // 创建concat列表文件
        let concat_list = NamedTempFile::with_suffix(".txt").context("无法创建concat列表")?;
        let concat_content = format!(
            "file '{}'\nfile '{}'\nfile '{}'\n",
            temp_seg1.path().display(),
            temp_seg2.path().display(),
            temp_seg3.path().display()
        );
        std::fs::write(concat_list.path(), concat_content).context("无法写入concat列表")?;

        // 使用concat demuxer拼接并编码（包含音频处理，与完整编码一致）
        let mut encode_cmd = Command::new("ffmpeg");
        encode_cmd
            .arg("-y")
            .arg("-f")
            .arg("concat")
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(concat_list.path())
            .arg("-c:v")
            .arg("libx265")
            .arg("-crf")
            .arg(crf.to_string())
            .arg("-preset")
            .arg(preset)
            .arg("-x265-params")
            .arg("fast=1:log-level=error")
            .arg("-f")
            .arg("mp4");

        // 音频处理（与完整编码保持一致）
        if audio_bitrate > 0 {
            encode_cmd.arg("-c:a").arg("aac").arg("-b:a").arg(format!("{}k", audio_bitrate));
        } else {
            encode_cmd.arg("-c:a").arg("copy");
        }

        encode_cmd.arg(temp_output_path);

        let output = encode_cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg预览编码失败")?;

        drop(concat_list);
        drop(temp_seg1);
        drop(temp_seg2);
        drop(temp_seg3);

        if !output.status.success() {
            anyhow::bail!("预览编码失败");
        }
    }

    // 获取编码后的大小
    let output_size = std::fs::metadata(temp_output_path)
        .context("无法读取预览输出文件")?
        .len() as f64;

    drop(temp_output);

    // 返回每秒输出大小，用于按比例推算完整视频大小
    let output_size_per_second = output_size / duration as f64;

    Ok(PreviewResult {
        output_size_per_second,
    })
}

fn full_encode(
    input_path: &Path,
    output_path: &Path,
    crf: u8,
    preset: &str,
    audio_bitrate: u32,
) -> Result<u64> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-i")
        .arg(input_path)
        .arg("-c:v")
        .arg("libx265")
        .arg("-crf")
        .arg(crf.to_string())
        .arg("-preset")
        .arg(preset)
        .arg("-x265-params")
        .arg("fast=1:log-level=error");

    // 音频处理
    if audio_bitrate > 0 {
        cmd.arg("-c:a")
            .arg("aac")
            .arg("-b:a")
            .arg(format!("{}k", audio_bitrate));
    } else {
        cmd.arg("-c:a").arg("copy");
    }

    cmd.arg(output_path);

    let output = cmd.output().context("执行FFmpeg失败")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg转换失败: {}", stderr);
    }

    Ok(std::fs::metadata(output_path)
        .context("无法读取输出文件")?
        .len())
}

/// Check if NVENC encoder is available
fn check_nvenc_available() -> Result<bool> {
    let output = Command::new("ffmpeg")
        .arg("-encoders")
        .output()
        .context("无法执行ffmpeg命令")?;

    if !output.status.success() {
        return Ok(false);
    }

    let encoders = String::from_utf8_lossy(&output.stdout);
    Ok(encoders.contains("hevc_nvenc"))
}

fn check_ffmpeg() -> Result<()> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("无法找到ffmpeg命令，请确保已安装FFmpeg")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg执行失败");
    }

    let version = String::from_utf8_lossy(&output.stdout);
    let version_lines: Vec<&str> = version.lines().collect();
    if let Some(line) = version_lines.first() {
        println!("检测到: {}", line);
    }

    // 检查是否支持libx265
    if version.contains("libx265") {
        println!("✓ H.265编码器可用");
    } else {
        anyhow::bail!("FFmpeg未编译libx265支持，无法进行H.265编码");
    }

    Ok(())
}

/// Build video encoder arguments for FFmpeg
fn build_encode_args(crf: u8, preset: &str, config: &EncoderConfig) -> Vec<String> {
    match config.encoder_type {
        EncoderType::Cpu => vec![
            "-c:v".to_string(),
            "libx265".to_string(),
            "-crf".to_string(),
            crf.to_string(),
            "-preset".to_string(),
            preset.to_string(),
            "-x265-params".to_string(),
            "fast=1:log-level=error".to_string(),
        ],
        EncoderType::Gpu => vec![
            "-c:v".to_string(),
            "hevc_nvenc".to_string(),
            "-preset".to_string(),
            "p7".to_string(),
            "-tune".to_string(),
            "hq".to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-rc-lookahead".to_string(),
            "32".to_string(),
            "-b_ref_mode".to_string(),
            "middle".to_string(),
            "-qmin".to_string(),
            "24".to_string(),
            "-qmax".to_string(),
            "28".to_string(),
            "-init_qpP".to_string(),
            "24".to_string(),
            "-init_qpB".to_string(),
            "26".to_string(),
            "-init_qpI".to_string(),
            "24".to_string(),
            "-no-scenecut".to_string(),
            "0".to_string(),
            "-spatial_aq".to_string(),
            "1".to_string(),
            "-temporal_aq".to_string(),
            "1".to_string(),
            "-aq-strength".to_string(),
            "8".to_string(),
        ],
        EncoderType::Auto => unreachable!("Auto should be resolved before calling build_encode_args"),
    }
}

fn scan_videos(dir: &Path) -> Result<Vec<PathBuf>> {
    let video_extensions = [
        "mp4", "mkv", "avi", "mov", "flv", "wmv", "webm", "m4v", "mpg", "mpeg", "3gp",
    ];

    let mut videos = Vec::new();

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if video_extensions.contains(&ext.to_lowercase().as_str()) {
                    videos.push(path.to_path_buf());
                }
            }
        }
    }

    videos.sort();
    Ok(videos)
}

#[derive(Debug)]
struct VideoInfo {
    codec_name: Option<String>,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    duration: f64,
    #[allow(dead_code)]
    bitrate: u64,
}

fn get_video_info(path: &Path) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=codec_name,width,height")
        .arg("-show_entries")
        .arg("format=duration,bit_rate")
        .arg("-of")
        .arg("default=nokey=1:noprint_wrappers=1")
        .arg(path)
        .output()
        .context("执行ffprobe失败")?;

    if !output.status.success() {
        anyhow::bail!("ffprobe执行失败");
    }

    let info = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = info.lines().collect();

    let codec_name = lines.get(0).map(|s| s.to_string());
    let width = lines.get(1).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height = lines.get(2).and_then(|s| s.parse().ok()).unwrap_or(1080);
    let duration = lines.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let bitrate = lines.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    Ok(VideoInfo {
        codec_name,
        width,
        height,
        duration,
        bitrate,
    })
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
