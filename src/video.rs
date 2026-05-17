use crate::config::{EncoderConfig, EncoderType};
use crate::encoder::{full_encode, preview_encode};
use crate::ffmpeg::get_video_info;
use crate::utils::{format_size, get_output_path, safe_move_file};
use anyhow::{Context, Result};
use filetime::{set_file_times, FileTime};
use std::path::Path;
use tempfile::NamedTempFile;

#[derive(Debug)]
pub enum ProcessResult {
    Converted(FileStats),
    Skipped(String),
}

#[derive(Debug, Default)]
pub struct FileStats {
    pub input_size: u64,
    pub output_size: u64,
    #[allow(dead_code)]
    pub used_crf: u8,
}

#[derive(Debug, Default)]
pub struct ConversionStats {
    pub successful: usize,
    pub skipped: usize,
    pub failed: usize,
    pub total_input_size: u64,
    pub total_output_size: u64,
}

pub fn process_video(
    input_path: &Path,
    encoder_config: &EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
) -> Result<ProcessResult> {
    const PREVIEW_CRF: u8 = 23; // Fixed preview CRF
    const PREVIEW_RATIO: f64 = 0.10; // Preview duration is 10% of video
    const MIN_PREVIEW_SECONDS: u32 = 5; // Minimum 5 seconds preview
    const JETSON_MAX_PREVIEW_SECONDS: u32 = 10; // Jetson CPU is slow, cap at 10s

    // Get input file metadata (for later restoration)
    let original_metadata =
        std::fs::metadata(input_path).context("无法读取输入文件元数据")?;
    let original_atime = FileTime::from_last_access_time(&original_metadata);
    let original_mtime = FileTime::from_last_modification_time(&original_metadata);
    let original_permissions = original_metadata.permissions();

    // Get input file size
    let input_size = original_metadata.len();

    // Get video info
    let video_info = get_video_info(input_path)?;
    let duration = video_info.duration;

    // Check source codec
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

    // Already HEVC, skip
    if is_hevc {
        return Ok(ProcessResult::Skipped("已是H.265编码".to_string()));
    }

    // Jetson NVMPI HEVC encoder has minimum resolution requirements (~160x160).
    // Below that, it enters an infinite error loop instead of failing cleanly.
    // Fall back to CPU encoding for small-resolution videos.
    let jetson_min_dim: u32 = 160;
    let needs_cpu_fallback = encoder_config.encoder_type == EncoderType::Jetson
        && (video_info.width < jetson_min_dim || video_info.height < jetson_min_dim);

    // Calculate preview duration: 10% of video, minimum 5 seconds
    // Jetson: cap at 10 seconds because CPU is slow
    let preview_duration = if duration > 0.0 {
        let preview_from_ratio = (duration * PREVIEW_RATIO) as u32;
        let base_duration = preview_from_ratio
            .max(MIN_PREVIEW_SECONDS)
            .min(duration as u32);

        if encoder_config.encoder_type == EncoderType::Jetson {
            base_duration.min(JETSON_MAX_PREVIEW_SECONDS)
        } else {
            base_duration
        }
    } else {
        MIN_PREVIEW_SECONDS
    };

    println!(
        "  预览编码 ({}秒, CRF={})...",
        preview_duration, PREVIEW_CRF
    );

    // For Jetson: preview encoding uses CPU to get baseline bitrate
    // Then Jetson full encode uses target_bitrate = CPU_bitrate * 1.8
    // If resolution is below Jetson minimum, entire pipeline falls back to CPU
    let (preview_config, jetson_target_bitrate) = if needs_cpu_fallback {
        let cpu_config = EncoderConfig {
            encoder_type: EncoderType::Cpu,
            preset: encoder_config.preset.clone(),
            target_bitrate_kbps: None,
            use_2pass: false,
        };
        println!(
            "  注意: 分辨率 {}x{} 低于Jetson最小要求({}x{})，回退到CPU编码",
            video_info.width, video_info.height, jetson_min_dim, jetson_min_dim
        );
        (cpu_config, false)
    } else if encoder_config.encoder_type == EncoderType::Jetson {
        // Create CPU config for preview encoding
        let cpu_config = EncoderConfig {
            encoder_type: EncoderType::Cpu,
            preset: encoder_config.preset.clone(),
            target_bitrate_kbps: None,
            use_2pass: false,
        };
        println!("  (Jetson: 使用CPU预览编码获取基准比特率)");
        (cpu_config, true) // flag to calculate target bitrate later
    } else {
        (encoder_config.clone(), false)
    };

    // Execute preview encode
    let preview_result = preview_encode(
        input_path,
        PREVIEW_CRF,
        &preview_config,
        audio_bitrate,
        preview_duration,
        duration,
        video_info.codec_name.as_deref(),
        video_info.pix_fmt.as_deref(),
    )?;

    // For Jetson: calculate target bitrate from preview result
    // CPU CRF23 bitrate * 2.0 = Jetson equivalent quality bitrate
    // If that exceeds 70% of original bitrate, cap at 70% and use maxrate constraint
    let final_encoder_config = if jetson_target_bitrate {
        let cpu_bitrate_bps = preview_result.output_size_per_second * 8.0; // bytes/sec to bits/sec
        let ideal_jetson_bitrate_kbps = (cpu_bitrate_bps * 2.0 / 1000.0) as u32;

        // Calculate 70% of original video bitrate
        let original_bitrate_kbps = if video_info.bitrate > 0 {
            (video_info.bitrate / 1000) as u32
        } else {
            // Fallback: estimate from file size
            ((input_size as f64 / duration) * 8.0 / 1000.0) as u32
        };
        let max_bitrate_kbps = (original_bitrate_kbps as f64 * 0.7) as u32;

        let (final_bitrate_kbps, use_bitrate_cap) = if ideal_jetson_bitrate_kbps > max_bitrate_kbps {
            println!(
                "  Jetson: 理想比特率 {}kbps > 原始70% ({}kbps)，使用70%限制",
                ideal_jetson_bitrate_kbps, max_bitrate_kbps
            );
            (max_bitrate_kbps, true)
        } else {
            println!(
                "  Jetson目标比特率: {}kbps (CPU基准×2.0)",
                ideal_jetson_bitrate_kbps
            );
            (ideal_jetson_bitrate_kbps, false)
        };

        EncoderConfig {
            encoder_type: EncoderType::Jetson,
            preset: encoder_config.preset.clone(),
            target_bitrate_kbps: Some(final_bitrate_kbps),
            use_2pass: use_bitrate_cap,
        }
    } else {
        encoder_config.clone()
    };

    // Calculate predicted compression ratio
    // Use preview output per-second size to estimate full video size
    // Apply correction factor (preview encode is typically ~1.8x smaller than full)
    const CORRECTION_FACTOR: f64 = 1.8;
    let expected_output_size = preview_result.output_size_per_second * duration * CORRECTION_FACTOR;
    let predicted_ratio = if input_size > 0 {
        1.0 - expected_output_size / input_size as f64
    } else {
        0.0
    };

    let target_ratio = min_compression_ratio as f64 / 100.0;

    println!(
        "  预估压缩比: {:.1}% (目标: {}%)",
        predicted_ratio * 100.0,
        min_compression_ratio
    );

    // Decide CRF to use
    let final_crf = if predicted_ratio >= target_ratio {
        // Predicted ratio meets target, use CRF=23
        PREVIEW_CRF
    } else {
        // Predicted ratio below target, extrapolate higher CRF
        // Use exponential formula: CRF +6 ≈ bitrate halved
        // new_output_ratio = old_output_ratio * 2^((old_crf - new_crf) / 6)
        let predicted_output_ratio = 1.0 - predicted_ratio; // output/input ratio
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

        println!(
            "  外推CRF: {} (需要减少输出至 {:.1}%)",
            extrapolated_crf,
            target_output_ratio * 100.0
        );

        // Check if exceeds max CRF limit
        if extrapolated_crf > max_crf as i32 {
            return Ok(ProcessResult::Skipped(format!(
                "需要CRF={} 超出限制({})",
                extrapolated_crf, max_crf
            )));
        }

        // Ensure CRF is in valid range
        extrapolated_crf.max(PREVIEW_CRF as i32) as u8
    };

    println!("  使用 CRF={} 进行完整编码...", final_crf);

    // Create temp file for encoding
    let temp_file = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;
    let temp_path = temp_file.path();

    // Execute full encode
    // If resolution below Jetson minimum, use CPU config instead
    let cpu_fallback_config;
    let effective_config = if needs_cpu_fallback {
        cpu_fallback_config = EncoderConfig {
            encoder_type: EncoderType::Cpu,
            preset: encoder_config.preset.clone(),
            target_bitrate_kbps: None,
            use_2pass: false,
        };
        &cpu_fallback_config
    } else {
        &final_encoder_config
    };
    let output_size = full_encode(input_path, temp_path, final_crf, effective_config, audio_bitrate, video_info.codec_name.as_deref(), video_info.pix_fmt.as_deref())?;

    let actual_ratio = 1.0 - output_size as f64 / input_size as f64;
    println!(
        "  实际压缩: {} -> {} ({:.1}%)",
        format_size(input_size),
        format_size(output_size),
        actual_ratio * 100.0
    );

    // Generate final output path
    let output_path = get_output_path(input_path);

    // Delete original file
    std::fs::remove_file(input_path).context("无法删除原文件")?;

    // Move temp file to output path
    safe_move_file(temp_path, &output_path).context("无法移动输出文件")?;

    // Restore timestamps and permissions
    set_file_times(&output_path, original_atime, original_mtime)
        .context("无法设置文件时间戳")?;
    std::fs::set_permissions(&output_path, original_permissions)
        .context("无法设置文件权限")?;

    // Consume temp_file to avoid deletion
    let _ = temp_file;

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: final_crf,
    }))
}
