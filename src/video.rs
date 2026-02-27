use crate::config::EncoderConfig;
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

    // Calculate preview duration (10% of video, minimum 5 seconds)
    let preview_duration = if duration > 0.0 {
        let preview_from_ratio = (duration * PREVIEW_RATIO) as u32;
        preview_from_ratio
            .max(MIN_PREVIEW_SECONDS)
            .min(duration as u32)
    } else {
        MIN_PREVIEW_SECONDS
    };

    println!(
        "  预览编码 ({}秒, CRF={})...",
        preview_duration, PREVIEW_CRF
    );

    // Execute preview encode
    let preview_result = preview_encode(
        input_path,
        PREVIEW_CRF,
        encoder_config,
        audio_bitrate,
        preview_duration,
        duration,
    )?;

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
    let output_size = full_encode(input_path, temp_path, final_crf, encoder_config, audio_bitrate)?;

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
