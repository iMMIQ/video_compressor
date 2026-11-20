use crate::config::{EncodeParams, EncoderConfig, EncoderType};
use crate::encoder::{create_encoder, full_encode, preview_encode, JetsonEncoder, VideoEncoder};
use crate::ffmpeg::get_video_info;
use crate::logging::{log_debug, log_info};
use crate::model::{FileStats, ProcessResult};
use crate::signal::is_shutdown_requested;
use crate::state::StateCache;
use crate::utils::{
    atomic_replace_from_temp, create_temp_in_same_dir, file_mtime_secs, format_size,
    get_output_path,
};
use anyhow::{Context, Result};
use std::path::Path;

const PREVIEW_CRF: u8 = 23;
const PREVIEW_RATIO: f64 = 0.10;
const MIN_PREVIEW_SECONDS: u32 = 5;
/// Preview encode is typically ~1.8x smaller than full encode due to sampling/overhead.
const CORRECTION_FACTOR: f64 = 1.8;

#[derive(Debug, PartialEq, Eq)]
pub enum CrfDecision {
    UseCrf(u8),
    Skip(String),
}

/// Decide the final CRF based on predicted compression ratio, target ratio, and max CRF.
/// Pure function extracted for clarity and unit testing.
pub fn decide_crf(predicted_ratio: f64, min_compression_ratio: u8, max_crf: u8) -> CrfDecision {
    let target_ratio = min_compression_ratio as f64 / 100.0;
    if predicted_ratio >= target_ratio {
        return CrfDecision::UseCrf(PREVIEW_CRF);
    }

    let predicted_output_ratio = 1.0 - predicted_ratio;
    let target_output_ratio = 1.0 - target_ratio;

    if predicted_output_ratio <= 0.0 || target_output_ratio <= 0.0 {
        return CrfDecision::Skip("无法计算有效CRF".to_string());
    }

    // Exponential extrapolation: CRF +6 ≈ bitrate halved
    // new_crf = PREVIEW_CRF - 6 * log2(target_output_ratio / predicted_output_ratio)
    let ratio = target_output_ratio / predicted_output_ratio;
    let crf_delta = 6.0 * ratio.log2();
    let extrapolated_crf = (PREVIEW_CRF as f64 - crf_delta).round() as i32;

    if extrapolated_crf > max_crf as i32 {
        return CrfDecision::Skip(format!(
            "需要CRF={} 超出限制({})",
            extrapolated_crf, max_crf
        ));
    }

    CrfDecision::UseCrf(extrapolated_crf.max(PREVIEW_CRF as i32) as u8)
}

pub fn process_video(
    input_path: &Path,
    encoder_config: &EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
    state_cache: Option<&StateCache>,
) -> Result<ProcessResult> {
    if is_shutdown_requested() {
        anyhow::bail!("收到中断信号，跳过处理");
    }

    let output_path = get_output_path(input_path);

    // Guard against clobbering a pre-existing distinct .mp4 target BEFORE probing/encoding.
    if output_path.exists() && output_path != input_path {
        return Ok(ProcessResult::Skipped(format!(
            "目标已存在，跳过避免覆盖: {}",
            output_path.display()
        )));
    }

    let original_metadata =
        std::fs::metadata(input_path).context("无法读取输入文件元数据")?;
    let input_size = original_metadata.len();
    let mtime_secs = file_mtime_secs(&original_metadata);

    if let Some(cache) = state_cache
        && let Some(reason) = cache.get_cached_skip(input_path, input_size, mtime_secs)
    {
        return Ok(ProcessResult::Skipped(format!("{} (缓存)", reason)));
    }

    let record_skip = |reason: String| -> Result<ProcessResult> {
        if let Some(cache) = state_cache {
            cache.record_skip(input_path, input_size, mtime_secs, &reason);
        }
        Ok(ProcessResult::Skipped(reason))
    };

    log_info(&format!("处理视频: {}", input_path.display()));

    let video_info = get_video_info(input_path)?;
    let duration = video_info.duration;

    let is_hevc = video_info
        .codec_name
        .as_deref()
        .map(|c| {
            let lower = c.to_ascii_lowercase();
            lower.contains("hevc") || lower.contains("h265")
        })
        .unwrap_or(false);

    log_info(&format!(
        "  源编码: {} | 分辨率: {}x{} | 时长: {:.1}s",
        video_info.codec_name.as_deref().unwrap_or("unknown"),
        video_info.width,
        video_info.height,
        duration
    ));

    if is_hevc {
        return record_skip("已是H.265编码".to_string());
    }

    // Select encoder backend and fall back to CPU if unsupported (e.g. Jetson small resolution).
    let primary_encoder = create_encoder(encoder_config.encoder_type);
    let active_encoder: Box<dyn VideoEncoder> = if primary_encoder.supports_video(&video_info) {
        primary_encoder
    } else {
        log_info(&format!(
            "  注意: 分辨率 {}x{} 低于Jetson最小要求({}x{})，回退到CPU编码",
            video_info.width,
            video_info.height,
            JetsonEncoder::MIN_DIMENSION,
            JetsonEncoder::MIN_DIMENSION
        ));
        create_encoder(EncoderType::Cpu)
    };

    // Preview duration: 10% of video, minimum 5s, optional backend cap (Jetson 10s).
    let preview_duration = if duration > 0.0 {
        let base = ((duration * PREVIEW_RATIO) as u32)
            .max(MIN_PREVIEW_SECONDS)
            .min((duration as u32).max(1));
        if let Some(cap) = active_encoder.preview_max_seconds() {
            base.min(cap)
        } else {
            base
        }
    } else {
        MIN_PREVIEW_SECONDS
    };

    let preview_encoder = active_encoder.preview_encoder();
    if active_encoder.encoder_type() == EncoderType::Jetson {
        log_info("  (Jetson: 使用CPU预览编码获取基准比特率)");
    }
    log_info(&format!(
        "  预览编码 ({}秒, CRF={})...",
        preview_duration, PREVIEW_CRF
    ));

    let preview_params = EncodeParams::from_crf(PREVIEW_CRF, &encoder_config.preset);
    let preview_result = preview_encode(
        input_path,
        &preview_params,
        preview_encoder.as_ref(),
        audio_bitrate,
        preview_duration,
        duration,
    )?;

    let expected_output_size = preview_result.output_size_per_second * duration * CORRECTION_FACTOR;
    let predicted_ratio = if input_size > 0 {
        1.0 - expected_output_size / input_size as f64
    } else {
        0.0
    };

    log_info(&format!(
        "  预估压缩比: {:.1}% (目标: {}%)",
        predicted_ratio * 100.0,
        min_compression_ratio
    ));

    let final_crf = match decide_crf(predicted_ratio, min_compression_ratio, max_crf) {
        CrfDecision::UseCrf(crf) => crf,
        CrfDecision::Skip(reason) => return record_skip(reason),
    };

    let full_params = active_encoder.resolve_full_params(
        final_crf,
        &encoder_config.preset,
        &preview_result,
        &video_info,
        input_size,
    );

    log_info(&format!("  使用 CRF={} 进行完整编码...", final_crf));
    log_debug(&format!("  完整编码参数: {:?}", full_params));

    // Crash-safe replacement: encode into a temp file in the SAME directory as output_path.
    let temp_file = create_temp_in_same_dir(&output_path, ".mp4")?;
    let output_size = full_encode(
        input_path,
        temp_file.path(),
        &full_params,
        active_encoder.as_ref(),
        audio_bitrate,
    )?;

    // Post-encode size guard: never replace the original with a larger file!
    if output_size >= input_size {
        drop(temp_file);
        return record_skip(format!(
            "编码后体积更大 ({} >= {})，保留原文件",
            format_size(output_size),
            format_size(input_size)
        ));
    }

    atomic_replace_from_temp(temp_file, input_path, &output_path, &original_metadata)?;

    let actual_ratio = (1.0 - output_size as f64 / input_size as f64) * 100.0;
    log_info(&format!(
        "  实际压缩: {} -> {} ({:.1}%)",
        format_size(input_size),
        format_size(output_size),
        actual_ratio
    ));

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: final_crf,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decide_crf_uses_preview_crf_when_target_met() {
        assert_eq!(decide_crf(0.45, 30, 35), CrfDecision::UseCrf(23));
        assert_eq!(decide_crf(0.30, 30, 35), CrfDecision::UseCrf(23));
    }

    #[test]
    fn decide_crf_extrapolates_within_limit() {
        // predicted_ratio = 0.0 -> output_ratio = 1.0, target_output_ratio = 0.7
        // 23 - 6 * log2(0.7) ≈ 23 + 3.09 = 26
        assert_eq!(decide_crf(0.0, 30, 35), CrfDecision::UseCrf(26));
    }

    #[test]
    fn decide_crf_skips_when_exceeding_max_crf() {
        // predicted_ratio = -3.0 -> output_ratio = 4.0, target = 0.7 -> 23 + 6*log2(4/0.7) ≈ 38 > 35
        match decide_crf(-3.0, 30, 35) {
            CrfDecision::Skip(reason) => assert!(reason.contains("超出限制(35)")),
            other => panic!("expected Skip, got {:?}", other),
        }
    }
}
