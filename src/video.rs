use crate::config::{EncodeParams, EncoderConfig, EncoderType, QualityProfile};
use crate::encoder::{
    EncodeOutcome, HwSlotPool, JetsonGstEncoder, JetsonNvmpiEncoder, ReferenceSample,
    VideoEncoder, create_encoder, create_encoder_of, extract_reference_sample,
};
use crate::ffmpeg::{
    AudioPlan, VideoInfo, get_video_info, measure_ssim_fast, measure_xpsnr_fast,
    probe_output_integrity,
};
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
use tempfile::NamedTempFile;

/// Container/mux overhead factor (MP4 moov/stbl/audio interleaving ~2%).
const MUX_OVERHEAD_FACTOR: f64 = 1.02;

/// Minimum BPP threshold below which an H.264 video (>=640px, >=5s) is already so heavily
/// compressed that re-encoding cannot shrink it meaningfully without visible degradation.
const LOW_BPP_SKIP_THRESHOLD: f64 = 0.032;

/// Noise-floor knee threshold (dB): if dropping Q by 4 points yields less than 1.0 dB XPSNR gain,
/// extra bits are mostly wasted coding random sensor/film noise rather than visible detail.
const NOISE_KNEE_DELTA_DB: f64 = 1.0;

/// SSIM target fallback when FFmpeg does not have the `xpsnr` filter built in.
const TARGET_SSIM: f64 = 0.980;

#[derive(Debug, Clone, Copy)]
struct ProbePoint {
    q: u8,
    score: f64,
    video_bps: f64,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chosen_q: u8,
    pub score_label: String,
    pub est_video_bps: f64,
}

/// Layer 1 fast pre-filter: skip already heavily compressed H.264 sources without running encode.
pub fn should_skip_low_bpp(info: &VideoInfo, input_size: u64) -> Option<String> {
    let codec = info.codec_name.as_deref()?.to_ascii_lowercase();
    if codec != "h264" || info.width < 640 || info.height < 360 || info.duration < 5.0 {
        return None;
    }
    let bpp = info.bits_per_pixel(input_size);
    if bpp > 0.0 && bpp < LOW_BPP_SKIP_THRESHOLD {
        Some(format!(
            "源H.264码率已极低 (BPP={:.4} < {:.3})，视觉无损下无有效压缩空间",
            bpp, LOW_BPP_SKIP_THRESHOLD
        ))
    } else {
        None
    }
}

/// Compute predicted total output size (bytes) cleanly separating pure video bitrate and audio bitrate.
pub fn predict_total_output_bytes(est_video_bps: f64, audio_bps: u64, duration: f64) -> f64 {
    ((est_video_bps + audio_bps as f64) / 8.0) * duration * MUX_OVERHEAD_FACTOR
}

/// Evaluate whether predicted output meets BOTH the hard zero-bloat rule (`< input_size`)
/// and the user's minimum net saving threshold (`min_saving_percent`).
pub fn check_predicted_budget(
    est_video_bps: f64,
    audio_bps: u64,
    duration: f64,
    input_size: u64,
    min_saving_percent: u8,
) -> Result<f64, String> {
    if input_size == 0 || duration <= 0.0 {
        return Ok(0.0);
    }
    let predicted_bytes = predict_total_output_bytes(est_video_bps, audio_bps, duration);
    let predicted_saving_pct = (1.0 - predicted_bytes / input_size as f64) * 100.0;

    if predicted_bytes >= input_size as f64 {
        return Err(format!(
            "视觉无损下预估体积将膨胀 ({} >= {})，跳过保留原片",
            format_size(predicted_bytes as u64),
            format_size(input_size)
        ));
    }

    if predicted_saving_pct < min_saving_percent as f64 {
        return Err(format!(
            "视觉无损下预估节省 {:.1}% < 最低要求 {}%，跳过保留原片",
            predicted_saving_pct, min_saving_percent
        ));
    }

    Ok(predicted_saving_pct)
}

/// Verify encoded output duration matches original duration within safe tolerance
/// (prevents a truncated hardware encode from being mistaken for a "smaller file").
pub fn is_duration_consistent(src_duration: f64, out_duration: f64) -> bool {
    if src_duration <= 0.0 {
        return out_duration > 0.0;
    }
    let diff = (src_duration - out_duration).abs();
    let tolerance = (src_duration * 0.03).max(2.0);
    diff <= tolerance
}

/// Layer 4 hard final gate before atomic replacement:
/// 1. Hard invariant: output MUST be strictly smaller than input (`output_size < input_size`).
/// 2. Net saving gate: must achieve at least `min_saving_percent` (when > 0).
pub fn check_final_size_gate(
    input_size: u64,
    output_size: u64,
    min_saving_percent: u8,
) -> Result<f64, String> {
    if output_size == 0 {
        return Err("输出文件为空，保留原片".to_string());
    }
    if output_size >= input_size {
        return Err(format!(
            "转换后体积未变小 ({} >= {})，严格保留原文件",
            format_size(output_size),
            format_size(input_size)
        ));
    }
    let actual_saving_pct = (1.0 - output_size as f64 / input_size as f64) * 100.0;
    if min_saving_percent > 0 && actual_saving_pct < min_saving_percent as f64 {
        return Err(format!(
            "实际节省 {:.1}% < 最低收益阈值 {}% ({} -> {})，保留原文件",
            actual_saving_pct,
            min_saving_percent,
            format_size(input_size),
            format_size(output_size)
        ));
    }
    Ok(actual_saving_pct)
}

/// Compute real-time watchdog size ceiling for the full-encode temp file.
/// Strictly `< input_size` so any encode reaching original size is aborted immediately mid-flight.
pub fn compute_watchdog_limit(
    input_size: u64,
    duration: f64,
    audio_bps: u64,
    separate_video_track: bool,
) -> Option<u64> {
    if input_size < 128 * 1024 {
        return None;
    }
    if separate_video_track && duration > 0.0 {
        let est_audio_bytes = ((audio_bps as f64 / 8.0) * duration) as u64;
        let video_cap = input_size.saturating_sub(est_audio_bytes.min(input_size / 2));
        Some(video_cap.max(input_size / 2))
    } else {
        Some(input_size)
    }
}

/// Probe a single quality value on the reference sample and measure its perceptual score + bitrate.
#[allow(clippy::too_many_arguments)]
fn probe_quality_point(
    encoder: &dyn VideoEncoder,
    sample: &ReferenceSample,
    q: u8,
    preset: &str,
    ten_bit: bool,
    gpu_index: Option<usize>,
    use_xpsnr: bool,
    use_ssim: bool,
    hw_slots: Option<&HwSlotPool>,
) -> Result<ProbePoint> {
    let suffix = if encoder.encoder_type() == EncoderType::JetsonGst {
        ".mkv"
    } else {
        ".mp4"
    };
    let tmp = NamedTempFile::with_suffix(suffix).context("无法创建样本编码临时文件")?;

    let mut params = EncodeParams::for_quality(q, preset, ten_bit, true);
    params.gpu_index = gpu_index;

    let bytes = encoder.encode_sample(
        sample.file.path(),
        &sample.info,
        tmp.path(),
        &params,
        hw_slots,
    )?;

    let video_bps = (bytes as f64 * 8.0) / sample.duration.max(0.5);

    let score = if use_xpsnr {
        measure_xpsnr_fast(sample.file.path(), tmp.path())?
    } else if use_ssim {
        measure_ssim_fast(sample.file.path(), tmp.path())?
    } else {
        0.0
    };

    log_debug(&format!(
        "    [感知探测] Q={} -> score={:.3}, est_vbr={:.2} Mbps",
        q,
        score,
        video_bps / 1_000_000.0
    ));

    Ok(ProbePoint {
        q,
        score,
        video_bps,
    })
}

/// 3-step noise-knee-aware perceptual quality search:
/// Finds the HIGHEST quality knob Q (smallest file size) that keeps perceptual loss
/// visually imperceptible (`XPSNR >= target_xpsnr`, or `SSIM >= 0.980`), while detecting
/// high-noise/grain footage where lowering Q wastes bitrate without visible improvement.
#[allow(clippy::too_many_arguments)]
fn search_perceptual_quality(
    encoder: &dyn VideoEncoder,
    sample: &ReferenceSample,
    profile: QualityProfile,
    preset: &str,
    ten_bit: bool,
    gpu_index: Option<usize>,
    target_xpsnr: f64,
    has_xpsnr: bool,
    has_ssim: bool,
    hw_slots: Option<&HwSlotPool>,
) -> Result<SearchResult> {
    let use_xpsnr = has_xpsnr;
    let use_ssim = !has_xpsnr && has_ssim;
    let target = if use_xpsnr { target_xpsnr } else { TARGET_SSIM };

    // Step 1: Anchor probe
    let p0 = probe_quality_point(
        encoder,
        sample,
        profile.anchor_q,
        preset,
        ten_bit,
        gpu_index,
        use_xpsnr,
        use_ssim,
        hw_slots,
    )?;

    if !use_xpsnr && !use_ssim {
        return Ok(SearchResult {
            chosen_q: p0.q,
            score_label: "无感知滤镜(使用基准Q)".to_string(),
            est_video_bps: p0.video_bps,
        });
    }

    let chosen = if p0.score >= target {
        // Case A: Clean / easy-to-compress footage!
        // Already visually transparent at anchor_q -> search higher Q (smaller file!) in 2 steps.
        let mut best = p0;
        let mut lo = profile.anchor_q.saturating_add(1);
        let mut hi = profile.max_q;
        for _ in 0..2 {
            if lo > hi {
                break;
            }
            let mid = lo + (hi - lo) / 2;
            let cand = probe_quality_point(
                encoder, sample, mid, preset, ten_bit, gpu_index, use_xpsnr, use_ssim, hw_slots,
            )?;
            if cand.score >= target {
                best = cand;
                lo = mid.saturating_add(1);
            } else if mid > lo {
                hi = mid.saturating_sub(1);
            } else {
                break;
            }
        }
        best
    } else {
        // Case B: Complex or noisy/grainy footage (anchor_q < target).
        // Probe lower Q to distinguish real detail loss from random high-frequency noise ceiling!
        let q_low = profile
            .anchor_q
            .saturating_sub(profile.low_probe_delta)
            .max(profile.min_q);
        if q_low >= profile.anchor_q {
            p0
        } else {
            let p_low = probe_quality_point(
                encoder, sample, q_low, preset, ten_bit, gpu_index, use_xpsnr, use_ssim, hw_slots,
            )?;
            let gain = p_low.score - p0.score;
            let knee_threshold = if use_xpsnr {
                NOISE_KNEE_DELTA_DB
            } else {
                0.004
            };

            if gain < knee_threshold {
                // Noise-floor knee detected: lowering Q spent ~2x bitrate for negligible
                // perceptual gain -> lock at anchor_q instead of bloating bitrate!
                log_debug(&format!(
                    "    [噪点拐点] Q={}->{} 增益仅 {:.2} < {:.2}，锁定拐点 Q={}",
                    p0.q, p_low.q, gain, knee_threshold, p0.q
                ));
                p0
            } else if p_low.score >= target {
                // Real detail gain reached target -> 1 midpoint check to maximize compression.
                let mid = q_low + (profile.anchor_q - q_low) / 2;
                if mid > q_low && mid < profile.anchor_q {
                    let p_mid = probe_quality_point(
                        encoder, sample, mid, preset, ten_bit, gpu_index, use_xpsnr, use_ssim,
                        hw_slots,
                    )?;
                    if p_mid.score >= target { p_mid } else { p_low }
                } else {
                    p_low
                }
            } else {
                p_low
            }
        }
    };

    let score_label = if use_xpsnr {
        format!("XPSNR={:.2}dB", chosen.score)
    } else {
        format!("SSIM={:.4}", chosen.score)
    };

    Ok(SearchResult {
        chosen_q: chosen.q,
        score_label,
        est_video_bps: chosen.video_bps,
    })
}

/// Process a single video through the 4-layer zero-bloat perceptual compression pipeline.
#[allow(clippy::too_many_arguments)]
pub fn process_video(
    input_path: &Path,
    encoder_config: &EncoderConfig,
    max_crf: u8,
    min_saving_percent: u8,
    audio_bitrate: u32,
    force: bool,
    gpu_index: Option<usize>,
    hw_slots: Option<&HwSlotPool>,
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

    if !force
        && let Some(cache) = state_cache
        && let Some(reason) = cache.get_cached_skip(input_path, input_size, mtime_secs)
    {
        return Ok(ProcessResult::Skipped(format!("{} (缓存)", reason)));
    }

    let record_skip = |reason: String, permanent: bool| -> Result<ProcessResult> {
        if let Some(cache) = state_cache {
            if permanent {
                cache.record_permanent_skip(input_path, input_size, mtime_secs, &reason);
            } else {
                cache.record_skip(input_path, input_size, mtime_secs, &reason);
            }
        }
        Ok(ProcessResult::Skipped(reason))
    };

    log_info(&format!("处理视频: {}", input_path.display()));

    // === Layer 1: Single-call ffprobe + zero-encode pre-filtering ===
    let video_info = get_video_info(input_path)?;
    let duration = video_info.duration;

    log_info(&format!(
        "  源编码: {} | {}x{}@{:.1}fps | 时长: {:.1}s | 体积: {}{}",
        video_info.codec_name.as_deref().unwrap_or("unknown"),
        video_info.width,
        video_info.height,
        video_info.fps,
        duration,
        format_size(input_size),
        if video_info.is_hdr() { " [HDR]" } else { "" }
    ));

    if !force {
        // Skip already modern codecs (HEVC, AV1, VP9) permanently.
        if let Some(reason) = video_info.is_modern_codec() {
            return record_skip(reason.to_string(), true);
        }

        // Skip already heavily compressed H.264 sources (ultra-low BPP).
        if let Some(reason) = should_skip_low_bpp(&video_info, input_size) {
            return record_skip(reason, false);
        }
    } else if let Some(reason) = video_info.is_modern_codec() {
        log_info(&format!("  (--force 已启用: {}，仍强制进行感知重编码评估)", reason));
    }

    // Select active encoder backend; fall back to 10-bit CPU if unsupported
    // (e.g., Jetson tiny resolution < 160px, or HDR source on 8-bit-only encoder).
    let primary_encoder = create_encoder(encoder_config);
    let active_encoder: Box<dyn VideoEncoder> = if !primary_encoder.supports_video(&video_info) {
        log_info(&format!(
            "  注意: 分辨率 {}x{} 低于硬件最小要求({}x{})，回退到CPU编码",
            video_info.width,
            video_info.height,
            JetsonGstEncoder::MIN_DIMENSION.min(JetsonNvmpiEncoder::MIN_DIMENSION),
            JetsonGstEncoder::MIN_DIMENSION.min(JetsonNvmpiEncoder::MIN_DIMENSION)
        ));
        create_encoder_of(EncoderType::Cpu, &encoder_config.capabilities)
    } else if (video_info.is_hdr() || video_info.is_high_bit_depth())
        && !primary_encoder.supports_10bit()
    {
        if encoder_config.capabilities.has_libx265 && encoder_config.capabilities.x265_10bit {
            log_info("  注意: 源视频为10-bit/HDR而当前硬件仅支持8-bit，回退到10-bit CPU避免色深降级或HDR发灰");
            create_encoder_of(EncoderType::Cpu, &encoder_config.capabilities)
        } else {
            return record_skip("源视频为10-bit/HDR且无10-bit编码器，跳过避免色彩毁损".to_string(), false);
        }
    } else {
        primary_encoder
    };

    let ten_bit = active_encoder.supports_10bit();
    let audio_plan = AudioPlan::decide(&video_info, audio_bitrate);
    let audio_bps = audio_plan.estimated_bps(&video_info);
    log_debug(&format!(
        "  音频策略: {:?} (估算 {} kbps) | 10-bit: {}",
        audio_plan,
        audio_bps / 1000,
        ten_bit
    ));

    // === Layer 2: Hardware-accelerated perceptual quality search + budget pruning ===
    let sample = extract_reference_sample(
        input_path,
        &video_info,
        encoder_config.capabilities.has_libx264,
    )?;

    let mut full_params = if active_encoder.supports_quality_search() {
        let profile = active_encoder.quality_profile(max_crf);
        let search = search_perceptual_quality(
            active_encoder.as_ref(),
            &sample,
            profile,
            &encoder_config.preset,
            ten_bit,
            gpu_index,
            encoder_config.target_xpsnr,
            encoder_config.capabilities.has_xpsnr,
            encoder_config.capabilities.has_ssim,
            hw_slots,
        )?;

        let predicted_saving = match check_predicted_budget(
            search.est_video_bps,
            audio_bps,
            duration,
            input_size,
            min_saving_percent,
        ) {
            Ok(pct) => pct,
            Err(reason) => {
                log_info(&format!(
                    "  感知搜索: Q={} ({}, 预估视频 {:.2}Mbps) -> {}",
                    search.chosen_q,
                    search.score_label,
                    search.est_video_bps / 1_000_000.0,
                    reason
                ));
                return record_skip(reason, false);
            }
        };

        log_info(&format!(
            "  感知最优: Q={} ({}) | 预估视频 {:.2}Mbps | 预估节省 {:.1}%",
            search.chosen_q,
            search.score_label,
            search.est_video_bps / 1_000_000.0,
            predicted_saving
        ));

        let mut p = EncodeParams::for_quality(
            search.chosen_q,
            &encoder_config.preset,
            ten_bit,
            false,
        );
        p.gpu_index = gpu_index;

        // Soft VBV peak cap (1.35x source effective video bitrate, never pseudo-CBR) to guard against
        // extreme noise spikes on un-sampled segments without constraining normal scenes.
        let src_video_kbps = (video_info.effective_video_bitrate(input_size) / 1000) as u32;
        if src_video_kbps >= 1500
            && matches!(
                active_encoder.encoder_type(),
                EncoderType::Gpu | EncoderType::JetsonGst
            )
        {
            p.maxrate_kbps = Some((src_video_kbps as f64 * 1.35) as u32);
        }
        p
    } else {
        // Legacy JetsonNvmpi fallback (bitrate-based, audio cleanly excluded).
        let cpu_enc = create_encoder_of(EncoderType::Cpu, &encoder_config.capabilities);
        let cpu_profile = cpu_enc.quality_profile(max_crf);
        let search = search_perceptual_quality(
            cpu_enc.as_ref(),
            &sample,
            cpu_profile,
            "fast",
            false,
            None,
            encoder_config.target_xpsnr,
            encoder_config.capabilities.has_xpsnr,
            encoder_config.capabilities.has_ssim,
            None,
        )?;
        let target_v_bps = (search.est_video_bps * 1.35).max(150_000.0);
        if let Err(reason) = check_predicted_budget(
            target_v_bps,
            audio_bps,
            duration,
            input_size,
            min_saving_percent,
        ) {
            return record_skip(reason, false);
        }
        let target_kbps = (target_v_bps / 1000.0) as u32;
        let mut p =
            EncodeParams::for_quality(search.chosen_q, &encoder_config.preset, false, false);
        p.target_bitrate_kbps = Some(target_kbps);
        p.maxrate_kbps = Some((target_kbps as f64 * 1.5) as u32);
        p
    };
    drop(sample); // Release sample temp file immediately before full encode.

    if full_params.gpu_index.is_none() {
        full_params.gpu_index = gpu_index;
    }

    // === Layer 3: Full hardware encode + real-time size watchdog ===
    let temp_file = create_temp_in_same_dir(&output_path, ".mp4")?;
    let separate_video_track = active_encoder.encoder_type() == EncoderType::JetsonGst;
    let watchdog_limit =
        compute_watchdog_limit(input_size, duration, audio_bps, separate_video_track);

    log_debug(&format!(
        "  开始完整编码: {:?} | 实时熔断阈值: {:?}",
        full_params,
        watchdog_limit.map(format_size)
    ));

    let output_size = match active_encoder.full_encode(
        input_path,
        &video_info,
        temp_file.path(),
        &full_params,
        audio_plan,
        watchdog_limit,
        hw_slots,
    )? {
        EncodeOutcome::AbortedSizeLimit { written_bytes } => {
            drop(temp_file);
            return record_skip(
                format!(
                    "编码中体积已达 {} (接近/超过原片 {})，已提前熔断保留原文件",
                    format_size(written_bytes),
                    format_size(input_size)
                ),
                false,
            );
        }
        EncodeOutcome::Completed(sz) => sz,
    };

    // === Layer 4: Final hard zero-bloat + min-saving + duration integrity gate ===
    let actual_saving_pct = match check_final_size_gate(input_size, output_size, min_saving_percent)
    {
        Ok(pct) => pct,
        Err(reason) => {
            drop(temp_file);
            return record_skip(reason, false);
        }
    };

    // Integrity check: verify valid video stream & matching duration so a truncated encode
    // can NEVER masquerade as a "smaller file" and overwrite the original!
    match probe_output_integrity(temp_file.path()) {
        Some(out_dur) if is_duration_consistent(duration, out_dur) => {}
        Some(out_dur) => {
            drop(temp_file);
            anyhow::bail!(
                "输出时长异常截断 (原片 {:.1}s vs 输出 {:.1}s)，已拒绝替换并保留原文件",
                duration,
                out_dur
            );
        }
        None => {
            drop(temp_file);
            anyhow::bail!("输出文件视频流校验失败，已拒绝替换并保留原文件");
        }
    }

    atomic_replace_from_temp(temp_file, input_path, &output_path, &original_metadata)?;

    log_info(&format!(
        "  实际压缩: {} -> {} ({:.1}% 节省)",
        format_size(input_size),
        format_size(output_size),
        actual_saving_pct
    ));

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: full_params.quality,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_size_gate_strictly_forbids_larger_or_equal_output() {
        assert!(check_final_size_gate(1_000_000, 1_000_000, 0).is_err());
        assert!(check_final_size_gate(1_000_000, 1_000_001, 0).is_err());
        assert!(check_final_size_gate(1_000_000, 0, 0).is_err());

        // Smaller by 5% fails 15% min-saving gate, passes 0% gate
        assert!(check_final_size_gate(1_000_000, 950_000, 15).is_err());
        assert!((check_final_size_gate(1_000_000, 950_000, 0).unwrap() - 5.0).abs() < 1e-6);

        // Smaller by 40% passes 15% gate
        assert!((check_final_size_gate(1_000_000, 600_000, 15).unwrap() - 40.0).abs() < 1e-6);
    }

    #[test]
    fn predicted_budget_separates_audio_and_rejects_bloat() {
        // 10s video, input = 10 MB (8 Mbps total).
        // Estimated video = 4 Mbps, audio = 128 kbps -> ~5.26 MB (~47% saving) -> OK
        let pct = check_predicted_budget(4_000_000.0, 128_000, 10.0, 10_000_000, 15).unwrap();
        assert!(pct > 45.0 && pct < 50.0);

        // Estimated video = 8.5 Mbps -> larger than 10 MB -> rejected!
        assert!(check_predicted_budget(8_500_000.0, 128_000, 10.0, 10_000_000, 15).is_err());

        // Estimated video = 7.0 Mbps -> ~10% saving < 15% threshold -> rejected!
        assert!(check_predicted_budget(7_000_000.0, 128_000, 10.0, 10_000_000, 15).is_err());
    }

    #[test]
    fn duration_consistency_rejects_truncated_encode() {
        assert!(is_duration_consistent(100.0, 99.5));
        assert!(is_duration_consistent(10.0, 9.0));
        assert!(!is_duration_consistent(100.0, 50.0));
        assert!(!is_duration_consistent(100.0, 90.0));
    }

    #[test]
    fn low_bpp_skips_already_tiny_h264() {
        let info = VideoInfo {
            codec_name: Some("h264".into()),
            width: 1920,
            height: 1080,
            fps: 30.0,
            duration: 60.0,
            video_bitrate: 1_000_000, // BPP = 0.016 < 0.032
            ..Default::default()
        };
        assert!(should_skip_low_bpp(&info, 8_000_000).is_some());

        let high_br = VideoInfo {
            video_bitrate: 10_000_000, // BPP = 0.16
            ..info
        };
        assert!(should_skip_low_bpp(&high_br, 75_000_000).is_none());
    }
}
