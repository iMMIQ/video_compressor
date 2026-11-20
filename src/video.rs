use crate::config::{EncodeParams, EncoderConfig, EncoderType, QualityProfile, VideoConfig};
use crate::encoder::{
    EncodeOutcome, Encoder, HwSlotPool, ReferenceSample, extract_reference_sample,
};
use crate::ffmpeg::{
    AudioPlan, QualityMetric, VideoInfo, get_video_info, measure_metric_fast,
    probe_output_integrity,
};
use crate::logging::{log_debug, log_info};
use crate::model::{FileStats, ProcessResult, savings_percent};
use crate::state::StateCache;
use crate::utils::{
    atomic_replace_from_temp, available_space_bytes, copy_file_to_temp, create_temp_in_same_dir,
    format_size, get_output_path, is_network_filesystem, preflight_media_file,
};
use anyhow::{Context, Result};
use std::path::Path;
use tempfile::NamedTempFile;

/// Container/mux overhead factor (MP4 moov/stbl/audio interleaving ~2%).
const MUX_OVERHEAD_FACTOR: f64 = 1.02;

/// Minimum BPP threshold below which an H.264 video (>=640px, >=5s) is already so heavily
/// compressed that re-encoding cannot shrink it meaningfully without visible degradation.
const LOW_BPP_SKIP_THRESHOLD: f64 = 0.032;

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

/// Layer 1 fast pre-filter: skip already heavily compressed H.264 sources without encoding.
pub fn should_skip_low_bpp(info: &VideoInfo, input_size: u64) -> Option<String> {
    let codec = info.codec_name.as_deref()?.to_ascii_lowercase();
    if codec != "h264" || info.width < 640 || info.height < 360 || info.duration < 5.0 {
        return None;
    }
    let bpp = info.bits_per_pixel(input_size);
    (bpp > 0.0 && bpp < LOW_BPP_SKIP_THRESHOLD).then(|| {
        format!(
            "源H.264码率已极低 (BPP={:.4} < {:.3})，视觉无损下无有效压缩空间",
            bpp, LOW_BPP_SKIP_THRESHOLD
        )
    })
}

/// Compute predicted total output size (bytes) separating pure video and audio bitrates.
pub fn predict_total_output_bytes(est_video_bps: f64, audio_bps: u64, duration: f64) -> f64 {
    ((est_video_bps + audio_bps as f64) / 8.0) * duration * MUX_OVERHEAD_FACTOR
}

/// Evaluate whether predicted output meets BOTH zero-bloat (`< input_size`)
/// and minimum net saving (`min_saving_percent`).
pub fn check_predicted_budget(
    est_video_bps: f64,
    audio_bps: u64,
    duration: f64,
    input_size: u64,
    min_saving_percent: u8,
) -> std::result::Result<f64, String> {
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

/// Verify encoded output duration matches original within tolerance (rejects truncated encodes).
pub fn is_duration_consistent(src_duration: f64, out_duration: f64) -> bool {
    if src_duration <= 0.0 {
        return out_duration > 0.0;
    }
    (src_duration - out_duration).abs() <= (src_duration * 0.03).max(2.0)
}

/// Layer 4 hard final gate: strictly `< input_size` and >= `min_saving_percent`.
pub fn check_final_size_gate(
    input_size: u64,
    output_size: u64,
    min_saving_percent: u8,
) -> std::result::Result<f64, String> {
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
    let actual_saving_pct = savings_percent(input_size, output_size);
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

struct SearchCtx<'a> {
    encoder: Encoder<'a>,
    sample: &'a ReferenceSample,
    preset: &'a str,
    ten_bit: bool,
    gpu_index: Option<usize>,
    metric: Option<QualityMetric>,
    target_xpsnr: f64,
    hw_slots: Option<&'a HwSlotPool>,
}

impl SearchCtx<'_> {
    fn probe(&self, q: u8) -> Result<ProbePoint> {
        let tmp = NamedTempFile::with_suffix(self.encoder.sample_extension())
            .context("无法创建样本编码临时文件")?;

        let mut params = EncodeParams::for_quality(q, self.preset, self.ten_bit, true);
        params.gpu_index = self.gpu_index;

        let bytes = self.encoder.encode_sample(
            self.sample.file.path(),
            &self.sample.info,
            tmp.path(),
            &params,
            self.hw_slots,
        )?;
        let video_bps = (bytes as f64 * 8.0) / self.sample.duration.max(0.5);

        let score = match self.metric {
            Some(m) => measure_metric_fast(m, self.sample.file.path(), tmp.path())?,
            None => 0.0,
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

    /// 3-step noise-knee-aware perceptual quality search: finds the highest Q (smallest file)
    /// that stays visually transparent while avoiding wasting bits on sensor noise.
    fn search(&self, profile: QualityProfile) -> Result<SearchResult> {
        let p0 = self.probe(profile.anchor_q)?;
        let Some(metric) = self.metric else {
            return Ok(SearchResult {
                chosen_q: p0.q,
                score_label: "无感知滤镜(使用基准Q)".to_string(),
                est_video_bps: p0.video_bps,
            });
        };

        let target = metric.target(self.target_xpsnr);
        let chosen = if p0.score >= target {
            // Clean footage: search higher Q (smaller output) in up to 2 binary-search steps.
            let mut best = p0;
            let mut lo = profile.anchor_q.saturating_add(1);
            let mut hi = profile.max_q;
            for _ in 0..2 {
                if lo > hi {
                    break;
                }
                let mid = lo + (hi - lo) / 2;
                let cand = self.probe(mid)?;
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
            // Complex/noisy footage: probe lower Q to distinguish real detail loss from noise floor.
            let q_low = profile
                .anchor_q
                .saturating_sub(profile.low_probe_delta)
                .max(profile.min_q);
            if q_low >= profile.anchor_q {
                p0
            } else {
                let p_low = self.probe(q_low)?;
                let gain = p_low.score - p0.score;
                let knee = metric.knee_threshold();
                if gain < knee {
                    log_debug(&format!(
                        "    [噪点拐点] Q={}->{} 增益仅 {:.2} < {:.2}，锁定拐点 Q={}",
                        p0.q, p_low.q, gain, knee, p0.q
                    ));
                    p0
                } else if p_low.score >= target {
                    let mid = q_low + (profile.anchor_q - q_low) / 2;
                    if mid > q_low && mid < profile.anchor_q {
                        let p_mid = self.probe(mid)?;
                        if p_mid.score >= target { p_mid } else { p_low }
                    } else {
                        p_low
                    }
                } else {
                    p_low
                }
            }
        };

        Ok(SearchResult {
            chosen_q: chosen.q,
            score_label: metric.format_score(chosen.score),
            est_video_bps: chosen.video_bps,
        })
    }
}

/// Select the active encoder backend, falling back to 10-bit CPU when the hardware encoder
/// cannot safely handle the video's resolution or HDR/10-bit depth.
fn select_active_encoder<'a>(
    encoder_cfg: &'a EncoderConfig,
    info: &VideoInfo,
) -> std::result::Result<Encoder<'a>, String> {
    let caps = &encoder_cfg.capabilities;
    let primary = Encoder::new(encoder_cfg.encoder_type, caps);

    if !primary.supports_video(info) {
        log_info(&format!(
            "  注意: 分辨率 {}x{} 低于硬件最小要求({}x{})，回退到CPU编码",
            info.width,
            info.height,
            Encoder::JETSON_MIN_DIMENSION,
            Encoder::JETSON_MIN_DIMENSION
        ));
        return Ok(Encoder::new(EncoderType::Cpu, caps));
    }

    if (info.is_hdr() || info.is_high_bit_depth()) && !primary.supports_10bit() {
        if caps.has_libx265 && caps.x265_10bit {
            log_info(
                "  注意: 源视频为10-bit/HDR而当前硬件仅支持8-bit，回退到10-bit CPU避免色深降级或HDR发灰",
            );
            return Ok(Encoder::new(EncoderType::Cpu, caps));
        }
        return Err("源视频为10-bit/HDR且无10-bit编码器，跳过避免色彩毁损".to_string());
    }

    Ok(primary)
}

/// Choose between local `/tmp` staging (for network mounts with ample free space)
/// and direct same-directory temp creation.
fn create_encode_temp(output_path: &Path, predicted_bytes: u64) -> Result<(NamedTempFile, bool)> {
    let needed_local = predicted_bytes
        .saturating_mul(2)
        .saturating_add(1024 * 1024 * 1024);
    let use_local_staging = is_network_filesystem(output_path)
        && available_space_bytes(&std::env::temp_dir()).is_some_and(|avail| avail >= needed_local);

    if use_local_staging {
        log_debug(
            "  [NFS优化] 输出目录为网络存储且本地临时盘空间充足，+faststart 与校验将在本地暂存后单次顺序回写",
        );
        Ok((
            NamedTempFile::with_suffix(".mp4").context("无法在本地临时目录创建暂存MP4")?,
            true,
        ))
    } else {
        Ok((create_temp_in_same_dir(output_path, ".mp4")?, false))
    }
}

/// Process a single video through the 4-layer zero-bloat perceptual compression pipeline.
pub fn process_video(
    input_path: &Path,
    cfg: &VideoConfig,
    gpu_index: Option<usize>,
    hw_slots: Option<&HwSlotPool>,
    state_cache: Option<&StateCache>,
) -> Result<ProcessResult> {
    let output_path = get_output_path(input_path);
    let preflight = match preflight_media_file(input_path, &output_path, state_cache, cfg.force)? {
        Ok(info) => info,
        Err(skipped) => return Ok(skipped),
    };
    let input_size = preflight.input_size;

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

    if !cfg.force {
        if let Some(reason) = video_info.is_modern_codec() {
            return preflight.record_skip(input_path, state_cache, reason.to_string(), true);
        }
        if let Some(reason) = should_skip_low_bpp(&video_info, input_size) {
            return preflight.record_skip(input_path, state_cache, reason, false);
        }
    } else if let Some(reason) = video_info.is_modern_codec() {
        log_info(&format!(
            "  (--force 已启用: {}，仍强制进行感知重编码评估)",
            reason
        ));
    }

    let active_encoder = match select_active_encoder(&cfg.encoder, &video_info) {
        Ok(enc) => enc,
        Err(reason) => return preflight.record_skip(input_path, state_cache, reason, false),
    };

    let ten_bit = active_encoder.supports_10bit();
    let audio_plan = AudioPlan::decide(&video_info, cfg.audio_bitrate);
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
        cfg.encoder.capabilities.has_libx264,
    )?;

    let search_ctx = SearchCtx {
        encoder: active_encoder,
        sample: &sample,
        preset: &cfg.encoder.preset,
        ten_bit,
        gpu_index,
        metric: QualityMetric::from_capabilities(&cfg.encoder.capabilities),
        target_xpsnr: cfg.encoder.target_xpsnr,
        hw_slots,
    };
    let search = search_ctx.search(active_encoder.quality_profile(cfg.max_crf))?;
    drop(sample); // Release sample temp file immediately before full encode.

    let predicted_saving = match check_predicted_budget(
        search.est_video_bps,
        audio_bps,
        duration,
        input_size,
        cfg.min_compression_ratio,
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
            return preflight.record_skip(input_path, state_cache, reason, false);
        }
    };

    log_info(&format!(
        "  感知最优: Q={} ({}) | 预估视频 {:.2}Mbps | 预估节省 {:.1}%",
        search.chosen_q,
        search.score_label,
        search.est_video_bps / 1_000_000.0,
        predicted_saving
    ));

    let mut full_params =
        EncodeParams::for_quality(search.chosen_q, &cfg.encoder.preset, ten_bit, false);
    full_params.gpu_index = gpu_index;

    let src_video_kbps = (video_info.effective_video_bitrate(input_size) / 1000) as u32;
    if src_video_kbps >= 1500
        && matches!(
            active_encoder.kind(),
            EncoderType::Gpu | EncoderType::JetsonGst
        )
    {
        full_params.maxrate_kbps = Some((src_video_kbps as f64 * 1.35) as u32);
    }

    let predicted_output_bytes = if duration > 0.0 {
        predict_total_output_bytes(search.est_video_bps, audio_bps, duration) as u64
    } else {
        input_size
    };

    // === Layer 3: Full hardware encode + real-time size watchdog ===
    let (encode_temp, use_local_staging) =
        create_encode_temp(&output_path, predicted_output_bytes)?;
    let watchdog_limit = compute_watchdog_limit(
        input_size,
        duration,
        audio_bps,
        active_encoder.kind() == EncoderType::JetsonGst,
    );

    log_debug(&format!(
        "  开始完整编码: {:?} | 实时熔断阈值: {:?}",
        full_params,
        watchdog_limit.map(format_size)
    ));

    let output_size = match active_encoder.full_encode(
        input_path,
        &video_info,
        encode_temp.path(),
        &full_params,
        audio_plan,
        watchdog_limit,
        hw_slots,
    )? {
        EncodeOutcome::AbortedSizeLimit { written_bytes } => {
            drop(encode_temp);
            return preflight.record_skip(
                input_path,
                state_cache,
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
    let actual_saving_pct =
        match check_final_size_gate(input_size, output_size, cfg.min_compression_ratio) {
            Ok(pct) => pct,
            Err(reason) => {
                drop(encode_temp);
                return preflight.record_skip(input_path, state_cache, reason, false);
            }
        };

    match probe_output_integrity(encode_temp.path()) {
        Some(out_dur) if is_duration_consistent(duration, out_dur) => {}
        Some(out_dur) => {
            drop(encode_temp);
            anyhow::bail!(
                "输出时长异常截断 (原片 {:.1}s vs 输出 {:.1}s)，已拒绝替换并保留原文件",
                duration,
                out_dur
            );
        }
        None => {
            drop(encode_temp);
            anyhow::bail!("输出文件视频流校验失败，已拒绝替换并保留原文件");
        }
    }

    let final_same_dir_temp = if use_local_staging {
        let mut dst_temp = create_temp_in_same_dir(&output_path, ".mp4")?;
        copy_file_to_temp(encode_temp.path(), &mut dst_temp)?;
        drop(encode_temp);
        dst_temp
    } else {
        encode_temp
    };

    atomic_replace_from_temp(
        final_same_dir_temp,
        input_path,
        &output_path,
        &preflight.metadata,
    )?;

    log_info(&format!(
        "  实际压缩: {} -> {} ({:.1}% 节省)",
        format_size(input_size),
        format_size(output_size),
        actual_saving_pct
    ));

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
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

        assert!(check_final_size_gate(1_000_000, 950_000, 15).is_err());
        assert!((check_final_size_gate(1_000_000, 950_000, 0).unwrap() - 5.0).abs() < 1e-6);
        assert!((check_final_size_gate(1_000_000, 600_000, 15).unwrap() - 40.0).abs() < 1e-6);
    }

    #[test]
    fn predicted_budget_separates_audio_and_rejects_bloat() {
        let pct = check_predicted_budget(4_000_000.0, 128_000, 10.0, 10_000_000, 15).unwrap();
        assert!(pct > 45.0 && pct < 50.0);
        assert!(check_predicted_budget(8_500_000.0, 128_000, 10.0, 10_000_000, 15).is_err());
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
            video_bitrate: 1_000_000,
            ..Default::default()
        };
        assert!(should_skip_low_bpp(&info, 8_000_000).is_some());

        let high_br = VideoInfo {
            video_bitrate: 10_000_000,
            ..info
        };
        assert!(should_skip_low_bpp(&high_br, 75_000_000).is_none());
    }
}
