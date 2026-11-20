use crate::config::{EncodeParams, EncoderType};
use crate::ffmpeg::VideoInfo;
use crate::logging::log_info;
use crate::signal::run_command_cancellable;
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use tempfile::NamedTempFile;

pub struct PreviewResult {
    pub output_size_per_second: f64, // Bytes per second after preview encoding
}

/// Strategy trait encapsulating backend-specific FFmpeg behavior for CPU, NVENC, and Jetson NVMPI.
pub trait VideoEncoder: Send + Sync {
    fn encoder_type(&self) -> EncoderType;
    fn decode_args(&self) -> Vec<String>;
    fn encode_args(&self, params: &EncodeParams) -> Vec<String>;

    /// Whether this encoder can safely handle the given video's properties.
    fn supports_video(&self, _info: &VideoInfo) -> bool {
        true
    }

    /// Optional cap on preview encode duration in seconds.
    fn preview_max_seconds(&self) -> Option<u32> {
        None
    }

    /// Encoder used for the preview pass (Jetson uses CPU to estimate baseline bitrate).
    fn preview_encoder(&self) -> Box<dyn VideoEncoder>;

    /// Resolve final full-encode parameters from preview output and chosen CRF.
    fn resolve_full_params(
        &self,
        final_crf: u8,
        preset: &str,
        _preview: &PreviewResult,
        _info: &VideoInfo,
        _input_size: u64,
    ) -> EncodeParams {
        EncodeParams::from_crf(final_crf, preset)
    }
}

pub struct CpuEncoder;
pub struct GpuEncoder;
pub struct JetsonEncoder;

pub fn create_encoder(encoder_type: EncoderType) -> Box<dyn VideoEncoder> {
    match encoder_type {
        EncoderType::Cpu => Box::new(CpuEncoder),
        EncoderType::Gpu => Box::new(GpuEncoder),
        EncoderType::Jetson => Box::new(JetsonEncoder),
    }
}

/// Map x265-style presets (or direct `p1`..`p7`) to NVENC presets.
pub fn map_nvenc_preset(preset: &str) -> &'static str {
    match preset.to_lowercase().as_str() {
        "p1" | "ultrafast" | "superfast" => "p1",
        "p2" | "veryfast" => "p2",
        "p3" | "faster" => "p3",
        "p4" | "fast" => "p4",
        "p5" | "medium" => "p5",
        "p6" | "slow" => "p6",
        "p7" | "slower" | "veryslow" => "p7",
        _ => "p7",
    }
}

impl VideoEncoder for CpuEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Cpu
    }

    fn decode_args(&self) -> Vec<String> {
        vec![]
    }

    fn encode_args(&self, params: &EncodeParams) -> Vec<String> {
        vec![
            "-c:v".to_string(),
            "libx265".to_string(),
            "-crf".to_string(),
            params.crf.to_string(),
            "-preset".to_string(),
            params.preset.clone(),
            "-x265-params".to_string(),
            "fast=1:log-level=error".to_string(),
        ]
    }

    fn preview_encoder(&self) -> Box<dyn VideoEncoder> {
        Box::new(CpuEncoder)
    }
}

impl VideoEncoder for GpuEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Gpu
    }

    fn decode_args(&self) -> Vec<String> {
        vec!["-hwaccel".to_string(), "cuda".to_string()]
    }

    fn encode_args(&self, params: &EncodeParams) -> Vec<String> {
        // NVENC CQ scale is shifted relative to x265 CRF; clamp to valid [0, 51] range.
        let cq = params.crf.saturating_add(7).min(51);
        let nvenc_preset = map_nvenc_preset(&params.preset);
        vec![
            "-c:v".to_string(),
            "hevc_nvenc".to_string(),
            "-preset".to_string(),
            nvenc_preset.to_string(),
            "-tune".to_string(),
            "hq".to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-cq".to_string(),
            cq.to_string(),
            "-rc-lookahead".to_string(),
            "32".to_string(),
            "-b_ref_mode".to_string(),
            "middle".to_string(),
            "-b".to_string(),
            "4".to_string(),
            "-spatial_aq".to_string(),
            "1".to_string(),
            "-temporal_aq".to_string(),
            "1".to_string(),
            "-aq-strength".to_string(),
            "8".to_string(),
        ]
    }

    fn preview_encoder(&self) -> Box<dyn VideoEncoder> {
        Box::new(GpuEncoder)
    }
}

impl JetsonEncoder {
    /// Jetson NVMPI HEVC encoder enters an infinite error loop below ~160x160.
    pub const MIN_DIMENSION: u32 = 160;
    const MAX_PREVIEW_SECONDS: u32 = 10;
}

impl VideoEncoder for JetsonEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Jetson
    }

    fn decode_args(&self) -> Vec<String> {
        // Jetson NVMPI: software decode. h264_nvmpi + hevc_nvmpi deadlocks
        // after ~2000 frames due to hardware resource contention.
        vec![]
    }

    fn encode_args(&self, params: &EncodeParams) -> Vec<String> {
        // Jetson NVMPI ignores -qp; must use -b:v bitrate control.
        let target_bitrate_kbps = params.target_bitrate_kbps.unwrap_or(match params.crf {
            18..=22 => 1500,
            23..=26 => 1000,
            27..=30 => 700,
            31..=35 => 500,
            _ => 800,
        });

        let mut args = vec![
            "-c:v".to_string(),
            "hevc_nvmpi".to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-b:v".to_string(),
            format!("{}k", target_bitrate_kbps),
        ];

        if let Some(maxrate) = params.maxrate_kbps {
            args.push("-maxrate".to_string());
            args.push(format!("{}k", maxrate));
            args.push("-bufsize".to_string());
            args.push(format!("{}k", maxrate.saturating_mul(2)));
        }

        args
    }

    fn supports_video(&self, info: &VideoInfo) -> bool {
        info.width >= Self::MIN_DIMENSION && info.height >= Self::MIN_DIMENSION
    }

    fn preview_max_seconds(&self) -> Option<u32> {
        Some(Self::MAX_PREVIEW_SECONDS)
    }

    fn preview_encoder(&self) -> Box<dyn VideoEncoder> {
        Box::new(CpuEncoder)
    }

    fn resolve_full_params(
        &self,
        final_crf: u8,
        preset: &str,
        preview: &PreviewResult,
        info: &VideoInfo,
        input_size: u64,
    ) -> EncodeParams {
        let cpu_bitrate_bps = preview.output_size_per_second * 8.0;
        let ideal_jetson_bitrate_kbps = (cpu_bitrate_bps * 2.0 / 1000.0).max(100.0) as u32;

        let original_bitrate_kbps = if info.bitrate > 0 {
            (info.bitrate / 1000) as u32
        } else if info.duration > 0.0 {
            ((input_size as f64 / info.duration) * 8.0 / 1000.0) as u32
        } else {
            ideal_jetson_bitrate_kbps
        };
        let max_bitrate_kbps = ((original_bitrate_kbps as f64 * 0.7) as u32).max(100);

        let (target_bitrate_kbps, maxrate_kbps) = if ideal_jetson_bitrate_kbps > max_bitrate_kbps {
            log_info(&format!(
                "  Jetson: 理想比特率 {}kbps > 原始70% ({}kbps)，使用70%限制",
                ideal_jetson_bitrate_kbps, max_bitrate_kbps
            ));
            (max_bitrate_kbps, Some(max_bitrate_kbps))
        } else {
            log_info(&format!(
                "  Jetson目标比特率: {}kbps (CPU基准×2.0)",
                ideal_jetson_bitrate_kbps
            ));
            (ideal_jetson_bitrate_kbps, None)
        };

        EncodeParams {
            crf: final_crf,
            preset: preset.to_string(),
            pix_fmt: None,
            target_bitrate_kbps: Some(target_bitrate_kbps),
            maxrate_kbps,
        }
    }
}

/// A `-c copy` segment is only usable if it actually contains a video stream.
/// ASF/WMV input seeking can land where no keyframe arrives within the
/// requested window; ffmpeg then writes an empty segment while still exiting 0.
fn segment_has_video(path: &Path) -> bool {
    let mut cmd = Command::new("ffprobe");
    cmd.arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=codec_type")
        .arg("-of")
        .arg("csv=p=0")
        .arg(path);
    run_command_cancellable(&mut cmd, "校验分段视频流失败")
        .map(|o| o.status.success() && !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false)
}

/// Tail of ffmpeg's stderr for concise error reporting.
fn stderr_tail(output: &std::process::Output) -> String {
    let text = String::from_utf8_lossy(&output.stderr);
    let mut lines: Vec<&str> = text.lines().rev().take(15).collect();
    lines.reverse();
    lines.join("\n")
}

/// Shared FFmpeg encode runner used by both preview and full encode paths.
fn run_ffmpeg_encode(
    encoder: &dyn VideoEncoder,
    input_args: &[String],
    params: &EncodeParams,
    audio_bitrate: u32,
    force_mp4_format: bool,
    output_path: &Path,
    context_msg: &str,
) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");
    for arg in encoder.decode_args() {
        cmd.arg(arg);
    }
    for arg in input_args {
        cmd.arg(arg);
    }
    for arg in encoder.encode_args(params) {
        cmd.arg(arg);
    }
    if let Some(fmt) = &params.pix_fmt {
        cmd.arg("-pix_fmt").arg(fmt);
    }
    if force_mp4_format {
        cmd.arg("-f").arg("mp4");
    }

    if audio_bitrate > 0 {
        cmd.arg("-c:a")
            .arg("aac")
            .arg("-b:a")
            .arg(format!("{}k", audio_bitrate));
    } else {
        cmd.arg("-c:a").arg("copy");
    }

    cmd.arg(output_path);

    let output = run_command_cancellable(&mut cmd, context_msg)?;
    if !output.status.success() {
        anyhow::bail!("{}: {}", context_msg, stderr_tail(&output));
    }
    Ok(())
}

pub fn preview_encode(
    input_path: &Path,
    params: &EncodeParams,
    encoder: &dyn VideoEncoder,
    audio_bitrate: u32,
    duration: u32,
    video_duration: f64,
) -> Result<PreviewResult> {
    let temp_output = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;
    let temp_output_path = temp_output.path();

    if video_duration <= duration as f64 * 1.5 {
        let temp_segment = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;

        let mut extract_cmd = Command::new("ffmpeg");
        extract_cmd
            .arg("-y")
            .arg("-i")
            .arg(input_path)
            .arg("-t")
            .arg(duration.to_string())
            .arg("-c")
            .arg("copy")
            .arg(temp_segment.path());

        let extract_out = run_command_cancellable(&mut extract_cmd, "执行FFmpeg提取预览片段失败")?;
        if !extract_out.status.success() {
            anyhow::bail!("提取预览片段失败");
        }

        let input_args = vec![
            "-i".to_string(),
            temp_segment.path().to_string_lossy().into_owned(),
        ];
        run_ffmpeg_encode(
            encoder,
            &input_args,
            params,
            audio_bitrate,
            true,
            temp_output_path,
            "预览编码失败",
        )?;
    } else {
        // Five-segment uniform sampling at 10%, 30%, 50%, 70%, 90%
        const NUM_SEGMENTS: u32 = 5;
        const SAMPLE_POSITIONS: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];

        let segment_duration = (duration / NUM_SEGMENTS).max(1);
        let remaining = duration % NUM_SEGMENTS;

        let seg_starts: Vec<f64> = SAMPLE_POSITIONS
            .iter()
            .map(|&pos| (video_duration * pos - segment_duration as f64 / 2.0).max(0.0))
            .collect();

        let seg_durs: Vec<u32> = (0..NUM_SEGMENTS)
            .map(|i| {
                if i == NUM_SEGMENTS - 1 {
                    segment_duration + remaining
                } else {
                    segment_duration
                }
            })
            .collect();

        let temp_segs: Vec<NamedTempFile> = (0..NUM_SEGMENTS)
            .map(|i| {
                NamedTempFile::with_suffix(".mkv").context(format!("无法创建临时文件{}", i + 1))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut extract_success = true;
        for (i, temp_seg) in temp_segs.iter().enumerate() {
            let mut extract = Command::new("ffmpeg");
            extract
                .arg("-y")
                .arg("-ss")
                .arg(format!("{:.1}", seg_starts[i]))
                .arg("-i")
                .arg(input_path)
                .arg("-t")
                .arg(seg_durs[i].to_string())
                .arg("-c")
                .arg("copy")
                .arg(temp_seg.path());

            let out = run_command_cancellable(&mut extract, &format!("提取片段{}失败", i + 1))?;
            if !out.status.success() || !segment_has_video(temp_seg.path()) {
                extract_success = false;
                break;
            }
        }

        if !extract_success {
            // Sparse-keyframe sources (e.g. WMV2/ASF): retry with output-side seek + re-encode.
            log_info("    (copy分段为空，改用解码重编码提取)");
            extract_success = true;
            for (i, temp_seg) in temp_segs.iter().enumerate() {
                let mut cmd = Command::new("ffmpeg");
                cmd.arg("-y")
                    .arg("-i")
                    .arg(input_path)
                    .arg("-ss")
                    .arg(format!("{:.1}", seg_starts[i]))
                    .arg("-t")
                    .arg(seg_durs[i].to_string())
                    .arg("-c:v")
                    .arg("libx264")
                    .arg("-preset")
                    .arg("ultrafast")
                    .arg("-crf")
                    .arg("18");
                if audio_bitrate > 0 {
                    cmd.arg("-c:a")
                        .arg("aac")
                        .arg("-b:a")
                        .arg(format!("{}k", audio_bitrate));
                } else {
                    cmd.arg("-c:a").arg("copy");
                }
                cmd.arg(temp_seg.path());

                let out = run_command_cancellable(
                    &mut cmd,
                    &format!("提取片段{}失败(重编码)", i + 1),
                )?;
                if !out.status.success() || !segment_has_video(temp_seg.path()) {
                    extract_success = false;
                    break;
                }
            }
        }

        if !extract_success {
            anyhow::bail!("提取多段预览片段失败");
        }

        let concat_list = NamedTempFile::with_suffix(".txt").context("无法创建concat列表")?;
        let concat_content = temp_segs
            .iter()
            .map(|seg| format!("file '{}'\n", seg.path().display()))
            .collect::<String>();
        std::fs::write(concat_list.path(), concat_content).context("无法写入concat列表")?;

        let input_args = vec![
            "-f".to_string(),
            "concat".to_string(),
            "-safe".to_string(),
            "0".to_string(),
            "-i".to_string(),
            concat_list.path().to_string_lossy().into_owned(),
        ];

        run_ffmpeg_encode(
            encoder,
            &input_args,
            params,
            audio_bitrate,
            true,
            temp_output_path,
            "预览编码失败",
        )?;
    }

    let output_size = std::fs::metadata(temp_output_path)
        .context("无法读取预览输出文件")?
        .len() as f64;

    // Divide by real encoded duration to avoid keyframe-snap inflation.
    let actual_duration =
        crate::ffmpeg::probe_duration(temp_output_path).unwrap_or(duration as f64);

    let output_size_per_second = output_size / actual_duration.max(0.1);

    Ok(PreviewResult {
        output_size_per_second,
    })
}

pub fn full_encode(
    input_path: &Path,
    output_path: &Path,
    params: &EncodeParams,
    encoder: &dyn VideoEncoder,
    audio_bitrate: u32,
) -> Result<u64> {
    let input_args = vec![
        "-i".to_string(),
        input_path.to_string_lossy().into_owned(),
    ];

    run_ffmpeg_encode(
        encoder,
        &input_args,
        params,
        audio_bitrate,
        false,
        output_path,
        "FFmpeg转换失败",
    )?;

    Ok(std::fs::metadata(output_path)
        .context("无法读取输出文件")?
        .len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_encoder_maps_preset_and_clamps_cq() {
        let gpu = GpuEncoder;
        let params = EncodeParams::from_crf(48, "fast");
        let args = gpu.encode_args(&params);
        assert!(args.windows(2).any(|w| w == ["-preset", "p4"]));
        assert!(args.windows(2).any(|w| w == ["-cq", "51"]));
    }

    #[test]
    fn jetson_encoder_rejects_small_resolution() {
        let jetson = JetsonEncoder;
        let small = VideoInfo {
            codec_name: Some("h264".into()),
            width: 120,
            height: 120,
            duration: 10.0,
            bitrate: 500_000,
            pix_fmt: None,
        };
        let normal = VideoInfo {
            width: 1280,
            height: 720,
            ..small.clone()
        };
        assert!(!jetson.supports_video(&small));
        assert!(jetson.supports_video(&normal));
    }

    #[test]
    fn jetson_encoder_caps_bitrate_at_70_percent() {
        let jetson = JetsonEncoder;
        let info = VideoInfo {
            codec_name: Some("h264".into()),
            width: 1920,
            height: 1080,
            duration: 100.0,
            bitrate: 2_000_000, // 2000 kbps -> 70% cap = 1400 kbps
            pix_fmt: None,
        };
        // Preview says 125_000 B/s = 1000 kbps -> * 2.0 = 2000 kbps > 1400 kbps
        let preview = PreviewResult {
            output_size_per_second: 125_000.0,
        };
        let params = jetson.resolve_full_params(23, "medium", &preview, &info, 25_000_000);
        assert_eq!(params.target_bitrate_kbps, Some(1400));
        assert_eq!(params.maxrate_kbps, Some(1400));
    }
}
