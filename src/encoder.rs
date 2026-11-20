use crate::config::{EncodeParams, EncoderConfig, EncoderType, HardwareCapabilities, QualityProfile};
use crate::ffmpeg::{AudioPlan, VideoInfo, probe_duration};
use crate::logging::{log_debug, log_info};
use crate::signal::{
    SizeWatchdog, WatchedOutcome, run_command_cancellable, run_command_watched,
    run_pipeline_watched,
};
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use std::sync::{Condvar, Mutex};
use tempfile::NamedTempFile;

/// Bounded hardware-encoder permit pool so I/O, ffprobe, XPSNR evaluation, and remux
/// can run concurrently without idling dedicated video encoding hardware.
pub struct HwSlotPool {
    available: Mutex<usize>,
    cvar: Condvar,
}

pub struct HwPermit<'a> {
    pool: &'a HwSlotPool,
}

impl HwSlotPool {
    pub fn new(slots: usize) -> Self {
        Self {
            available: Mutex::new(slots.max(1)),
            cvar: Condvar::new(),
        }
    }

    pub fn acquire(&self) -> HwPermit<'_> {
        let mut avail = self.available.lock().unwrap_or_else(|e| e.into_inner());
        while *avail == 0 {
            avail = self.cvar.wait(avail).unwrap_or_else(|e| e.into_inner());
        }
        *avail -= 1;
        HwPermit { pool: self }
    }
}

impl Drop for HwPermit<'_> {
    fn drop(&mut self) {
        let mut avail = self.pool.available.lock().unwrap_or_else(|e| e.into_inner());
        *avail += 1;
        self.pool.cvar.notify_one();
    }
}

/// Result of a watched full-encode invocation.
#[derive(Debug)]
pub enum EncodeOutcome {
    Completed(u64),
    AbortedSizeLimit { written_bytes: u64 },
}

/// Strategy trait encapsulating backend-specific video encoding across
/// Jetson official GStreamer nvv4l2, Desktop NVENC, QSV, VAAPI, AMF, VideoToolbox,
/// legacy NVMPI, and CPU libx265.
pub trait VideoEncoder: Send + Sync {
    fn encoder_type(&self) -> EncoderType;

    /// Whether this encoder can safely handle the given video's resolution/properties.
    fn supports_video(&self, _info: &VideoInfo) -> bool {
        true
    }

    /// Whether this encoder produces 10-bit HEVC (Main10) output.
    fn supports_10bit(&self) -> bool {
        false
    }

    /// Whether this encoder supports direct QP/CQ/CRF perceptual search.
    /// Legacy Jetson NVMPI only supports bitrate control and returns `false`.
    fn supports_quality_search(&self) -> bool {
        true
    }

    /// Native quality search bounds clamped by user's `max_crf`.
    fn quality_profile(&self, max_crf: u8) -> QualityProfile;

    /// Encode a pure-video (`-an`) reference sample for perceptual quality & bitrate estimation.
    /// Returns output file size in bytes.
    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64>;

    /// Full encode with optional real-time output size watchdog and hardware-slot gating.
    #[allow(clippy::too_many_arguments)]
    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome>;
}

pub struct CpuEncoder {
    caps: HardwareCapabilities,
}

pub struct GpuEncoder {
    caps: HardwareCapabilities,
}

pub struct JetsonGstEncoder {
    caps: HardwareCapabilities,
}

pub struct JetsonNvmpiEncoder;

pub struct QsvEncoder;

pub struct VaapiEncoder {
    device: String,
}

pub struct AmfEncoder;

pub struct VideoToolboxEncoder;

pub fn create_encoder(config: &EncoderConfig) -> Box<dyn VideoEncoder> {
    create_encoder_of(config.encoder_type, &config.capabilities)
}

pub fn create_encoder_of(
    encoder_type: EncoderType,
    caps: &HardwareCapabilities,
) -> Box<dyn VideoEncoder> {
    match encoder_type {
        EncoderType::Cpu => Box::new(CpuEncoder { caps: caps.clone() }),
        EncoderType::Gpu => Box::new(GpuEncoder { caps: caps.clone() }),
        EncoderType::JetsonGst => Box::new(JetsonGstEncoder { caps: caps.clone() }),
        EncoderType::JetsonNvmpi => Box::new(JetsonNvmpiEncoder),
        EncoderType::Qsv => Box::new(QsvEncoder),
        EncoderType::Vaapi => Box::new(VaapiEncoder {
            device: caps
                .vaapi_device
                .clone()
                .unwrap_or_else(|| "/dev/dri/renderD128".to_string()),
        }),
        EncoderType::Amf => Box::new(AmfEncoder),
        EncoderType::VideoToolbox => Box::new(VideoToolboxEncoder),
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
        _ => "p6",
    }
}

/// Tail of stderr for concise error reporting.
fn stderr_tail(output: &std::process::Output) -> String {
    let text = String::from_utf8_lossy(&output.stderr);
    let mut lines: Vec<&str> = text.lines().rev().take(15).collect();
    lines.reverse();
    lines.join("\n")
}

/// Append HDR / color metadata tags to FFmpeg output when known.
fn append_color_tags(cmd: &mut Command, info: &VideoInfo) {
    if let Some(cp) = &info.color_primaries {
        cmd.arg("-color_primaries").arg(cp);
    }
    if let Some(ct) = &info.color_transfer {
        cmd.arg("-color_trc").arg(ct);
    }
    if let Some(cs) = &info.color_space {
        cmd.arg("-colorspace").arg(cs);
    }
}

/// Execute a standard FFmpeg encode command with optional size watchdog and HW slot permit.
#[allow(clippy::too_many_arguments)]
fn run_ffmpeg_watched(
    pre_input_args: &[String],
    input_path: &Path,
    video_args: &[String],
    info: &VideoInfo,
    audio_plan: AudioPlan,
    output_path: &Path,
    max_bytes: Option<u64>,
    hw_slots: Option<&HwSlotPool>,
    context_msg: &str,
) -> Result<EncodeOutcome> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-hide_banner", "-v", "error"]);
    for arg in pre_input_args {
        cmd.arg(arg);
    }
    cmd.arg("-i").arg(input_path);
    cmd.args(["-map", "0:v:0"]);
    if !matches!(audio_plan, AudioPlan::None) {
        cmd.args(["-map", "0:a:0?"]);
    }
    for arg in video_args {
        cmd.arg(arg);
    }
    append_color_tags(&mut cmd, info);
    audio_plan.append_ffmpeg_args(&mut cmd);
    cmd.args(["-map_metadata", "0", "-tag:v", "hvc1", "-movflags", "+faststart"]);
    cmd.arg(output_path);

    let watchdog = max_bytes.map(|limit| SizeWatchdog {
        watch_path: output_path,
        max_bytes: limit,
    });

    let outcome = {
        let _permit = hw_slots.map(|pool| pool.acquire());
        run_command_watched(&mut cmd, watchdog, context_msg)?
    };

    match outcome {
        WatchedOutcome::SizeLimitExceeded { written_bytes } => {
            Ok(EncodeOutcome::AbortedSizeLimit { written_bytes })
        }
        WatchedOutcome::Finished(out) => {
            if !out.status.success() {
                anyhow::bail!("{}: {}", context_msg, stderr_tail(&out));
            }
            let size = std::fs::metadata(output_path)
                .context("无法读取编码输出文件")?
                .len();
            Ok(EncodeOutcome::Completed(size))
        }
    }
}

// ============================================================================
// 1. CPU Encoder (libx265, 10-bit + psychovisual tuning)
// ============================================================================

impl CpuEncoder {
    fn build_video_args(&self, params: &EncodeParams) -> Vec<String> {
        let preset = if params.fast_preview {
            "fast".to_string()
        } else {
            params.preset.clone()
        };
        let mut args = vec![
            "-c:v".to_string(),
            "libx265".to_string(),
            "-crf".to_string(),
            params.quality.to_string(),
            "-preset".to_string(),
            preset,
        ];
        if params.ten_bit && self.caps.x265_10bit {
            args.push("-pix_fmt".to_string());
            args.push("yuv420p10le".to_string());
        } else if let Some(fmt) = &params.pix_fmt {
            args.push("-pix_fmt".to_string());
            args.push(fmt.clone());
        }
        // Psychovisual tuning: dark-scene AQ + limited SAO (prevents plastic smoothing) + mild deblock.
        // Note: removed invalid `fast=1` that x265 previously warned as unknown option.
        args.push("-x265-params".to_string());
        args.push("aq-mode=3:limit-sao=1:deblock=-1,-1:log-level=error".to_string());
        args
    }
}

impl VideoEncoder for CpuEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Cpu
    }

    fn supports_10bit(&self) -> bool {
        self.caps.x265_10bit
    }

    fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        let max_q = max_crf.clamp(22, 32);
        QualityProfile {
            min_q: 18,
            anchor_q: 24.min(max_q),
            max_q,
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        _hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let video_args = self.build_video_args(params);
        match run_ffmpeg_watched(
            &[],
            sample_path,
            &video_args,
            sample_info,
            AudioPlan::None,
            output_path,
            None,
            None,
            "CPU样本预览编码失败",
        )? {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        _hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let video_args = self.build_video_args(params);
        run_ffmpeg_watched(
            &[],
            input_path,
            &video_args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            None,
            "CPU完整编码失败",
        )
    }
}

// ============================================================================
// 2. Desktop NVIDIA GPU Encoder (hevc_nvenc — zero-copy CUDA + fixed -bf/-b:v 0 + 10-bit)
// ============================================================================

impl GpuEncoder {
    pub fn build_decode_args(&self, info: &VideoInfo, gpu_index: Option<usize>) -> Vec<String> {
        if !info.is_nvdec_compatible() {
            return vec![];
        }
        let mut args = vec![
            "-hwaccel".to_string(),
            "cuda".to_string(),
            "-hwaccel_output_format".to_string(),
            "cuda".to_string(),
        ];
        if let Some(idx) = gpu_index {
            args.push("-hwaccel_device".to_string());
            args.push(idx.to_string());
        }
        args
    }

    pub fn build_video_args(&self, params: &EncodeParams, using_cuda_frames: bool) -> Vec<String> {
        let cq = params.quality.clamp(1, 51);
        let nvenc_preset = if params.fast_preview {
            "p4"
        } else {
            map_nvenc_preset(&params.preset)
        };
        let tune = if !params.fast_preview && self.caps.nvenc_uhq {
            "uhq"
        } else {
            "hq"
        };

        let mut args = vec![
            "-c:v".to_string(),
            "hevc_nvenc".to_string(),
            "-preset".to_string(),
            nvenc_preset.to_string(),
            "-tune".to_string(),
            tune.to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-cq".to_string(),
            cq.to_string(),
            // CRITICAL: -b:v 0 enables true constant-quality VBR (previously `-b 4` forced 4 bps!).
            "-b:v".to_string(),
            "0".to_string(),
            "-g".to_string(),
            "240".to_string(),
            "-rc-lookahead".to_string(),
            "32".to_string(),
            "-spatial_aq".to_string(),
            "1".to_string(),
            "-temporal_aq".to_string(),
            "1".to_string(),
            "-aq-strength".to_string(),
            "8".to_string(),
        ];

        if let Some(idx) = params.gpu_index {
            args.push("-gpu".to_string());
            args.push(idx.to_string());
        }

        if self.caps.nvenc_bframes {
            args.push("-b_ref_mode".to_string());
            args.push("middle".to_string());
            args.push("-bf".to_string());
            args.push("4".to_string());
        }

        if !params.fast_preview && self.caps.nvenc_multipass {
            args.push("-multipass".to_string());
            args.push("qres".to_string());
        }

        if params.ten_bit && self.caps.nvenc_10bit {
            args.push("-profile:v".to_string());
            args.push("main10".to_string());
            if using_cuda_frames {
                // Keep zero-copy CUDA frames on GPU and promote 8->10 bit inside NVENC.
                args.push("-highbitdepth".to_string());
                args.push("1".to_string());
            } else {
                args.push("-pix_fmt".to_string());
                args.push("p010le".to_string());
            }
        }

        if let Some(maxrate) = params.maxrate_kbps {
            args.push("-maxrate".to_string());
            args.push(format!("{}k", maxrate));
            args.push("-bufsize".to_string());
            args.push(format!("{}k", maxrate.saturating_mul(2)));
        }

        args
    }
}

impl VideoEncoder for GpuEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Gpu
    }

    fn supports_10bit(&self) -> bool {
        self.caps.nvenc_10bit
    }

    fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        let max_q = max_crf.clamp(24, 36);
        QualityProfile {
            min_q: 18,
            anchor_q: 26.min(max_q),
            max_q,
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let decode_args = self.build_decode_args(sample_info, params.gpu_index);
        let using_cuda = !decode_args.is_empty();
        let video_args = self.build_video_args(params, using_cuda);
        let res = run_ffmpeg_watched(
            &decode_args,
            sample_path,
            &video_args,
            sample_info,
            AudioPlan::None,
            output_path,
            None,
            hw_slots,
            "GPU样本预览编码失败",
        );
        match res {
            Ok(EncodeOutcome::Completed(sz)) => Ok(sz),
            Ok(EncodeOutcome::AbortedSizeLimit { .. }) => unreachable!(),
            Err(_) if using_cuda => {
                // Graceful fallback to SW decode -> NVENC encode if CUDA decode fails on this stream.
                let sw_video_args = self.build_video_args(params, false);
                match run_ffmpeg_watched(
                    &[],
                    sample_path,
                    &sw_video_args,
                    sample_info,
                    AudioPlan::None,
                    output_path,
                    None,
                    hw_slots,
                    "GPU样本预览编码失败(软解回退)",
                )? {
                    EncodeOutcome::Completed(sz) => Ok(sz),
                    EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
                }
            }
            Err(e) => Err(e),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let decode_args = self.build_decode_args(info, params.gpu_index);
        let using_cuda = !decode_args.is_empty();
        let video_args = self.build_video_args(params, using_cuda);
        let res = run_ffmpeg_watched(
            &decode_args,
            input_path,
            &video_args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            hw_slots,
            "GPU完整编码失败",
        );
        match res {
            Ok(outcome) => Ok(outcome),
            Err(e) if using_cuda && !is_nvenc_session_limit_error(&e.to_string()) => {
                log_info("  (NVDEC硬解不兼容此流，自动切回软解+NVENC硬编)");
                let sw_video_args = self.build_video_args(params, false);
                run_ffmpeg_watched(
                    &[],
                    input_path,
                    &sw_video_args,
                    info,
                    audio_plan,
                    output_path,
                    max_bytes,
                    hw_slots,
                    "GPU完整编码失败(软解回退)",
                )
            }
            Err(e) => Err(e),
        }
    }
}

// ============================================================================
// 3. Official Jetson GStreamer Backend (nvv4l2h265enc 10-bit + nvv4l2decoder NVMM zero-copy)
//    Zero custom-compiled FFmpeg required!
// ============================================================================

impl JetsonGstEncoder {
    pub const MIN_DIMENSION: u32 = 160;

    fn build_nvv4l2_enc_args(&self, params: &EncodeParams) -> Vec<String> {
        let qp = params.quality.clamp(16, 44);
        let qp_p_min = qp.saturating_sub(1).max(1);
        let qp_p_max = qp.saturating_add(1).min(51);
        let qp_i_min = qp.saturating_sub(3).max(1);
        let qp_i_max = qp.saturating_sub(1).max(1);
        let qp_range = format!(
            "{},{}:{},{}:{},{}",
            qp_p_min, qp_p_max, qp_i_min, qp_i_max, qp_p_min, qp_p_max
        );

        // Empirically validated on Jetson Orin:
        // - FastPreset (2) = ~142fps, ideal for rapid perceptual search samples
        // - MediumPreset (3) == SlowPreset (4) in output quality/size (~80fps, beats libx265 CRF23!)
        let preset_level = if params.fast_preview {
            "2"
        } else {
            match params.preset.to_lowercase().as_str() {
                "ultrafast" | "superfast" => "1",
                "veryfast" | "faster" | "fast" => "2",
                "medium" => "3",
                "slow" | "slower" | "veryslow" => "4",
                _ => "3",
            }
        };

        let profile = if params.ten_bit && self.caps.jetson_gst_10bit {
            "1" // Main10
        } else {
            "0" // Main
        };

        // Generous VBR bitrate ceiling so qp-range governs perceptual quality cleanly.
        let peak_bps: u64 = params
            .maxrate_kbps
            .map(|k| (k as u64).saturating_mul(1000).max(2_000_000))
            .unwrap_or(50_000_000);
        let avg_bps: u64 = (peak_bps * 7 / 10).max(1_500_000);

        vec![
            "nvv4l2h265enc".to_string(),
            "maxperf-enable=1".to_string(),
            "iframeinterval=240".to_string(),
            format!("profile={}", profile),
            "control-rate=0".to_string(), // variable_bitrate
            format!("preset-level={}", preset_level),
            format!("bitrate={}", avg_bps),
            format!("peak-bitrate={}", peak_bps),
            format!("qp-range={}", qp_range),
        ]
    }

    fn nvmm_caps(&self, params: &EncodeParams) -> String {
        if params.ten_bit && self.caps.jetson_gst_10bit {
            "video/x-raw(memory:NVMM),format=P010_10LE".to_string()
        } else {
            "video/x-raw(memory:NVMM),format=NV12".to_string()
        }
    }

    /// Path A: Full NVMM zero-copy decode -> color/10-bit VIC -> NVENC H.265 -> Matroska (preserves PTS/VFR).
    #[allow(clippy::too_many_arguments)]
    fn run_zerocopy_encode(
        &self,
        input_path: &Path,
        demux: &str,
        parser: &str,
        out_mkv: &Path,
        params: &EncodeParams,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let mut cmd = Command::new("gst-launch-1.0");
        cmd.args(["-q", "-e", "filesrc"])
            .arg(format!("location={}", input_path.to_string_lossy()))
            .args([
                "!",
                demux,
                "!",
                parser,
                "!",
                "nvv4l2decoder",
                "!",
                "nvvidconv",
                "!",
            ])
            .arg(self.nvmm_caps(params))
            .arg("!");
        for arg in self.build_nvv4l2_enc_args(params) {
            cmd.arg(arg);
        }
        cmd.args(["!", "h265parse", "!", "matroskamux", "!", "filesink"])
            .arg(format!("location={}", out_mkv.to_string_lossy()));

        let watchdog = max_bytes.map(|limit| SizeWatchdog {
            watch_path: out_mkv,
            max_bytes: limit,
        });

        let outcome = {
            let _permit = hw_slots.map(|pool| pool.acquire());
            run_command_watched(&mut cmd, watchdog, "Jetson GStreamer零拷贝硬编失败")?
        };

        match outcome {
            WatchedOutcome::SizeLimitExceeded { written_bytes } => {
                Ok(EncodeOutcome::AbortedSizeLimit { written_bytes })
            }
            WatchedOutcome::Finished(out) => {
                if !out.status.success() {
                    anyhow::bail!("Jetson GStreamer零拷贝硬编失败: {}", stderr_tail(&out));
                }
                let size = std::fs::metadata(out_mkv)
                    .context("无法读取GStreamer输出文件")?
                    .len();
                if size == 0 {
                    anyhow::bail!("Jetson GStreamer输出为空");
                }
                Ok(EncodeOutcome::Completed(size))
            }
        }
    }

    /// Path B: Universal compatibility pipe — standard FFmpeg SW decode (I420, CFR, even dims, noautorotate)
    /// -> OS pipe -> GStreamer `fdsrc ! rawvideoparse format=i420 ! nvvidconv(NVMM) ! nvv4l2h265enc ! matroskamux`.
    fn run_pipe_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        out_mkv: &Path,
        params: &EncodeParams,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let even_w = (info.width + (info.width % 2)).max(Self::MIN_DIMENSION);
        let even_h = (info.height + (info.height % 2)).max(Self::MIN_DIMENSION);
        let fps_num = (info.fps.clamp(1.0, 240.0) * 1000.0).round() as u32;
        let fps_str = format!("{}/1000", fps_num);

        let mut producer = Command::new("ffmpeg");
        producer
            .args(["-hide_banner", "-v", "error", "-noautorotate", "-i"])
            .arg(input_path)
            .args(["-map", "0:v:0", "-an", "-fps_mode", "cfr", "-r", &fps_str]);
        if !info.width.is_multiple_of(2) || !info.height.is_multiple_of(2) {
            producer.args(["-vf", &format!("pad={}:{}:0:0", even_w, even_h)]);
        }
        producer.args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]);

        let mut consumer = Command::new("gst-launch-1.0");
        consumer
            .args([
                "-q",
                "-e",
                "fdsrc",
                "!",
                "rawvideoparse",
                "format=i420",
                &format!("width={}", even_w),
                &format!("height={}", even_h),
                &format!("framerate={}", fps_str),
                "!",
                "nvvidconv",
                "!",
            ])
            .arg(self.nvmm_caps(params))
            .arg("!");
        for arg in self.build_nvv4l2_enc_args(params) {
            consumer.arg(arg);
        }
        consumer
            .args(["!", "h265parse", "!", "matroskamux", "!", "filesink"])
            .arg(format!("location={}", out_mkv.to_string_lossy()));

        let watchdog = max_bytes.map(|limit| SizeWatchdog {
            watch_path: out_mkv,
            max_bytes: limit,
        });

        let outcome = {
            let _permit = hw_slots.map(|pool| pool.acquire());
            run_pipeline_watched(
                &mut producer,
                &mut consumer,
                watchdog,
                "Jetson软解管道+硬件编码失败",
            )?
        };

        match outcome {
            WatchedOutcome::SizeLimitExceeded { written_bytes } => {
                Ok(EncodeOutcome::AbortedSizeLimit { written_bytes })
            }
            WatchedOutcome::Finished(out) => {
                if !out.status.success() {
                    anyhow::bail!("Jetson软解管道+硬件编码失败: {}", stderr_tail(&out));
                }
                let size = std::fs::metadata(out_mkv)
                    .context("无法读取GStreamer管道输出")?
                    .len();
                if size == 0 {
                    anyhow::bail!("Jetson GStreamer管道输出为空");
                }
                Ok(EncodeOutcome::Completed(size))
            }
        }
    }

    fn encode_video_track(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        out_mkv: &Path,
        params: &EncodeParams,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        // GStreamer `filesrc location=...` parses pipeline tokens directly; route exotic paths
        // containing `!` or quotes safely through Path B (FFmpeg stdin pipe).
        let path_safe_for_gst = input_path
            .to_str()
            .map(|s| !s.contains('!') && !s.contains('"') && !s.contains('\''))
            .unwrap_or(false);

        if path_safe_for_gst
            && self.caps.jetson_gst_decoder
            && let Some((demux, parser)) = info.gst_zerocopy_elements()
        {
            match self.run_zerocopy_encode(
                input_path, demux, parser, out_mkv, params, max_bytes, hw_slots,
            ) {
                Ok(outcome) => return Ok(outcome),
                Err(e) => {
                    log_debug(&format!(
                        "Jetson零拷贝路径失败，回退到标准FFmpeg软解管道: {}",
                        e
                    ));
                    let _ = std::fs::remove_file(out_mkv);
                }
            }
        }

        self.run_pipe_encode(input_path, info, out_mkv, params, max_bytes, hw_slots)
    }

    /// Standard FFmpeg remux: millisecond-fast `-c:v copy` combining the NVENC video track
    /// with original audio (smart copy/transcode), metadata, rotation, HDR tags, `hvc1`, `faststart`.
    fn remux_to_mp4(
        video_mkv: &Path,
        original_input: &Path,
        info: &VideoInfo,
        audio_plan: AudioPlan,
        output_mp4: &Path,
    ) -> Result<u64> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-v", "error"]);
        if info.rotation != 0 {
            cmd.arg("-display_rotation:v:0")
                .arg(info.rotation.to_string());
        }
        cmd.arg("-i")
            .arg(video_mkv)
            .arg("-i")
            .arg(original_input)
            .args(["-map", "0:v:0"]);
        if !matches!(audio_plan, AudioPlan::None) {
            cmd.args(["-map", "1:a:0?"]);
        }
        cmd.args(["-c:v", "copy", "-tag:v", "hvc1"]);
        append_color_tags(&mut cmd, info);
        audio_plan.append_ffmpeg_args(&mut cmd);
        cmd.args(["-map_metadata", "1", "-movflags", "+faststart"])
            .arg(output_mp4);

        let out = run_command_cancellable(&mut cmd, "FFmpeg合流封装MP4失败")?;
        if !out.status.success() {
            anyhow::bail!("FFmpeg合流封装MP4失败: {}", stderr_tail(&out));
        }

        Ok(std::fs::metadata(output_mp4)
            .context("无法读取最终封装MP4")?
            .len())
    }
}

impl VideoEncoder for JetsonGstEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::JetsonGst
    }

    fn supports_video(&self, info: &VideoInfo) -> bool {
        info.width >= Self::MIN_DIMENSION && info.height >= Self::MIN_DIMENSION
    }

    fn supports_10bit(&self) -> bool {
        self.caps.jetson_gst_10bit
    }

    fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        // Empirically calibrated on Jetson Orin nvv4l2h265enc 10-bit:
        // QP 28-32 is the visual transparency sweet spot; cap strictly to preserve quality.
        let max_q = max_crf.clamp(26, 36);
        QualityProfile {
            min_q: 22,
            anchor_q: 28.min(max_q),
            max_q,
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        match self.encode_video_track(sample_path, sample_info, output_path, params, None, hw_slots)?
        {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        // Stage 1 (holds HW slot): hardware encode pure video to temp .mkv with size watchdog.
        let tmp_mkv = NamedTempFile::with_suffix(".mkv").context("无法创建临时视频轨文件")?;
        let mkv_outcome = self.encode_video_track(
            input_path,
            info,
            tmp_mkv.path(),
            params,
            max_bytes,
            hw_slots,
        )?;

        match mkv_outcome {
            EncodeOutcome::AbortedSizeLimit { written_bytes } => {
                Ok(EncodeOutcome::AbortedSizeLimit { written_bytes })
            }
            EncodeOutcome::Completed(_) => {
                // Stage 2 (HW slot already released!): standard FFmpeg millisecond remux to MP4.
                let final_size = Self::remux_to_mp4(
                    tmp_mkv.path(),
                    input_path,
                    info,
                    audio_plan,
                    output_path,
                )?;
                Ok(EncodeOutcome::Completed(final_size))
            }
        }
    }
}

// ============================================================================
// 4. Legacy Jetson NVMPI Backend (kept as opt-in/fallback; bitrate bugs fixed)
// ============================================================================

impl JetsonNvmpiEncoder {
    pub const MIN_DIMENSION: u32 = 160;

    fn build_video_args(&self, params: &EncodeParams) -> Vec<String> {
        let target_kbps = params.target_bitrate_kbps.unwrap_or(match params.quality {
            18..=22 => 2000,
            23..=26 => 1400,
            27..=30 => 1000,
            _ => 750,
        });

        let mut args = vec![
            "-c:v".to_string(),
            "hevc_nvmpi".to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-b:v".to_string(),
            format!("{}k", target_kbps),
        ];

        // Avoid pseudo-CBR: only set maxrate when strictly greater than target_bitrate.
        if let Some(maxrate) = params.maxrate_kbps {
            let safe_max = maxrate.max(target_kbps.saturating_mul(13) / 10);
            args.push("-maxrate".to_string());
            args.push(format!("{}k", safe_max));
            args.push("-bufsize".to_string());
            args.push(format!("{}k", safe_max.saturating_mul(2)));
        }

        args
    }
}

impl VideoEncoder for JetsonNvmpiEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::JetsonNvmpi
    }

    fn supports_video(&self, info: &VideoInfo) -> bool {
        info.width >= Self::MIN_DIMENSION && info.height >= Self::MIN_DIMENSION
    }

    fn supports_quality_search(&self) -> bool {
        false
    }

    fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        QualityProfile {
            min_q: 20,
            anchor_q: 24,
            max_q: max_crf.clamp(24, 32),
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let video_args = self.build_video_args(params);
        match run_ffmpeg_watched(
            &[],
            sample_path,
            &video_args,
            sample_info,
            AudioPlan::None,
            output_path,
            None,
            hw_slots,
            "NVMPI样本编码失败",
        )? {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let video_args = self.build_video_args(params);
        run_ffmpeg_watched(
            &[],
            input_path,
            &video_args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            hw_slots,
            "NVMPI完整编码失败",
        )
    }
}

// ============================================================================
// 5. Cross-platform Hardware Encoders: Intel QSV, VA-API, AMD AMF, Apple VideoToolbox
// ============================================================================

impl VideoEncoder for QsvEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Qsv
    }

    fn supports_10bit(&self) -> bool {
        true
    }

    fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        let max_q = max_crf.clamp(24, 34);
        QualityProfile {
            min_q: 18,
            anchor_q: 25.min(max_q),
            max_q,
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let preset = if params.fast_preview { "fast" } else { "slow" };
        let args = vec![
            "-c:v".into(),
            "hevc_qsv".into(),
            "-preset".into(),
            preset.into(),
            "-global_quality".into(),
            params.quality.to_string(),
            "-g".into(),
            "240".into(),
        ];
        match run_ffmpeg_watched(
            &[],
            sample_path,
            &args,
            sample_info,
            AudioPlan::None,
            output_path,
            None,
            hw_slots,
            "QSV样本预览编码失败",
        )? {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let mut args = vec![
            "-c:v".into(),
            "hevc_qsv".into(),
            "-preset".into(),
            "slow".into(),
            "-global_quality".into(),
            params.quality.to_string(),
            "-g".into(),
            "240".into(),
        ];
        if params.ten_bit {
            args.push("-profile:v".into());
            args.push("main10".into());
            args.push("-pix_fmt".into());
            args.push("p010le".into());
        }
        run_ffmpeg_watched(
            &[],
            input_path,
            &args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            hw_slots,
            "QSV完整编码失败",
        )
    }
}

impl VideoEncoder for VaapiEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Vaapi
    }

    fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        let max_q = max_crf.clamp(24, 34);
        QualityProfile {
            min_q: 18,
            anchor_q: 26.min(max_q),
            max_q,
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let pre = vec!["-vaapi_device".into(), self.device.clone()];
        let args = vec![
            "-vf".into(),
            "format=nv12,hwupload".into(),
            "-c:v".into(),
            "hevc_vaapi".into(),
            "-rc_mode".into(),
            "CQP".into(),
            "-qp".into(),
            params.quality.to_string(),
            "-g".into(),
            "240".into(),
        ];
        match run_ffmpeg_watched(
            &pre,
            sample_path,
            &args,
            sample_info,
            AudioPlan::None,
            output_path,
            None,
            hw_slots,
            "VAAPI样本预览编码失败",
        )? {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let pre = vec!["-vaapi_device".into(), self.device.clone()];
        let args = vec![
            "-vf".into(),
            "format=nv12,hwupload".into(),
            "-c:v".into(),
            "hevc_vaapi".into(),
            "-rc_mode".into(),
            "CQP".into(),
            "-qp".into(),
            params.quality.to_string(),
            "-g".into(),
            "240".into(),
        ];
        run_ffmpeg_watched(
            &pre,
            input_path,
            &args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            hw_slots,
            "VAAPI完整编码失败",
        )
    }
}

impl VideoEncoder for AmfEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::Amf
    }

    fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        let max_q = max_crf.clamp(24, 34);
        QualityProfile {
            min_q: 18,
            anchor_q: 26.min(max_q),
            max_q,
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let q = params.quality;
        let args = vec![
            "-c:v".into(),
            "hevc_amf".into(),
            "-quality".into(),
            if params.fast_preview {
                "balanced"
            } else {
                "quality"
            }
            .into(),
            "-rc".into(),
            "cqp".into(),
            "-qp_i".into(),
            q.saturating_sub(2).max(1).to_string(),
            "-qp_p".into(),
            q.to_string(),
            "-g".into(),
            "240".into(),
        ];
        match run_ffmpeg_watched(
            &[],
            sample_path,
            &args,
            sample_info,
            AudioPlan::None,
            output_path,
            None,
            hw_slots,
            "AMF样本预览编码失败",
        )? {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let q = params.quality;
        let args = vec![
            "-c:v".into(),
            "hevc_amf".into(),
            "-quality".into(),
            "quality".into(),
            "-rc".into(),
            "cqp".into(),
            "-qp_i".into(),
            q.saturating_sub(2).max(1).to_string(),
            "-qp_p".into(),
            q.to_string(),
            "-g".into(),
            "240".into(),
        ];
        run_ffmpeg_watched(
            &[],
            input_path,
            &args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            hw_slots,
            "AMF完整编码失败",
        )
    }
}

impl VideoEncoder for VideoToolboxEncoder {
    fn encoder_type(&self) -> EncoderType {
        EncoderType::VideoToolbox
    }

    fn supports_10bit(&self) -> bool {
        true
    }

    fn quality_profile(&self, _max_crf: u8) -> QualityProfile {
        // VideoToolbox `-q:v` is inverted (1..100, higher=better); map internal `q`
        // via `vt_q = 100 - q * 2` so higher `q` consistently means smaller output!
        QualityProfile {
            min_q: 12,
            anchor_q: 20,
            max_q: 30,
            low_probe_delta: 4,
        }
    }

    fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let vt_q = 100u8
            .saturating_sub(params.quality.saturating_mul(2))
            .clamp(20, 90);
        let args = vec![
            "-c:v".into(),
            "hevc_videotoolbox".into(),
            "-q:v".into(),
            vt_q.to_string(),
            "-g".into(),
            "240".into(),
        ];
        match run_ffmpeg_watched(
            &[],
            sample_path,
            &args,
            sample_info,
            AudioPlan::None,
            output_path,
            None,
            hw_slots,
            "VideoToolbox样本预览编码失败",
        )? {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let vt_q = 100u8
            .saturating_sub(params.quality.saturating_mul(2))
            .clamp(20, 90);
        let mut args = vec![
            "-c:v".into(),
            "hevc_videotoolbox".into(),
            "-q:v".into(),
            vt_q.to_string(),
            "-g".into(),
            "240".into(),
        ];
        if params.ten_bit {
            args.push("-profile:v".into());
            args.push("main10".into());
        }
        run_ffmpeg_watched(
            &[],
            input_path,
            &args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            hw_slots,
            "VideoToolbox完整编码失败",
        )
    }
}

// ============================================================================
// Single-Command Pure-Video Reference Sample Extraction (replaces 14-process loop!)
// ============================================================================

/// Extracted near-lossless pure-video reference sample for perceptual search.
pub struct ReferenceSample {
    pub file: NamedTempFile,
    pub duration: f64,
    pub info: VideoInfo,
}

/// Extract a representative pure-video (`-an`) reference sample in ONE FFmpeg command:
/// - Short videos (`<= 18s`): up to 6s from the middle (or full clip if `<= 8s`).
/// - Longer videos (`> 18s`): 3 uniform 2s segments at 20%, 50%, 80% (6s total), concatenated
///   cleanly into a single near-lossless H.264 MP4 (`crf 12`, `ultrafast`) so hardware decoders
///   (`nvv4l2decoder`, CUDA NVDEC) can decode it at 150+ fps during every search iteration.
pub fn extract_reference_sample(
    input_path: &Path,
    info: &VideoInfo,
    has_libx264: bool,
) -> Result<ReferenceSample> {
    let temp = NamedTempFile::with_suffix(".mp4").context("无法创建样本临时文件")?;
    let dur = info.duration;
    let even_w = (info.width + (info.width % 2)).max(2);
    let even_h = (info.height + (info.height % 2)).max(2);
    let pad_expr = if !info.width.is_multiple_of(2) || !info.height.is_multiple_of(2) {
        format!(",pad={}:{}:0:0", even_w, even_h)
    } else {
        String::new()
    };

    let build_codec_args = |cmd: &mut Command| {
        if has_libx264 {
            cmd.args(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "12"]);
        } else {
            cmd.args(["-c:v", "mpeg4", "-q:v", "2"]);
        }
        cmd.args(["-an", "-movflags", "+faststart"]);
    };

    let mut extracted_ok = false;

    if dur > 18.0 {
        // 3 uniform segments x 2s = 6s at 20%, 50%, 80% in ONE FFmpeg command!
        let starts = [
            (dur * 0.20 - 1.0).max(0.0),
            (dur * 0.50 - 1.0).max(0.0),
            (dur * 0.80 - 1.0).max(0.0),
        ];
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-v", "error"]);
        for s in starts {
            cmd.args(["-noautorotate", "-ss", &format!("{:.2}", s), "-t", "2", "-i"])
                .arg(input_path);
        }
        let filt = format!(
            "[0:v]setpts=PTS-STARTPTS,format=yuv420p{p}[v0];\
             [1:v]setpts=PTS-STARTPTS,format=yuv420p{p}[v1];\
             [2:v]setpts=PTS-STARTPTS,format=yuv420p{p}[v2];\
             [v0][v1][v2]concat=n=3:v=1:a=0[v]",
            p = pad_expr
        );
        cmd.args(["-filter_complex", &filt, "-map", "[v]"]);
        build_codec_args(&mut cmd);
        cmd.arg(temp.path());

        if let Ok(out) = run_command_cancellable(&mut cmd, "提取多段感知样本失败")
            && out.status.success()
            && probe_duration(temp.path()).unwrap_or(0.0) >= 2.0
        {
            extracted_ok = true;
        }
    }

    if !extracted_ok {
        // Single-segment extraction (short clips or sparse-keyframe ASF/WMV fallback).
        let sample_len = if dur > 0.0 { dur.clamp(1.0, 6.0) } else { 6.0 };
        let start = if dur > 10.0 {
            ((dur - sample_len) / 2.0).max(0.0)
        } else {
            0.0
        };
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-v", "error", "-noautorotate", "-i"])
            .arg(input_path);
        if start > 0.5 {
            cmd.args(["-ss", &format!("{:.2}", start)]);
        }
        cmd.args([
            "-t",
            &format!("{:.2}", sample_len),
            "-map",
            "0:v:0",
            "-vf",
            &format!("format=yuv420p{}", pad_expr),
        ]);
        build_codec_args(&mut cmd);
        cmd.arg(temp.path());

        let out = run_command_cancellable(&mut cmd, "提取感知样本失败")?;
        if !out.status.success() {
            anyhow::bail!("提取感知样本失败: {}", stderr_tail(&out));
        }
    }

    let actual_dur = probe_duration(temp.path())
        .filter(|&d| d >= 0.5)
        .unwrap_or_else(|| dur.clamp(1.0, 6.0));

    let sample_info = VideoInfo {
        codec_name: Some(if has_libx264 { "h264" } else { "mpeg4" }.to_string()),
        width: even_w,
        height: even_h,
        duration: actual_dur,
        fps: info.fps,
        pix_fmt: Some("yuv420p".to_string()),
        format_name: Some("mov,mp4,m4a,3gp,3g2,mj2".to_string()),
        has_audio: false,
        ..Default::default()
    };

    Ok(ReferenceSample {
        file: temp,
        duration: actual_dur,
        info: sample_info,
    })
}

/// Detect NVIDIA consumer GPU NVENC concurrent session limit errors so the scheduler
/// can dynamically cap concurrency and retry instead of marking videos failed.
pub fn is_nvenc_session_limit_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("openencodesessionex failed")
        || lower.contains("incompatible client key")
        || lower.contains("too many concurrent sessions")
        || (lower.contains("nvenc") && lower.contains("out of memory"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_encoder_fixes_bframes_and_sets_zero_bitrate_for_vbr_cq() {
        let caps = HardwareCapabilities {
            nvenc_ok: true,
            nvenc_10bit: true,
            nvenc_bframes: true,
            nvenc_multipass: true,
            nvenc_uhq: false,
            ..Default::default()
        };
        let gpu = GpuEncoder { caps };
        let params = EncodeParams::for_quality(26, "slow", true, false);
        let args = gpu.build_video_args(&params, true);

        // Must use -bf 4 (NOT broken `-b 4`!) and explicit `-b:v 0`
        assert!(args.windows(2).any(|w| w == ["-bf", "4"]));
        assert!(args.windows(2).any(|w| w == ["-b:v", "0"]));
        assert!(!args.windows(2).any(|w| w == ["-b", "4"]));
        assert!(args.windows(2).any(|w| w == ["-preset", "p6"]));
        assert!(args.windows(2).any(|w| w == ["-multipass", "qres"]));
        assert!(args.windows(2).any(|w| w == ["-profile:v", "main10"]));
        assert!(args.windows(2).any(|w| w == ["-highbitdepth", "1"]));
        assert!(!args.windows(2).any(|w| w == ["-pix_fmt", "p010le"]));
    }

    #[test]
    fn gpu_encoder_uses_p010le_when_sw_decoding_and_cuda_when_nvdec_compatible() {
        let caps = HardwareCapabilities {
            nvenc_ok: true,
            nvenc_10bit: true,
            ..Default::default()
        };
        let gpu = GpuEncoder { caps };
        let h264_info = VideoInfo {
            codec_name: Some("h264".into()),
            pix_fmt: Some("yuv420p".into()),
            ..Default::default()
        };
        let wmv_info = VideoInfo {
            codec_name: Some("wmv2".into()),
            pix_fmt: Some("yuv420p".into()),
            ..Default::default()
        };
        assert!(gpu
            .build_decode_args(&h264_info, Some(0))
            .windows(2)
            .any(|w| w == ["-hwaccel_output_format", "cuda"]));
        assert!(gpu.build_decode_args(&wmv_info, Some(0)).is_empty());

        let params = EncodeParams::for_quality(26, "medium", true, false);
        let sw_args = gpu.build_video_args(&params, false);
        assert!(sw_args.windows(2).any(|w| w == ["-pix_fmt", "p010le"]));
    }

    #[test]
    fn cpu_encoder_drops_invalid_fast_param_and_uses_10bit() {
        let caps = HardwareCapabilities {
            has_libx265: true,
            x265_10bit: true,
            ..Default::default()
        };
        let cpu = CpuEncoder { caps };
        let params = EncodeParams::for_quality(24, "medium", true, false);
        let args = cpu.build_video_args(&params);
        assert!(args.windows(2).any(|w| w == ["-pix_fmt", "yuv420p10le"]));
        assert!(!args.iter().any(|a| a.contains("fast=1")));
        assert!(args.iter().any(|a| a.contains("aq-mode=3")));
    }

    #[test]
    fn jetson_gst_rejects_tiny_resolution_and_uses_fast_preview_preset() {
        let caps = HardwareCapabilities {
            jetson_gst_ok: true,
            jetson_gst_10bit: true,
            jetson_gst_decoder: true,
            ..Default::default()
        };
        let enc = JetsonGstEncoder { caps };
        let small = VideoInfo {
            width: 120,
            height: 120,
            ..Default::default()
        };
        let hd = VideoInfo {
            width: 1920,
            height: 1080,
            ..Default::default()
        };
        assert!(!enc.supports_video(&small));
        assert!(enc.supports_video(&hd));

        let preview_args =
            enc.build_nvv4l2_enc_args(&EncodeParams::for_quality(28, "medium", true, true));
        let full_args =
            enc.build_nvv4l2_enc_args(&EncodeParams::for_quality(28, "medium", true, false));
        assert!(preview_args.contains(&"preset-level=2".to_string()));
        assert!(full_args.contains(&"preset-level=3".to_string()));
        assert!(full_args.contains(&"profile=1".to_string()));
        assert!(full_args.contains(&"qp-range=27,29:25,27:27,29".to_string()));
    }

    #[test]
    fn nvenc_session_limit_detection() {
        assert!(is_nvenc_session_limit_error(
            "[hevc_nvenc] OpenEncodeSessionEx failed: incompatible client key (21)"
        ));
        assert!(!is_nvenc_session_limit_error(
            "Invalid data found when processing input"
        ));
    }
}
