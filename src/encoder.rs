use crate::config::{EncodeParams, EncoderType, HardwareCapabilities, QualityProfile};
use crate::ffmpeg::{AudioPlan, VideoInfo, probe_duration, probe_has_audio};
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

struct SlotState {
    in_use: usize,
    limit: usize,
}

/// Bounded hardware-encoder permit pool with dynamic session-limit shrinking.
/// Allows I/O, ffprobe, XPSNR evaluation, and remux to overlap with hardware encoding
/// without idling dedicated video silicon.
pub struct HwSlotPool {
    state: Mutex<SlotState>,
    cvar: Condvar,
}

pub struct HwPermit<'a> {
    pool: &'a HwSlotPool,
}

impl HwSlotPool {
    pub fn new(slots: usize) -> Self {
        Self {
            state: Mutex::new(SlotState {
                in_use: 0,
                limit: slots.max(1),
            }),
            cvar: Condvar::new(),
        }
    }

    pub fn limit(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).limit
    }

    pub fn acquire(&self) -> HwPermit<'_> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while st.in_use >= st.limit {
            st = self.cvar.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.in_use += 1;
        HwPermit { pool: self }
    }

    /// Shrink concurrency limit safely when an NVENC session-limit error occurs.
    /// Using `observed_limit` prevents multiple workers failing in the same wave
    /// from over-shrinking the cap below the true concurrent capacity.
    pub fn shrink_below(&self, observed_limit: usize) -> usize {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let target = st.in_use.min(observed_limit.saturating_sub(1)).max(1);
        if target < st.limit {
            st.limit = target;
        }
        st.limit
    }
}

impl Drop for HwPermit<'_> {
    fn drop(&mut self) {
        let mut st = self.pool.state.lock().unwrap_or_else(|e| e.into_inner());
        st.in_use = st.in_use.saturating_sub(1);
        self.pool.cvar.notify_all();
    }
}

/// Result of a watched full-encode invocation.
#[derive(Debug)]
pub enum EncodeOutcome {
    Completed(u64),
    AbortedSizeLimit { written_bytes: u64 },
}

/// Lightweight enum-dispatched encoder backend (zero heap allocation, zero capability cloning).
#[derive(Debug, Clone, Copy)]
pub struct Encoder<'a> {
    kind: EncoderType,
    caps: &'a HardwareCapabilities,
}

impl<'a> Encoder<'a> {
    pub const JETSON_MIN_DIMENSION: u32 = 160;

    pub fn new(kind: EncoderType, caps: &'a HardwareCapabilities) -> Self {
        Self { kind, caps }
    }

    pub fn kind(&self) -> EncoderType {
        self.kind
    }

    pub fn supports_video(&self, info: &VideoInfo) -> bool {
        match self.kind {
            EncoderType::JetsonGst => {
                info.width >= Self::JETSON_MIN_DIMENSION
                    && info.height >= Self::JETSON_MIN_DIMENSION
            }
            _ => true,
        }
    }

    pub fn supports_10bit(&self) -> bool {
        match self.kind {
            EncoderType::Cpu => self.caps.x265_10bit,
            EncoderType::Gpu => self.caps.nvenc_10bit,
            EncoderType::JetsonGst => self.caps.jetson_gst_10bit,
            EncoderType::Qsv | EncoderType::VideoToolbox => true,
            EncoderType::Vaapi | EncoderType::Amf => false,
        }
    }

    pub fn sample_extension(&self) -> &'static str {
        if self.kind == EncoderType::JetsonGst {
            ".mkv"
        } else {
            ".mp4"
        }
    }

    pub fn quality_profile(&self, max_crf: u8) -> QualityProfile {
        match self.kind {
            EncoderType::Cpu => {
                let max_q = max_crf.clamp(22, 32);
                QualityProfile {
                    min_q: 18,
                    anchor_q: 24.min(max_q),
                    max_q,
                    low_probe_delta: 4,
                }
            }
            EncoderType::Gpu => {
                let max_q = max_crf.clamp(24, 36);
                QualityProfile {
                    min_q: 18,
                    anchor_q: 26.min(max_q),
                    max_q,
                    low_probe_delta: 4,
                }
            }
            EncoderType::JetsonGst => {
                let max_q = max_crf.clamp(26, 36);
                QualityProfile {
                    min_q: 22,
                    anchor_q: 28.min(max_q),
                    max_q,
                    low_probe_delta: 4,
                }
            }
            EncoderType::Qsv => {
                let max_q = max_crf.clamp(24, 34);
                QualityProfile {
                    min_q: 18,
                    anchor_q: 25.min(max_q),
                    max_q,
                    low_probe_delta: 4,
                }
            }
            EncoderType::Vaapi | EncoderType::Amf => {
                let max_q = max_crf.clamp(24, 34);
                QualityProfile {
                    min_q: 18,
                    anchor_q: 26.min(max_q),
                    max_q,
                    low_probe_delta: 4,
                }
            }
            EncoderType::VideoToolbox => QualityProfile {
                // VideoToolbox `-q:v` is inverted (1..100, higher=better); mapped as `100 - q * 2`.
                min_q: 12,
                anchor_q: 20,
                max_q: 30,
                low_probe_delta: 4,
            },
        }
    }

    /// Encode a pure-video (`-an`) reference sample for perceptual search.
    pub fn encode_sample(
        &self,
        sample_path: &Path,
        sample_info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<u64> {
        let outcome = if self.kind == EncoderType::JetsonGst {
            // Skip MP4 remux for sample probing; measure Matroska output directly.
            self.jetson_encode_video_track(
                sample_path,
                sample_info,
                output_path,
                AudioPlan::None,
                None,
                params,
                None,
                hw_slots,
            )?
        } else {
            self.full_encode(
                sample_path,
                sample_info,
                output_path,
                params,
                AudioPlan::None,
                None,
                hw_slots,
            )?
        };
        match outcome {
            EncodeOutcome::Completed(sz) => Ok(sz),
            EncodeOutcome::AbortedSizeLimit { .. } => unreachable!(),
        }
    }

    /// Full encode with optional real-time output size watchdog and hardware-slot gating.
    #[allow(clippy::too_many_arguments)]
    pub fn full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        match self.kind {
            EncoderType::JetsonGst => self.jetson_full_encode(
                input_path,
                info,
                output_path,
                params,
                audio_plan,
                max_bytes,
                hw_slots,
            ),
            EncoderType::Gpu => self.gpu_encode(
                input_path,
                info,
                output_path,
                params,
                audio_plan,
                max_bytes,
                hw_slots,
            ),
            _ => {
                let (pre_args, video_args) = self.build_standard_ffmpeg_args(params);
                let slots = self.kind.is_hardware().then_some(hw_slots).flatten();
                let ctx = format!("{}编码失败", self.kind.short_id());
                run_ffmpeg_watched(
                    &pre_args,
                    input_path,
                    &video_args,
                    info,
                    audio_plan,
                    output_path,
                    max_bytes,
                    slots,
                    &ctx,
                )
            }
        }
    }

    // ------------------------------------------------------------------------
    // Standard FFmpeg backends (CPU, QSV, VAAPI, AMF, VideoToolbox)
    // ------------------------------------------------------------------------

    fn build_standard_ffmpeg_args(&self, params: &EncodeParams) -> (Vec<String>, Vec<String>) {
        match self.kind {
            EncoderType::Cpu => {
                let preset = if params.fast_preview {
                    "fast"
                } else {
                    params.preset.as_str()
                };
                let mut args = vec_str(&[
                    "-c:v",
                    "libx265",
                    "-crf",
                    &params.quality.to_string(),
                    "-preset",
                    preset,
                ]);
                if params.ten_bit && self.caps.x265_10bit {
                    args.extend(vec_str(&["-pix_fmt", "yuv420p10le"]));
                }
                args.extend(vec_str(&[
                    "-x265-params",
                    "aq-mode=3:limit-sao=1:deblock=-1,-1:log-level=error",
                ]));
                (vec![], args)
            }
            EncoderType::Qsv => {
                let preset = if params.fast_preview { "fast" } else { "slow" };
                let mut args = vec_str(&[
                    "-c:v",
                    "hevc_qsv",
                    "-preset",
                    preset,
                    "-global_quality",
                    &params.quality.to_string(),
                    "-g",
                    "240",
                ]);
                if params.ten_bit {
                    args.extend(vec_str(&["-profile:v", "main10", "-pix_fmt", "p010le"]));
                }
                (vec![], args)
            }
            EncoderType::Vaapi => {
                let device = self
                    .caps
                    .vaapi_device
                    .as_deref()
                    .unwrap_or("/dev/dri/renderD128");
                let pre = vec_str(&["-vaapi_device", device]);
                let args = vec_str(&[
                    "-vf",
                    "format=nv12,hwupload",
                    "-c:v",
                    "hevc_vaapi",
                    "-rc_mode",
                    "CQP",
                    "-qp",
                    &params.quality.to_string(),
                    "-g",
                    "240",
                ]);
                (pre, args)
            }
            EncoderType::Amf => {
                let q = params.quality;
                let quality_mode = if params.fast_preview {
                    "balanced"
                } else {
                    "quality"
                };
                let args = vec_str(&[
                    "-c:v",
                    "hevc_amf",
                    "-quality",
                    quality_mode,
                    "-rc",
                    "cqp",
                    "-qp_i",
                    &q.saturating_sub(2).max(1).to_string(),
                    "-qp_p",
                    &q.to_string(),
                    "-g",
                    "240",
                ]);
                (vec![], args)
            }
            EncoderType::VideoToolbox => {
                let vt_q = 100u8
                    .saturating_sub(params.quality.saturating_mul(2))
                    .clamp(20, 90);
                let mut args = vec_str(&[
                    "-c:v",
                    "hevc_videotoolbox",
                    "-q:v",
                    &vt_q.to_string(),
                    "-g",
                    "240",
                ]);
                if params.ten_bit {
                    args.extend(vec_str(&["-profile:v", "main10"]));
                }
                (vec![], args)
            }
            EncoderType::Gpu | EncoderType::JetsonGst => unreachable!(),
        }
    }

    // ------------------------------------------------------------------------
    // Desktop NVIDIA NVENC (zero-copy CUDA decode with automatic SW fallback)
    // ------------------------------------------------------------------------

    fn build_gpu_decode_args(&self, info: &VideoInfo, gpu_index: Option<usize>) -> Vec<String> {
        if !info.is_nvdec_compatible() {
            return vec![];
        }
        let mut args = vec_str(&["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]);
        if let Some(idx) = gpu_index {
            args.extend(vec_str(&["-hwaccel_device", &idx.to_string()]));
        }
        args
    }

    fn build_gpu_video_args(&self, params: &EncodeParams, using_cuda_frames: bool) -> Vec<String> {
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

        let mut args = vec_str(&[
            "-c:v",
            "hevc_nvenc",
            "-preset",
            nvenc_preset,
            "-tune",
            tune,
            "-rc",
            "vbr",
            "-cq",
            &cq.to_string(),
            "-b:v",
            "0",
            "-g",
            "240",
            "-rc-lookahead",
            "32",
            "-spatial_aq",
            "1",
            "-temporal_aq",
            "1",
            "-aq-strength",
            "8",
        ]);

        if let Some(idx) = params.gpu_index {
            args.extend(vec_str(&["-gpu", &idx.to_string()]));
        }
        if self.caps.nvenc_bframes {
            args.extend(vec_str(&["-b_ref_mode", "middle", "-bf", "4"]));
        }
        if !params.fast_preview && self.caps.nvenc_multipass {
            args.extend(vec_str(&["-multipass", "qres"]));
        }
        if params.ten_bit && self.caps.nvenc_10bit {
            args.extend(vec_str(&["-profile:v", "main10"]));
            if using_cuda_frames {
                args.extend(vec_str(&["-highbitdepth", "1"]));
            } else {
                args.extend(vec_str(&["-pix_fmt", "p010le"]));
            }
        }
        if let Some(maxrate) = params.maxrate_kbps {
            args.extend(vec_str(&[
                "-maxrate",
                &format!("{}k", maxrate),
                "-bufsize",
                &format!("{}k", maxrate.saturating_mul(2)),
            ]));
        }

        args
    }

    #[allow(clippy::too_many_arguments)]
    fn gpu_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let decode_args = self.build_gpu_decode_args(info, params.gpu_index);
        let using_cuda = !decode_args.is_empty();
        let video_args = self.build_gpu_video_args(params, using_cuda);
        let res = run_ffmpeg_watched(
            &decode_args,
            input_path,
            &video_args,
            info,
            audio_plan,
            output_path,
            max_bytes,
            hw_slots,
            "GPU编码失败",
        );
        match res {
            Ok(outcome) => Ok(outcome),
            Err(e) if using_cuda && !is_nvenc_session_limit_error(&e.to_string()) => {
                if !params.fast_preview {
                    log_info("  (NVDEC硬解不兼容此流，自动切回软解+NVENC硬编)");
                }
                let sw_video_args = self.build_gpu_video_args(params, false);
                run_ffmpeg_watched(
                    &[],
                    input_path,
                    &sw_video_args,
                    info,
                    audio_plan,
                    output_path,
                    max_bytes,
                    hw_slots,
                    "GPU编码失败(软解回退)",
                )
            }
            Err(e) => Err(e),
        }
    }

    // ------------------------------------------------------------------------
    // Official Jetson GStreamer Backend (nvv4l2h265enc 10-bit + NVMM zero-copy)
    // ------------------------------------------------------------------------

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
            "1"
        } else {
            "0"
        };

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
            "control-rate=0".to_string(),
            format!("preset-level={}", preset_level),
            format!("bitrate={}", avg_bps),
            format!("peak-bitrate={}", peak_bps),
            format!("qp-range={}", qp_range),
        ]
    }

    fn nvmm_caps(&self, params: &EncodeParams) -> &'static str {
        if params.ten_bit && self.caps.jetson_gst_10bit {
            "video/x-raw(memory:NVMM),format=P010_10LE"
        } else {
            "video/x-raw(memory:NVMM),format=NV12"
        }
    }

    fn build_gst_audio_branch(
        &self,
        info: &VideoInfo,
        audio_plan: AudioPlan,
    ) -> Option<Vec<String>> {
        if !self.caps.jetson_gst_audio_mux || !info.has_audio {
            return None;
        }
        match audio_plan {
            AudioPlan::None => None,
            AudioPlan::Copy => {
                let parser = match info.audio_codec.as_deref()?.to_ascii_lowercase().as_str() {
                    "aac" => "aacparse",
                    "mp3" => "mpegaudioparse",
                    "ac3" | "eac3" => "ac3parse",
                    _ => return None,
                };
                Some(vec_str(&[parser, "!", "mp4mux"]))
            }
            AudioPlan::TranscodeAac(kbps) => Some(vec_str(&[
                "decodebin",
                "!",
                "audioconvert",
                "!",
                "audioresample",
                "!",
                "avenc_aac",
                &format!("bitrate={}", kbps.saturating_mul(1000)),
                "!",
                "aacparse",
                "!",
                "mp4mux",
            ])),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn jetson_run_zerocopy(
        &self,
        input_path: &Path,
        demux: &str,
        parser: &str,
        out_mkv: &Path,
        audio_branch: Option<(&[String], &Path)>,
        params: &EncodeParams,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let mut cmd = Command::new("gst-launch-1.0");
        cmd.args(["-q", "-e", "filesrc"])
            .arg(format!("location={}", input_path.to_string_lossy()))
            .args(["!", demux]);

        if audio_branch.is_some() {
            cmd.args([
                "name=d",
                "d.video_0",
                "!",
                "queue",
                "max-size-buffers=0",
                "max-size-time=0",
                "max-size-bytes=0",
            ]);
        }

        cmd.args([
            "!",
            parser,
            "!",
            "nvv4l2decoder",
            "!",
            "nvvidconv",
            "!",
            self.nvmm_caps(params),
            "!",
        ])
        .args(self.build_nvv4l2_enc_args(params))
        .args(["!", "h265parse", "!", "matroskamux", "!", "filesink"])
        .arg(format!("location={}", out_mkv.to_string_lossy()));

        if let Some((abranch, aud_path)) = audio_branch {
            cmd.args([
                "d.audio_0",
                "!",
                "queue",
                "max-size-buffers=0",
                "max-size-time=0",
                "max-size-bytes=0",
                "!",
            ])
            .args(abranch)
            .args(["!", "filesink"])
            .arg(format!("location={}", aud_path.to_string_lossy()));
        }

        let watchdog = make_watchdog(out_mkv, max_bytes);
        let outcome = {
            let _permit = hw_slots.map(HwSlotPool::acquire);
            run_command_watched(&mut cmd, watchdog, "Jetson GStreamer零拷贝硬编失败")?
        };
        finish_watched_encode(outcome, out_mkv, true, "Jetson GStreamer零拷贝硬编失败")
    }

    #[allow(clippy::too_many_arguments)]
    fn jetson_run_pipe(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        out_mkv: &Path,
        audio_plan: AudioPlan,
        out_aud_m4a: Option<&Path>,
        params: &EncodeParams,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let even_w = (info.width + (info.width % 2)).max(Self::JETSON_MIN_DIMENSION);
        let even_h = (info.height + (info.height % 2)).max(Self::JETSON_MIN_DIMENSION);
        let fps_str = format!(
            "{}/1000",
            (info.fps.clamp(1.0, 240.0) * 1000.0).round() as u32
        );

        let mut producer = Command::new("ffmpeg");
        producer
            .args(["-y", "-hide_banner", "-v", "error", "-noautorotate", "-i"])
            .arg(input_path)
            .args(["-map", "0:v:0", "-an", "-fps_mode", "cfr", "-r", &fps_str]);
        if !info.width.is_multiple_of(2) || !info.height.is_multiple_of(2) {
            producer.args(["-vf", &format!("pad={}:{}:0:0", even_w, even_h)]);
        }
        producer.args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]);

        // Second output in the same FFmpeg process: extract/transcode audio in one pass.
        if let Some(aud_path) = out_aud_m4a
            && info.has_audio
            && !matches!(audio_plan, AudioPlan::None)
        {
            producer.args(["-map", "0:a:0?", "-vn"]);
            audio_plan.append_ffmpeg_args(&mut producer);
            producer.args(["-f", "mp4"]).arg(aud_path);
        }

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
                self.nvmm_caps(params),
                "!",
            ])
            .args(self.build_nvv4l2_enc_args(params))
            .args(["!", "h265parse", "!", "matroskamux", "!", "filesink"])
            .arg(format!("location={}", out_mkv.to_string_lossy()));

        let watchdog = make_watchdog(out_mkv, max_bytes);
        let outcome = {
            let _permit = hw_slots.map(HwSlotPool::acquire);
            run_pipeline_watched(
                &mut producer,
                &mut consumer,
                watchdog,
                "Jetson软解管道+硬件编码失败",
            )?
        };
        finish_watched_encode(outcome, out_mkv, true, "Jetson软解管道+硬件编码失败")
    }

    #[allow(clippy::too_many_arguments)]
    fn jetson_encode_video_track(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        out_mkv: &Path,
        audio_plan: AudioPlan,
        out_aud_m4a: Option<&Path>,
        params: &EncodeParams,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let path_safe_for_gst = input_path
            .to_str()
            .is_some_and(|s| !s.contains('!') && !s.contains('"') && !s.contains('\''));

        if path_safe_for_gst
            && self.caps.jetson_gst_decoder
            && let Some((demux, parser)) = info.gst_zerocopy_elements()
        {
            if let Some(aud_path) = out_aud_m4a
                && let Some(abranch) = self.build_gst_audio_branch(info, audio_plan)
            {
                match self.jetson_run_zerocopy(
                    input_path,
                    demux,
                    parser,
                    out_mkv,
                    Some((abranch.as_slice(), aud_path)),
                    params,
                    max_bytes,
                    hw_slots,
                ) {
                    Ok(outcome) => return Ok(outcome),
                    Err(e) => {
                        log_debug(&format!(
                            "Jetson单遍音视频零拷贝失败，回退纯视频零拷贝: {}",
                            e
                        ));
                        let _ = std::fs::remove_file(out_mkv);
                        let _ = std::fs::remove_file(aud_path);
                    }
                }
            }

            match self.jetson_run_zerocopy(
                input_path, demux, parser, out_mkv, None, params, max_bytes, hw_slots,
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

        self.jetson_run_pipe(
            input_path,
            info,
            out_mkv,
            audio_plan,
            out_aud_m4a,
            params,
            max_bytes,
            hw_slots,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn jetson_full_encode(
        &self,
        input_path: &Path,
        info: &VideoInfo,
        output_path: &Path,
        params: &EncodeParams,
        audio_plan: AudioPlan,
        max_bytes: Option<u64>,
        hw_slots: Option<&HwSlotPool>,
    ) -> Result<EncodeOutcome> {
        let tmp_mkv = NamedTempFile::with_suffix(".mkv").context("无法创建临时视频轨文件")?;
        let tmp_aud = if info.has_audio && !matches!(audio_plan, AudioPlan::None) {
            Some(NamedTempFile::with_suffix(".m4a").context("无法创建临时音频轨文件")?)
        } else {
            None
        };

        match self.jetson_encode_video_track(
            input_path,
            info,
            tmp_mkv.path(),
            audio_plan,
            tmp_aud.as_ref().map(|f| f.path()),
            params,
            max_bytes,
            hw_slots,
        )? {
            EncodeOutcome::AbortedSizeLimit { written_bytes } => {
                Ok(EncodeOutcome::AbortedSizeLimit { written_bytes })
            }
            EncodeOutcome::Completed(_) => {
                let final_size = jetson_remux_to_mp4(
                    tmp_mkv.path(),
                    tmp_aud.as_ref().map(|f| f.path()),
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

fn jetson_remux_to_mp4(
    video_mkv: &Path,
    pre_extracted_audio: Option<&Path>,
    original_input: &Path,
    info: &VideoInfo,
    audio_plan: AudioPlan,
    output_mp4: &Path,
) -> Result<u64> {
    let valid_local_audio = pre_extracted_audio.filter(|p| {
        !matches!(audio_plan, AudioPlan::None)
            && std::fs::metadata(p).is_ok_and(|m| m.len() > 0)
            && probe_has_audio(p)
    });

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-hide_banner", "-v", "error"]);
    if info.rotation != 0 {
        cmd.args(["-display_rotation:v:0", &info.rotation.to_string()]);
    }

    cmd.arg("-i").arg(video_mkv);
    let metadata_src_idx = if let Some(aud_path) = valid_local_audio {
        log_debug("  [合流优化] 命中单遍分流本地音频轨，纯 -c copy 极速合流 (免二次通读源文件)");
        cmd.arg("-i")
            .arg(aud_path)
            .arg("-i")
            .arg(original_input)
            .args(["-map", "0:v:0", "-map", "1:a:0", "-c", "copy"]);
        "2"
    } else {
        cmd.arg("-i").arg(original_input).args(["-map", "0:v:0"]);
        if !matches!(audio_plan, AudioPlan::None) {
            cmd.args(["-map", "1:a:0?"]);
        }
        cmd.args(["-c:v", "copy"]);
        audio_plan.append_ffmpeg_args(&mut cmd);
        "1"
    };

    cmd.args(["-tag:v", "hvc1"]);
    append_color_tags(&mut cmd, info);
    cmd.args(["-map_metadata", metadata_src_idx, "-movflags", "+faststart"])
        .arg(output_mp4);

    let out = run_command_cancellable(&mut cmd, "FFmpeg合流封装MP4失败")?;
    if !out.status.success() {
        anyhow::bail!("FFmpeg合流封装MP4失败: {}", stderr_tail(&out));
    }
    Ok(std::fs::metadata(output_mp4)
        .context("无法读取最终封装MP4")?
        .len())
}

// ============================================================================
// Shared FFmpeg command helpers
// ============================================================================

fn vec_str(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

fn make_watchdog(watch_path: &Path, max_bytes: Option<u64>) -> Option<SizeWatchdog<'_>> {
    max_bytes.map(|limit| SizeWatchdog {
        watch_path,
        max_bytes: limit,
    })
}

fn finish_watched_encode(
    outcome: WatchedOutcome,
    output_path: &Path,
    require_non_empty: bool,
    context_msg: &str,
) -> Result<EncodeOutcome> {
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
            if require_non_empty && size == 0 {
                anyhow::bail!("{}: 输出文件为空", context_msg);
            }
            Ok(EncodeOutcome::Completed(size))
        }
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

fn stderr_tail(output: &std::process::Output) -> String {
    let text = String::from_utf8_lossy(&output.stderr);
    let mut lines: Vec<&str> = text.lines().rev().take(15).collect();
    lines.reverse();
    lines.join("\n")
}

fn append_color_tags(cmd: &mut Command, info: &VideoInfo) {
    if let Some(cp) = &info.color_primaries {
        cmd.args(["-color_primaries", cp]);
    }
    if let Some(ct) = &info.color_transfer {
        cmd.args(["-color_trc", ct]);
    }
    if let Some(cs) = &info.color_space {
        cmd.args(["-colorspace", cs]);
    }
}

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
    cmd.args(["-y", "-hide_banner", "-v", "error"])
        .args(pre_input_args)
        .arg("-i")
        .arg(input_path)
        .args(["-map", "0:v:0"]);
    if !matches!(audio_plan, AudioPlan::None) {
        cmd.args(["-map", "0:a:0?"]);
    }
    cmd.args(video_args);
    append_color_tags(&mut cmd, info);
    audio_plan.append_ffmpeg_args(&mut cmd);
    cmd.args([
        "-map_metadata",
        "0",
        "-tag:v",
        "hvc1",
        "-movflags",
        "+faststart",
    ])
    .arg(output_path);

    let watchdog = make_watchdog(output_path, max_bytes);
    let outcome = {
        let _permit = hw_slots.map(HwSlotPool::acquire);
        run_command_watched(&mut cmd, watchdog, context_msg)?
    };
    finish_watched_encode(outcome, output_path, false, context_msg)
}

// ============================================================================
// Single-Command Pure-Video Reference Sample Extraction
// ============================================================================

/// Extracted near-lossless pure-video reference sample for perceptual search.
pub struct ReferenceSample {
    pub file: NamedTempFile,
    pub duration: f64,
    pub info: VideoInfo,
}

/// Extract a representative pure-video (`-an`) reference sample in ONE FFmpeg command.
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
        let starts = [
            (dur * 0.20 - 1.0).max(0.0),
            (dur * 0.50 - 1.0).max(0.0),
            (dur * 0.80 - 1.0).max(0.0),
        ];
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-v", "error"]);
        for s in starts {
            cmd.args([
                "-noautorotate",
                "-ss",
                &format!("{:.2}", s),
                "-t",
                "2",
                "-i",
            ])
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

/// Detect NVIDIA consumer GPU NVENC concurrent session limit errors so the pool
/// can dynamically cap concurrency and retry cleanly.
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
            ..Default::default()
        };
        let gpu = Encoder::new(EncoderType::Gpu, &caps);
        let params = EncodeParams::for_quality(26, "slow", true, false);
        let args = gpu.build_gpu_video_args(&params, true);

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
        let gpu = Encoder::new(EncoderType::Gpu, &caps);
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
        assert!(
            gpu.build_gpu_decode_args(&h264_info, Some(0))
                .windows(2)
                .any(|w| w == ["-hwaccel_output_format", "cuda"])
        );
        assert!(gpu.build_gpu_decode_args(&wmv_info, Some(0)).is_empty());

        let params = EncodeParams::for_quality(26, "medium", true, false);
        let sw_args = gpu.build_gpu_video_args(&params, false);
        assert!(sw_args.windows(2).any(|w| w == ["-pix_fmt", "p010le"]));
    }

    #[test]
    fn cpu_encoder_drops_invalid_fast_param_and_uses_10bit() {
        let caps = HardwareCapabilities {
            has_libx265: true,
            x265_10bit: true,
            ..Default::default()
        };
        let cpu = Encoder::new(EncoderType::Cpu, &caps);
        let params = EncodeParams::for_quality(24, "medium", true, false);
        let (_, args) = cpu.build_standard_ffmpeg_args(&params);
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
        let enc = Encoder::new(EncoderType::JetsonGst, &caps);
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
    fn jetson_gst_audio_branch_handles_copy_and_transcode() {
        let caps = HardwareCapabilities {
            jetson_gst_ok: true,
            jetson_gst_audio_mux: true,
            ..Default::default()
        };
        let enc = Encoder::new(EncoderType::JetsonGst, &caps);
        let aac_info = VideoInfo {
            has_audio: true,
            audio_codec: Some("aac".into()),
            ..Default::default()
        };
        assert_eq!(
            enc.build_gst_audio_branch(&aac_info, AudioPlan::Copy)
                .unwrap(),
            vec!["aacparse", "!", "mp4mux"]
        );

        let branch_trans = enc
            .build_gst_audio_branch(&aac_info, AudioPlan::TranscodeAac(128))
            .unwrap();
        assert!(branch_trans.contains(&"avenc_aac".to_string()));
        assert!(branch_trans.contains(&"bitrate=128000".to_string()));
        assert!(branch_trans.contains(&"mp4mux".to_string()));

        let opus_info = VideoInfo {
            has_audio: true,
            audio_codec: Some("opus".into()),
            ..Default::default()
        };
        assert!(
            enc.build_gst_audio_branch(&opus_info, AudioPlan::Copy)
                .is_none()
        );
    }

    #[test]
    fn hw_slot_pool_shrinks_safely_without_double_decrement() {
        let pool = HwSlotPool::new(4);
        let _p1 = pool.acquire();
        let _p2 = pool.acquire();
        // Simulate 2 other workers failing simultaneously in the wave at limit=4 (in_use now 2).
        assert_eq!(pool.shrink_below(4), 2);
        assert_eq!(pool.shrink_below(4), 2); // second worker in same wave must not over-shrink
        assert_eq!(pool.limit(), 2);

        // If a later wave at limit=2 fails while 1 slot is in use, shrinks cleanly to 1.
        drop(_p2);
        assert_eq!(pool.shrink_below(2), 1);
        assert_eq!(pool.shrink_below(1), 1);
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
