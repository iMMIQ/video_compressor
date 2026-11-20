use crate::config::{EncoderType, HardwareCapabilities};
use crate::logging::{log_debug, log_info};
use crate::signal::run_command_cancellable;
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

/// Parsed metadata from a single `ffprobe` call covering video stream,
/// primary audio stream, and container format.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoInfo {
    pub codec_name: Option<String>,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    /// Container-level total bitrate (bps), 0 if unavailable.
    pub bitrate: u64,
    /// Video stream bitrate (bps), 0 if unavailable.
    pub video_bitrate: u64,
    pub fps: f64,
    pub pix_fmt: Option<String>,
    pub color_transfer: Option<String>,
    pub color_primaries: Option<String>,
    pub color_space: Option<String>,
    pub format_name: Option<String>,
    pub rotation: i32,
    // Primary audio stream metadata
    pub has_audio: bool,
    pub audio_codec: Option<String>,
    pub audio_bitrate: u64,
    pub audio_channels: u32,
}

impl Default for VideoInfo {
    fn default() -> Self {
        Self {
            codec_name: None,
            width: 1920,
            height: 1080,
            duration: 0.0,
            bitrate: 0,
            video_bitrate: 0,
            fps: 30.0,
            pix_fmt: None,
            color_transfer: None,
            color_primaries: None,
            color_space: None,
            format_name: None,
            rotation: 0,
            has_audio: false,
            audio_codec: None,
            audio_bitrate: 0,
            audio_channels: 0,
        }
    }
}

impl VideoInfo {
    /// Already encoded in a modern high-efficiency codec (HEVC, AV1, VP9).
    pub fn is_modern_codec(&self) -> Option<&'static str> {
        let codec = self.codec_name.as_deref()?.to_ascii_lowercase();
        if codec.contains("hevc") || codec.contains("h265") {
            Some("已是H.265编码")
        } else if codec == "av1" || codec.contains("av01") {
            Some("已是AV1编码")
        } else if codec == "vp9" {
            Some("已是VP9编码")
        } else {
            None
        }
    }

    /// True if the source video uses HDR transfer characteristics (PQ / HLG).
    pub fn is_hdr(&self) -> bool {
        self.color_transfer.as_deref().is_some_and(|t| {
            t.eq_ignore_ascii_case("smpte2084") || t.eq_ignore_ascii_case("arib-std-b67")
        })
    }

    /// True if the source pixel format is 10-bit or deeper.
    pub fn is_high_bit_depth(&self) -> bool {
        self.pix_fmt.as_deref().is_some_and(|p| {
            let lower = p.to_ascii_lowercase();
            lower.contains("10") || lower.contains("12") || lower.contains("16")
        })
    }

    /// True if pixel format is unknown or standard 4:2:0 (supported by HW NVDEC).
    pub fn is_420_pix_fmt(&self) -> bool {
        self.pix_fmt.as_deref().is_none_or(|fmt| {
            matches!(
                fmt.to_ascii_lowercase().as_str(),
                "yuv420p" | "yuvj420p" | "nv12" | "yuv420p10le" | "p010le"
            )
        })
    }

    /// Effective video stream bitrate in bps, subtracting audio when only container bitrate is known.
    pub fn effective_video_bitrate(&self, input_size: u64) -> u64 {
        if self.video_bitrate > 0 {
            return self.video_bitrate;
        }
        let total_bps = if self.bitrate > 0 {
            self.bitrate
        } else if self.duration > 0.0 {
            ((input_size as f64 * 8.0) / self.duration) as u64
        } else {
            0
        };
        total_bps.saturating_sub(self.audio_bitrate)
    }

    /// Bits per pixel per frame (BPP) — identifies already heavily compressed sources.
    pub fn bits_per_pixel(&self, input_size: u64) -> f64 {
        let v_bps = self.effective_video_bitrate(input_size) as f64;
        let pixels_per_sec = self.width as f64 * self.height as f64 * self.fps.max(1.0);
        if pixels_per_sec <= 0.0 {
            0.0
        } else {
            v_bps / pixels_per_sec
        }
    }

    /// Whether source codec + pixel format is safely supported by desktop CUDA NVDEC zero-copy decode.
    pub fn is_nvdec_compatible(&self) -> bool {
        let codec_ok = self.codec_name.as_deref().is_some_and(|codec| {
            matches!(
                codec.to_ascii_lowercase().as_str(),
                "h264" | "hevc" | "vp9" | "vp8" | "mpeg2video" | "mpeg4" | "vc1" | "av1" | "mjpeg"
            )
        });
        codec_ok && self.is_420_pix_fmt()
    }

    /// Determine GStreamer zero-copy (demuxer, parser) pair if the container + codec + pixel format
    /// can go straight through `nvv4l2decoder -> nvvidconv -> nvv4l2h265enc` in NVMM memory.
    pub fn gst_zerocopy_elements(&self) -> Option<(&'static str, &'static str)> {
        if !self.is_420_pix_fmt() {
            return None;
        }
        let fmt = self.format_name.as_deref()?.to_ascii_lowercase();
        let demux = if fmt.contains("mp4") || fmt.contains("mov") || fmt.contains("3gp") {
            "qtdemux"
        } else if fmt.contains("matroska") || fmt.contains("webm") {
            "matroskademux"
        } else if fmt.contains("mpegts") {
            "tsdemux"
        } else {
            return None;
        };

        let codec = self.codec_name.as_deref()?.to_ascii_lowercase();
        let parser = match codec.as_str() {
            "h264" => "h264parse",
            "hevc" | "h265" => "h265parse",
            "mpeg4" => "mpeg4videoparse",
            "mpeg2video" => "mpegvideoparse",
            _ => return None,
        };

        Some((demux, parser))
    }
}

/// Smart audio handling strategy to avoid generational loss, 5.1 channel starvation,
/// and low-bitrate audio bloat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPlan {
    None,
    Copy,
    TranscodeAac(u32),
}

impl AudioPlan {
    /// Decide whether to copy or transcode audio based on source stream properties and CLI bitrate.
    pub fn decide(info: &VideoInfo, requested_bitrate_kbps: u32) -> Self {
        if !info.has_audio {
            return Self::None;
        }
        if requested_bitrate_kbps == 0 {
            return Self::Copy;
        }

        // Adapt target bitrate to channel count so 5.1 surround isn't destroyed at 128kbps.
        let target_kbps = match info.audio_channels {
            0 | 2 => requested_bitrate_kbps,
            1 => requested_bitrate_kbps.min(96),
            3..=4 => requested_bitrate_kbps.max(192),
            _ => requested_bitrate_kbps.max(256),
        };

        let mp4_compatible = info.audio_codec.as_deref().is_some_and(|c| {
            matches!(
                c.to_ascii_lowercase().as_str(),
                "aac" | "mp3" | "ac3" | "eac3" | "opus"
            )
        });

        // If already MP4-compatible and within 15% of (or below!) target bitrate, copy directly.
        if mp4_compatible && info.audio_bitrate > 0 {
            let threshold_bps = (target_kbps as u64) * 1000 * 115 / 100;
            if info.audio_bitrate <= threshold_bps {
                return Self::Copy;
            }
        }

        Self::TranscodeAac(target_kbps)
    }

    /// Estimated final audio bitrate in bits/sec for accurate video-budget accounting.
    pub fn estimated_bps(self, info: &VideoInfo) -> u64 {
        match self {
            Self::None => 0,
            Self::Copy => {
                if info.audio_bitrate > 0 {
                    info.audio_bitrate
                } else {
                    128_000
                }
            }
            Self::TranscodeAac(kbps) => (kbps as u64) * 1000,
        }
    }

    /// Append audio arguments to an FFmpeg command.
    pub fn append_ffmpeg_args(self, cmd: &mut Command) {
        match self {
            Self::None => {
                cmd.arg("-an");
            }
            Self::Copy => {
                cmd.args(["-c:a", "copy"]);
            }
            Self::TranscodeAac(kbps) => {
                cmd.args(["-c:a", "aac", "-b:a", &format!("{}k", kbps)]);
            }
        }
    }
}

/// Perceptual quality metric filter supported by standard FFmpeg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMetric {
    Xpsnr,
    Ssim,
}

impl QualityMetric {
    pub fn from_capabilities(caps: &HardwareCapabilities) -> Option<Self> {
        if caps.has_xpsnr {
            Some(Self::Xpsnr)
        } else if caps.has_ssim {
            Some(Self::Ssim)
        } else {
            None
        }
    }

    pub fn target(self, target_xpsnr: f64) -> f64 {
        match self {
            Self::Xpsnr => target_xpsnr,
            Self::Ssim => 0.980,
        }
    }

    /// Minimum score gain for a 4-point Q reduction before declaring a noise-floor knee.
    pub fn knee_threshold(self) -> f64 {
        match self {
            Self::Xpsnr => 1.0,
            Self::Ssim => 0.004,
        }
    }

    pub fn format_score(self, score: f64) -> String {
        match self {
            Self::Xpsnr => format!("XPSNR={:.2}dB", score),
            Self::Ssim => format!("SSIM={:.4}", score),
        }
    }
}

/// Summary of FFmpeg compiled-in encoders detected in a single probe.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct AvailableEncoders {
    libx265: bool,
    libx264: bool,
    nvenc: bool,
    qsv: bool,
    vaapi: bool,
    amf: bool,
    videotoolbox: bool,
}

fn parse_ffmpeg_encoders(encoders_stdout: &str) -> AvailableEncoders {
    let mut avail = AvailableEncoders::default();
    for line in encoders_stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('V') {
            continue;
        }
        avail.libx265 |= trimmed.contains("libx265");
        avail.libx264 |= trimmed.contains("libx264");
        avail.nvenc |= trimmed.contains("hevc_nvenc");
        avail.qsv |= trimmed.contains("hevc_qsv");
        avail.vaapi |= trimmed.contains("hevc_vaapi");
        avail.amf |= trimmed.contains("hevc_amf");
        avail.videotoolbox |= trimmed.contains("hevc_videotoolbox");
    }
    avail
}

fn query_ffmpeg_encoders() -> Result<AvailableEncoders> {
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .context("无法执行ffmpeg命令，请确保已安装FFmpeg")?;

    if !output.status.success() {
        return Ok(AvailableEncoders::default());
    }
    Ok(parse_ffmpeg_encoders(&String::from_utf8_lossy(
        &output.stdout,
    )))
}

fn query_ffmpeg_filters() -> (bool, bool) {
    let Ok(output) = Command::new("ffmpeg")
        .args(["-hide_banner", "-filters"])
        .output()
    else {
        return (false, false);
    };
    if !output.status.success() {
        return (false, false);
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let (mut has_xpsnr, mut has_ssim) = (false, false);
    for line in text.lines() {
        match line.split_whitespace().nth(1) {
            Some("xpsnr") => has_xpsnr = true,
            Some("ssim") => has_ssim = true,
            _ => {}
        }
    }
    (has_xpsnr, has_ssim)
}

fn probe_x265_10bit() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-h", "encoder=libx265"])
        .output()
        .is_ok_and(|o| {
            o.status.success() && String::from_utf8_lossy(&o.stdout).contains("yuv420p10le")
        })
}

/// Run a tiny 2-frame lavfi encode to verify an FFmpeg hardware encoder initializes at runtime.
fn probe_ffmpeg_hw_encode(pre_args: &[&str], post_args: &[&str]) -> bool {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-v", "error"])
        .args(pre_args)
        .args(["-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.07:r=30"])
        .args(post_args)
        .args(["-f", "null", "-"]);
    cmd.output().is_ok_and(|o| o.status.success())
}

fn probe_gst_nvv4l2(ten_bit: bool) -> bool {
    let fmt = if ten_bit { "P010_10LE" } else { "NV12" };
    let profile = if ten_bit { "profile=1" } else { "profile=0" };
    let caps = format!("video/x-raw(memory:NVMM),format={}", fmt);
    Command::new("gst-launch-1.0")
        .args([
            "-q",
            "-e",
            "videotestsrc",
            "num-buffers=2",
            "!",
            "video/x-raw,width=256,height=256,framerate=30/1,format=NV12",
            "!",
            "nvvidconv",
            "!",
            &caps,
            "!",
            "nvv4l2h265enc",
            profile,
            "control-rate=0",
            "preset-level=2",
            "bitrate=2000000",
            "qp-range=27,29:25,27:27,29",
            "!",
            "h265parse",
            "!",
            "fakesink",
        ])
        .output()
        .is_ok_and(|o| o.status.success())
}

fn probe_gst_element_exists(element: &str) -> bool {
    Command::new("gst-inspect-1.0")
        .arg(element)
        .output()
        .is_ok_and(|o| o.status.success())
}

fn detect_nvidia_gpu_count() -> usize {
    Command::new("nvidia-smi")
        .args(["--query-gpu=index", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count()
        })
        .filter(|&c| c > 0)
        .unwrap_or(1)
}

fn find_vaapi_device() -> Option<String> {
    (128..=135)
        .map(|idx| format!("/dev/dri/renderD{}", idx))
        .find(|p| Path::new(p).exists())
}

fn is_jetson_platform() -> bool {
    std::fs::read_to_string("/proc/sys/kernel/osrelease").is_ok_and(|r| r.contains("tegra"))
        || Path::new("/dev/v4l2-nvenc").exists()
        || Path::new("/sys/devices/platform/17000000.gpu").exists()
}

fn probe_jetson_gst_caps(caps: &mut HardwareCapabilities) {
    if probe_gst_nvv4l2(true) {
        caps.jetson_gst_ok = true;
        caps.jetson_gst_10bit = true;
    } else if probe_gst_nvv4l2(false) {
        caps.jetson_gst_ok = true;
    }
    if caps.jetson_gst_ok {
        caps.jetson_gst_decoder =
            probe_gst_element_exists("nvv4l2decoder") && probe_gst_element_exists("matroskamux");
        caps.jetson_gst_audio_mux = probe_gst_element_exists("mp4mux")
            && probe_gst_element_exists("aacparse")
            && probe_gst_element_exists("avenc_aac");
    }
}

fn probe_nvenc_caps(caps: &mut HardwareCapabilities) {
    let probe_nvenc = |extra: &[&str]| {
        let mut args = vec!["-c:v", "hevc_nvenc", "-rc", "vbr", "-cq", "28", "-b:v", "0"];
        args.extend_from_slice(extra);
        probe_ffmpeg_hw_encode(&[], &args)
    };

    // Probe full modern NVENC feature set first (Turing/Ampere/Ada/Blackwell).
    if probe_nvenc(&[
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-profile:v",
        "main10",
        "-pix_fmt",
        "p010le",
        "-bf",
        "4",
        "-b_ref_mode",
        "middle",
        "-multipass",
        "qres",
    ]) {
        caps.nvenc_ok = true;
        caps.nvenc_10bit = true;
        caps.nvenc_bframes = true;
        caps.nvenc_multipass = true;
        caps.nvenc_uhq = probe_nvenc(&["-preset", "p6", "-tune", "uhq"]);
        caps.nvenc_gpu_count = detect_nvidia_gpu_count();
    } else if probe_nvenc(&["-preset", "p4"]) {
        // Older GPU (e.g. Pascal) — verify sub-features individually.
        caps.nvenc_ok = true;
        caps.nvenc_10bit = probe_nvenc(&["-profile:v", "main10", "-pix_fmt", "p010le"]);
        caps.nvenc_bframes = probe_nvenc(&["-bf", "4", "-b_ref_mode", "middle"]);
        caps.nvenc_multipass = probe_nvenc(&["-multipass", "qres"]);
        caps.nvenc_gpu_count = detect_nvidia_gpu_count();
    }
}

fn probe_qsv_caps(caps: &mut HardwareCapabilities) {
    caps.qsv_ok = probe_ffmpeg_hw_encode(
        &[],
        &[
            "-c:v",
            "hevc_qsv",
            "-global_quality",
            "26",
            "-look_ahead",
            "0",
        ],
    );
}

fn probe_videotoolbox_caps(caps: &mut HardwareCapabilities) {
    caps.videotoolbox_ok =
        probe_ffmpeg_hw_encode(&[], &["-c:v", "hevc_videotoolbox", "-q:v", "65"]);
}

fn probe_amf_caps(caps: &mut HardwareCapabilities) {
    caps.amf_ok = probe_ffmpeg_hw_encode(
        &[],
        &[
            "-c:v", "hevc_amf", "-rc", "cqp", "-qp_i", "26", "-qp_p", "26",
        ],
    );
}

fn probe_vaapi_caps(caps: &mut HardwareCapabilities) {
    if let Some(dev) = find_vaapi_device()
        && probe_ffmpeg_hw_encode(
            &["-vaapi_device", &dev],
            &[
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "hevc_vaapi",
                "-rc_mode",
                "CQP",
                "-qp",
                "26",
            ],
        )
    {
        caps.vaapi_ok = true;
        caps.vaapi_device = Some(dev);
    }
}

/// Parse CLI `--encoder` string (`None` represents `"auto"`).
pub fn parse_encoder_arg(s: &str) -> Result<Option<EncoderType>> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(None),
        "cpu" => Ok(Some(EncoderType::Cpu)),
        "gpu" | "nvenc" => Ok(Some(EncoderType::Gpu)),
        "jetson" | "jetson-gst" | "gstreamer" | "v4l2" => Ok(Some(EncoderType::JetsonGst)),
        "qsv" => Ok(Some(EncoderType::Qsv)),
        "vaapi" => Ok(Some(EncoderType::Vaapi)),
        "amf" => Ok(Some(EncoderType::Amf)),
        "videotoolbox" | "vt" => Ok(Some(EncoderType::VideoToolbox)),
        _ => anyhow::bail!(
            "无效的编码器类型: {}。可选: auto, cpu, gpu, jetson, jetson-gst, qsv, vaapi, amf, videotoolbox",
            s
        ),
    }
}

/// Perform real runtime hardware capability probing once at startup.
/// When `target` is explicit (`Some(...)`), probes only the requested backend
/// (avoiding unnecessary probes on `--encoder cpu` and ensuring `--encoder vaapi/qsv`
/// works on multi-GPU hosts). When `None` (`auto`), probes in priority order.
pub fn detect_hardware_capabilities(target: Option<EncoderType>) -> Result<HardwareCapabilities> {
    let enc = query_ffmpeg_encoders()?;
    let (has_xpsnr, has_ssim) = query_ffmpeg_filters();
    let x265_10bit = enc.libx265 && probe_x265_10bit();

    let mut caps = HardwareCapabilities {
        has_libx265: enc.libx265,
        x265_10bit,
        has_libx264: enc.libx264,
        has_xpsnr,
        has_ssim,
        ..Default::default()
    };

    match target {
        Some(EncoderType::Cpu) => {}
        Some(EncoderType::JetsonGst) => probe_jetson_gst_caps(&mut caps),
        Some(EncoderType::Gpu) if enc.nvenc => probe_nvenc_caps(&mut caps),
        Some(EncoderType::Qsv) if enc.qsv => probe_qsv_caps(&mut caps),
        Some(EncoderType::Vaapi) if enc.vaapi => probe_vaapi_caps(&mut caps),
        Some(EncoderType::Amf) if enc.amf => probe_amf_caps(&mut caps),
        Some(EncoderType::VideoToolbox) if enc.videotoolbox => probe_videotoolbox_caps(&mut caps),
        Some(_) => {}
        None if is_jetson_platform() => probe_jetson_gst_caps(&mut caps),
        None => {
            if enc.nvenc {
                probe_nvenc_caps(&mut caps);
            }
            if !caps.nvenc_ok && enc.qsv {
                probe_qsv_caps(&mut caps);
            }
            if !caps.nvenc_ok && !caps.qsv_ok && enc.videotoolbox {
                probe_videotoolbox_caps(&mut caps);
            }
            if !caps.nvenc_ok && !caps.qsv_ok && !caps.videotoolbox_ok && enc.amf {
                probe_amf_caps(&mut caps);
            }
            if !caps.nvenc_ok && !caps.qsv_ok && !caps.videotoolbox_ok && !caps.amf_ok && enc.vaapi
            {
                probe_vaapi_caps(&mut caps);
            }
        }
    }

    log_debug(&format!("硬件探测结果: {:?}", caps));
    Ok(caps)
}

/// Resolve the active `EncoderType` from requested CLI target and runtime capabilities.
pub fn resolve_encoder_type(
    target: Option<EncoderType>,
    caps: &HardwareCapabilities,
) -> EncoderType {
    if let Some(explicit) = target {
        return explicit;
    }
    let chosen = if caps.jetson_gst_ok {
        EncoderType::JetsonGst
    } else if caps.nvenc_ok {
        EncoderType::Gpu
    } else if caps.qsv_ok {
        EncoderType::Qsv
    } else if caps.videotoolbox_ok {
        EncoderType::VideoToolbox
    } else if caps.amf_ok {
        EncoderType::Amf
    } else if caps.vaapi_ok {
        EncoderType::Vaapi
    } else {
        log_info("注意: 未检测到可用硬件编码器，使用 CPU (libx265)");
        return EncoderType::Cpu;
    };
    log_info(&format!("自动选择: {}", chosen.display_name()));
    chosen
}

/// Verify FFmpeg and the selected encoder backend are usable at runtime.
pub fn check_ffmpeg(encoder_type: EncoderType, caps: &HardwareCapabilities) -> Result<()> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("无法找到ffmpeg命令，请确保已安装标准FFmpeg")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg执行失败");
    }

    if let Some(line) = String::from_utf8_lossy(&output.stdout).lines().next() {
        log_info(&format!("检测到: {}", line));
    }

    let metric_str = match QualityMetric::from_capabilities(caps) {
        Some(QualityMetric::Xpsnr) => "XPSNR (感知加权)",
        Some(QualityMetric::Ssim) => "SSIM (结构相似)",
        None => "仅码率估算",
    };
    log_info(&format!("感知质量评估引擎: {}", metric_str));

    match encoder_type {
        EncoderType::Cpu => {
            if !caps.has_libx265 {
                anyhow::bail!("FFmpeg未编译libx265支持，无法进行CPU H.265编码");
            }
            log_info(&format!(
                "✓ CPU libx265 可用 (10-bit: {})",
                if caps.x265_10bit { "开启" } else { "关闭" }
            ));
        }
        EncoderType::Gpu => {
            if !caps.nvenc_ok {
                anyhow::bail!("NVENC硬件初始化失败（请确认NVIDIA驱动与GPU可用）");
            }
            log_info(&format!(
                "✓ NVENC 可用 (GPU数: {} | 10-bit: {} | B帧: {} | Multipass: {} | UHQ: {})",
                caps.nvenc_gpu_count,
                caps.nvenc_10bit,
                caps.nvenc_bframes,
                caps.nvenc_multipass,
                caps.nvenc_uhq
            ));
        }
        EncoderType::JetsonGst => {
            if !caps.jetson_gst_ok {
                anyhow::bail!(
                    "未检测到可用的 JetPack 官方 GStreamer nvv4l2h265enc（请确认已安装 nvidia-l4t-gstreamer）"
                );
            }
            log_info(&format!(
                "✓ Jetson 官方 nvv4l2h265enc 可用 (10-bit Main10: {} | NVMM零拷贝硬解: {})",
                caps.jetson_gst_10bit, caps.jetson_gst_decoder
            ));
        }
        EncoderType::Qsv if caps.qsv_ok => log_info("✓ Intel QSV (hevc_qsv) 可用"),
        EncoderType::Qsv => anyhow::bail!("Intel QSV (hevc_qsv) 初始化失败"),
        EncoderType::Vaapi if caps.vaapi_ok => log_info(&format!(
            "✓ VA-API (hevc_vaapi) 可用 (设备: {})",
            caps.vaapi_device.as_deref().unwrap_or("default")
        )),
        EncoderType::Vaapi => anyhow::bail!("Linux VA-API (hevc_vaapi) 初始化失败"),
        EncoderType::Amf if caps.amf_ok => log_info("✓ AMD AMF (hevc_amf) 可用"),
        EncoderType::Amf => anyhow::bail!("AMD AMF (hevc_amf) 初始化失败"),
        EncoderType::VideoToolbox if caps.videotoolbox_ok => {
            log_info("✓ Apple VideoToolbox (hevc_videotoolbox) 可用");
        }
        EncoderType::VideoToolbox => {
            anyhow::bail!("Apple VideoToolbox (hevc_videotoolbox) 初始化失败")
        }
    }

    Ok(())
}

/// Parse an FFmpeg rational frame rate string (e.g. `"30000/1001"` or `"25/1"`).
pub fn parse_fps_fraction(raw: &str) -> Option<f64> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("N/A") {
        return None;
    }
    let val = if let Some((num_s, den_s)) = trimmed.split_once('/') {
        let num: f64 = num_s.trim().parse().ok()?;
        let den: f64 = den_s.trim().parse().ok()?;
        if den <= 0.0 || num <= 0.0 {
            return None;
        }
        num / den
    } else {
        trimmed.parse::<f64>().ok()?
    };
    (val.is_finite() && (1.0..=240.0).contains(&val)).then_some(val)
}

#[derive(Default)]
struct StreamAcc {
    codec_type: Option<String>,
    codec: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    pix_fmt: Option<String>,
    bitrate: Option<u64>,
    avg_fps: Option<f64>,
    r_fps: Option<f64>,
    color_transfer: Option<String>,
    color_primaries: Option<String>,
    color_space: Option<String>,
    channels: Option<u32>,
    rotation: Option<i32>,
}

impl StreamAcc {
    fn flush_into(self, info: &mut VideoInfo, seen_video: &mut bool, seen_audio: &mut bool) {
        match self.codec_type.as_deref() {
            Some("video") if !*seen_video => {
                *seen_video = true;
                info.codec_name = self.codec;
                if let Some(w) = self.width {
                    info.width = w;
                }
                if let Some(h) = self.height {
                    info.height = h;
                }
                info.pix_fmt = self.pix_fmt;
                info.video_bitrate = self.bitrate.unwrap_or(0);
                info.fps = self.avg_fps.or(self.r_fps).unwrap_or(30.0);
                info.color_transfer = self.color_transfer;
                info.color_primaries = self.color_primaries;
                info.color_space = self.color_space;
                info.rotation = self.rotation.unwrap_or(0);
            }
            Some("audio") if !*seen_audio => {
                *seen_audio = true;
                info.has_audio = true;
                info.audio_codec = self.codec;
                info.audio_bitrate = self.bitrate.unwrap_or(0);
                info.audio_channels = self.channels.unwrap_or(2);
            }
            _ => {}
        }
    }
}

/// Parse `ffprobe -of default=noprint_wrappers=0` sectioned output containing
/// `[STREAM]` blocks (video + audio) and `[FORMAT]` block.
pub fn parse_ffprobe_sections(output: &str) -> VideoInfo {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Section {
        None,
        Stream,
        SideData,
        Format,
    }

    let mut info = VideoInfo::default();
    let mut section = Section::None;
    let mut acc = StreamAcc::default();
    let (mut seen_video, mut seen_audio) = (false, false);

    for line in output.lines().map(str::trim) {
        match line {
            "[STREAM]" => {
                section = Section::Stream;
                continue;
            }
            "[/STREAM]" => {
                std::mem::take(&mut acc).flush_into(&mut info, &mut seen_video, &mut seen_audio);
                section = Section::None;
                continue;
            }
            "[SIDE_DATA]" => {
                section = Section::SideData;
                continue;
            }
            "[/SIDE_DATA]" => {
                section = Section::Stream;
                continue;
            }
            "[FORMAT]" => {
                section = Section::Format;
                continue;
            }
            "[/FORMAT]" => {
                section = Section::None;
                continue;
            }
            _ => {}
        }

        let Some((key, raw_val)) = line.split_once('=') else {
            continue;
        };
        let val = raw_val.trim();
        if val.is_empty() || val.eq_ignore_ascii_case("N/A") {
            continue;
        }

        match section {
            Section::Stream => match key.trim() {
                "codec_type" => acc.codec_type = Some(val.to_ascii_lowercase()),
                "codec_name" => acc.codec = Some(val.to_string()),
                "width" => acc.width = val.parse().ok(),
                "height" => acc.height = val.parse().ok(),
                "pix_fmt" => acc.pix_fmt = Some(val.to_string()),
                "bit_rate" => acc.bitrate = val.parse().ok(),
                "avg_frame_rate" => acc.avg_fps = parse_fps_fraction(val),
                "r_frame_rate" => acc.r_fps = parse_fps_fraction(val),
                "color_transfer" => acc.color_transfer = Some(val.to_string()),
                "color_primaries" => acc.color_primaries = Some(val.to_string()),
                "color_space" => acc.color_space = Some(val.to_string()),
                "channels" => acc.channels = val.parse().ok(),
                "TAG:rotate" => acc.rotation = val.parse().ok(),
                _ => {}
            },
            Section::SideData if key.trim() == "rotation" => {
                if let Ok(r) = val.parse::<f64>() {
                    acc.rotation = Some(r.round() as i32);
                }
            }
            Section::Format => match key.trim() {
                "duration" => {
                    if let Ok(d) = val.parse::<f64>()
                        && d > 0.0
                    {
                        info.duration = d;
                    }
                }
                "bit_rate" => {
                    if let Ok(b) = val.parse() {
                        info.bitrate = b;
                    }
                }
                "format_name" => info.format_name = Some(val.to_string()),
                _ => {}
            },
            _ => {}
        }
    }

    info
}

fn run_ffprobe(path: &Path, args: &[&str], context_msg: &str) -> Result<String> {
    let mut cmd = Command::new("ffprobe");
    cmd.args(["-v", "error"]).args(args).arg(path);
    let output = run_command_cancellable(&mut cmd, context_msg)?;
    if !output.status.success() {
        anyhow::bail!("{}", context_msg);
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Probe video, primary audio, and container metadata in a single `ffprobe` call.
pub fn get_video_info(path: &Path) -> Result<VideoInfo> {
    let stdout = run_ffprobe(
        path,
        &[
            "-show_entries",
            "stream=codec_type,codec_name,width,height,pix_fmt,bit_rate,avg_frame_rate,r_frame_rate,color_transfer,color_primaries,color_space,channels:stream_tags=rotate:stream_side_data=rotation:format=duration,bit_rate,format_name",
            "-of",
            "default=noprint_wrappers=0",
        ],
        "执行ffprobe失败",
    )?;
    Ok(parse_ffprobe_sections(&stdout))
}

/// Probe real duration (seconds) and verify a valid video stream exists.
pub fn probe_output_integrity(path: &Path) -> Option<f64> {
    let text = run_ffprobe(
        path,
        &[
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "default=noprint_wrappers=1",
        ],
        "校验输出视频完整性失败",
    )
    .ok()?;

    let mut has_video = false;
    let mut duration = None;
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            match k.trim() {
                "codec_type" if v.trim() == "video" => has_video = true,
                "duration" => duration = v.trim().parse::<f64>().ok().filter(|&d| d > 0.0),
                _ => {}
            }
        }
    }
    has_video.then_some(duration).flatten()
}

/// Probe whether a media file contains a valid audio stream.
pub fn probe_has_audio(path: &Path) -> bool {
    run_ffprobe(
        path,
        &[
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
        ],
        "校验暂存音频流失败",
    )
    .is_ok_and(|s| s.trim().eq_ignore_ascii_case("audio"))
}

/// Probe the real duration (seconds) of a media file via ffprobe.
pub fn probe_duration(path: &Path) -> Option<f64> {
    run_ffprobe(
        path,
        &[
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
        ],
        "执行ffprobe获取时长失败",
    )
    .ok()?
    .trim()
    .parse::<f64>()
    .ok()
    .filter(|&d| d > 0.0)
}

/// Parse XPSNR Y-channel value from FFmpeg stderr (e.g. `XPSNR  y: 42.9312  u: 46.1 ...`).
pub fn parse_xpsnr_y(stderr: &str) -> Option<f64> {
    for line in stderr.lines().rev() {
        if let Some(pos) = line.find("XPSNR")
            && let Some(ypos) = line[pos..].find("y:")
        {
            let token = line[pos + ypos + 2..].split_whitespace().next()?;
            if token.eq_ignore_ascii_case("inf") {
                return Some(99.0);
            }
            if let Ok(val) = token.parse::<f64>()
                && val.is_finite()
            {
                return Some(val);
            }
        }
    }
    None
}

/// Parse SSIM `All:` value from FFmpeg stderr (e.g. `All:0.984123 (17.99)`).
pub fn parse_ssim_all(stderr: &str) -> Option<f64> {
    for line in stderr.lines().rev() {
        if let Some(pos) = line.find("All:") {
            let token = line[pos + 4..].split_whitespace().next()?;
            if let Ok(val) = token.parse::<f64>()
                && val.is_finite()
            {
                return Some(val);
            }
        }
    }
    None
}

/// Fast 1/3-frame-subsampled perceptual quality measurement (XPSNR or SSIM).
/// For XPSNR, subtracts the empirically calibrated +0.35 dB subsampling offset.
pub fn measure_metric_fast(
    metric: QualityMetric,
    ref_path: &Path,
    dist_path: &Path,
) -> Result<f64> {
    const XPSNR_SUBSAMPLE_OFFSET: f64 = 0.35;
    let (filter_name, ctx_msg) = match metric {
        QualityMetric::Xpsnr => ("xpsnr", "执行XPSNR感知质量评估失败"),
        QualityMetric::Ssim => ("ssim", "执行SSIM质量评估失败"),
    };
    let filt = format!(
        "[0:v]select='not(mod(n,3))',setpts=N/FRAME_RATE/TB,format=yuv420p[r];\
         [1:v]select='not(mod(n,3))',setpts=N/FRAME_RATE/TB,format=yuv420p[d];\
         [r][d]{}",
        filter_name
    );

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-v", "info", "-i"])
        .arg(ref_path)
        .arg("-i")
        .arg(dist_path)
        .args(["-filter_complex", &filt, "-f", "null", "-"]);

    let output = run_command_cancellable(&mut cmd, ctx_msg)?;
    if !output.status.success() {
        anyhow::bail!("{}", ctx_msg);
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    match metric {
        QualityMetric::Xpsnr => {
            let raw = parse_xpsnr_y(&stderr).context("无法解析FFmpeg XPSNR输出")?;
            Ok((raw - XPSNR_SUBSAMPLE_OFFSET).max(0.0))
        }
        QualityMetric::Ssim => parse_ssim_all(&stderr).context("无法解析FFmpeg SSIM输出"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ffprobe_sections_complete_with_audio_hdr_and_rotation() {
        let raw = "\
[STREAM]
codec_name=h264
codec_type=video
width=3840
height=2160
pix_fmt=yuv420p10le
color_space=bt2020nc
color_transfer=smpte2084
color_primaries=bt2020
r_frame_rate=30000/1001
avg_frame_rate=30000/1001
bit_rate=25000000
[SIDE_DATA]
rotation=-90
[/SIDE_DATA]
[/STREAM]
[STREAM]
codec_name=aac
codec_type=audio
channels=6
bit_rate=384000
[/STREAM]
[FORMAT]
format_name=mov,mp4,m4a,3gp,3g2,mj2
duration=120.5
bit_rate=25500000
[/FORMAT]
";
        let info = parse_ffprobe_sections(raw);
        assert_eq!(info.codec_name.as_deref(), Some("h264"));
        assert_eq!(info.width, 3840);
        assert_eq!(info.height, 2160);
        assert_eq!(info.pix_fmt.as_deref(), Some("yuv420p10le"));
        assert!(info.is_hdr());
        assert!(info.is_high_bit_depth());
        assert!((info.fps - 29.97002997).abs() < 1e-4);
        assert_eq!(info.video_bitrate, 25_000_000);
        assert_eq!(info.rotation, -90);
        assert!(info.has_audio);
        assert_eq!(info.audio_codec.as_deref(), Some("aac"));
        assert_eq!(info.audio_channels, 6);
        assert_eq!(info.audio_bitrate, 384_000);
        assert!((info.duration - 120.5).abs() < 1e-6);
        assert_eq!(info.gst_zerocopy_elements(), Some(("qtdemux", "h264parse")));

        let hevc_mkv = VideoInfo {
            codec_name: Some("hevc".into()),
            pix_fmt: Some("yuv420p10le".into()),
            format_name: Some("matroska,webm".into()),
            ..Default::default()
        };
        assert_eq!(
            hevc_mkv.gst_zerocopy_elements(),
            Some(("matroskademux", "h265parse"))
        );
    }

    #[test]
    fn audio_plan_copies_low_bitrate_and_protects_surround() {
        let mut info = VideoInfo {
            has_audio: true,
            audio_codec: Some("aac".into()),
            audio_bitrate: 96_000,
            audio_channels: 2,
            ..Default::default()
        };
        assert_eq!(AudioPlan::decide(&info, 128), AudioPlan::Copy);

        info.audio_codec = Some("pcm_s16le".into());
        info.audio_bitrate = 4_608_000;
        info.audio_channels = 6;
        assert_eq!(AudioPlan::decide(&info, 128), AudioPlan::TranscodeAac(256));
        assert_eq!(AudioPlan::decide(&info, 0), AudioPlan::Copy);
    }

    #[test]
    fn modern_codec_detection_skips_hevc_av1_vp9() {
        let mut info = VideoInfo::default();
        for codec in ["hevc", "av1", "vp9"] {
            info.codec_name = Some(codec.into());
            assert!(info.is_modern_codec().is_some());
        }
        info.codec_name = Some("h264".into());
        assert!(info.is_modern_codec().is_none());
    }

    #[test]
    fn parse_xpsnr_and_ssim_output() {
        let x_stderr = "[Parsed_xpsnr_2 @ 0x123] XPSNR  y: 42.9312  u: 46.1200  v: 46.5000  (minimum: 42.9312)\n";
        assert!((parse_xpsnr_y(x_stderr).unwrap() - 42.9312).abs() < 1e-4);

        let s_stderr =
            "[Parsed_ssim_2 @ 0x123] SSIM Y:0.985 (18.2) U:0.991 V:0.992 All:0.987123 (18.90)\n";
        assert!((parse_ssim_all(s_stderr).unwrap() - 0.987123).abs() < 1e-6);
    }

    #[test]
    fn parse_ffmpeg_encoders_list() {
        let raw = "\
 V..... libx265              libx265 H.265 / HEVC
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder
 V..... hevc_qsv             HEVC (Intel Quick Sync Video acceleration)
 A..... aac                  AAC (Advanced Audio Coding)
";
        let avail = parse_ffmpeg_encoders(raw);
        assert!(avail.libx265);
        assert!(avail.nvenc);
        assert!(avail.qsv);
        assert!(!avail.vaapi);
    }

    #[test]
    fn parse_encoder_arg_handles_aliases_and_rejects_unknown() {
        assert_eq!(parse_encoder_arg("auto").unwrap(), None);
        assert_eq!(
            parse_encoder_arg("jetson").unwrap(),
            Some(EncoderType::JetsonGst)
        );
        assert_eq!(parse_encoder_arg("nvenc").unwrap(), Some(EncoderType::Gpu));
        assert_eq!(
            parse_encoder_arg("vt").unwrap(),
            Some(EncoderType::VideoToolbox)
        );
        assert!(parse_encoder_arg("nvmpi").is_err());
    }
}
