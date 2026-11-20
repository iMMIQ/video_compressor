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
        self.color_transfer
            .as_deref()
            .map(|t| {
                let lower = t.to_ascii_lowercase();
                lower == "smpte2084" || lower == "arib-std-b67"
            })
            .unwrap_or(false)
    }

    /// True if the source pixel format is 10-bit or deeper.
    pub fn is_high_bit_depth(&self) -> bool {
        self.pix_fmt
            .as_deref()
            .map(|p| {
                let lower = p.to_ascii_lowercase();
                lower.contains("10") || lower.contains("12") || lower.contains("16")
            })
            .unwrap_or(false)
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

    /// Bits per pixel per frame (BPP) — useful for identifying already heavily compressed sources.
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
        let Some(codec) = self.codec_name.as_deref() else {
            return false;
        };
        let codec_ok = matches!(
            codec.to_ascii_lowercase().as_str(),
            "h264" | "hevc" | "vp9" | "vp8" | "mpeg2video" | "mpeg4" | "vc1" | "av1" | "mjpeg"
        );
        if !codec_ok {
            return false;
        }
        match self.pix_fmt.as_deref() {
            None => true,
            Some(fmt) => matches!(
                fmt.to_ascii_lowercase().as_str(),
                "yuv420p" | "yuvj420p" | "nv12" | "yuv420p10le" | "p010le"
            ),
        }
    }

    /// Determine GStreamer zero-copy (demuxer, parser) pair if the container + codec + pixel format
    /// can go straight through `nvv4l2decoder -> nvvidconv -> nvv4l2h265enc` in NVMM memory.
    pub fn gst_zerocopy_elements(&self) -> Option<(&'static str, &'static str)> {
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

        // Ensure pixel format is 4:2:0 (NVDEC on Jetson does not decode 4:2:2 / 4:4:4 / RGB).
        if let Some(pix) = self.pix_fmt.as_deref() {
            let p = pix.to_ascii_lowercase();
            if !matches!(
                p.as_str(),
                "yuv420p" | "yuvj420p" | "nv12" | "yuv420p10le" | "p010le"
            ) {
                return None;
            }
        }

        Some((demux, parser))
    }
}

/// Smart audio handling strategy to avoid generational loss, 5.1 channel starvation,
/// and low-bitrate audio bloat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPlan {
    /// Input has no audio stream.
    None,
    /// Copy original audio stream untouched (zero quality loss, zero size bloat).
    Copy,
    /// Transcode audio to AAC at the chosen bitrate (kbps).
    TranscodeAac(u32),
}

impl AudioPlan {
    /// Decide whether to copy or transcode audio based on source stream properties and CLI bitrate.
    pub fn decide(info: &VideoInfo, requested_bitrate_kbps: u32) -> Self {
        if !info.has_audio {
            return AudioPlan::None;
        }
        if requested_bitrate_kbps == 0 {
            return AudioPlan::Copy;
        }

        // Adapt target bitrate to channel count so 5.1 surround isn't destroyed at 128kbps.
        let target_kbps = match info.audio_channels {
            0 | 2 => requested_bitrate_kbps,
            1 => requested_bitrate_kbps.min(96),
            3..=4 => requested_bitrate_kbps.max(192),
            _ => requested_bitrate_kbps.max(256),
        };

        let mp4_compatible = info
            .audio_codec
            .as_deref()
            .map(|c| {
                matches!(
                    c.to_ascii_lowercase().as_str(),
                    "aac" | "mp3" | "ac3" | "eac3" | "opus"
                )
            })
            .unwrap_or(false);

        // If already MP4-compatible and within 15% of (or below!) target bitrate, copy directly.
        if mp4_compatible && info.audio_bitrate > 0 {
            let threshold_bps = (target_kbps as u64) * 1000 * 115 / 100;
            if info.audio_bitrate <= threshold_bps {
                return AudioPlan::Copy;
            }
        }

        AudioPlan::TranscodeAac(target_kbps)
    }

    /// Estimated final audio bitrate in bits/sec for accurate video-budget accounting.
    pub fn estimated_bps(self, info: &VideoInfo) -> u64 {
        match self {
            AudioPlan::None => 0,
            AudioPlan::Copy => {
                if info.audio_bitrate > 0 {
                    info.audio_bitrate
                } else {
                    128_000
                }
            }
            AudioPlan::TranscodeAac(kbps) => (kbps as u64) * 1000,
        }
    }

    /// Append audio arguments to an FFmpeg command.
    pub fn append_ffmpeg_args(self, cmd: &mut Command) {
        match self {
            AudioPlan::None => {
                cmd.arg("-an");
            }
            AudioPlan::Copy => {
                cmd.arg("-c:a").arg("copy");
            }
            AudioPlan::TranscodeAac(kbps) => {
                cmd.arg("-c:a")
                    .arg("aac")
                    .arg("-b:a")
                    .arg(format!("{}k", kbps));
            }
        }
    }
}

/// Summary of FFmpeg compiled-in encoders detected in a single probe.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AvailableEncoders {
    pub libx265: bool,
    pub libx264: bool,
    pub nvenc: bool,
    pub nvmpi: bool,
    pub qsv: bool,
    pub vaapi: bool,
    pub amf: bool,
    pub videotoolbox: bool,
}

/// Parse `ffmpeg -encoders` output once.
pub fn parse_ffmpeg_encoders(encoders_stdout: &str) -> AvailableEncoders {
    let mut avail = AvailableEncoders::default();
    for line in encoders_stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('V') {
            continue;
        }
        if trimmed.contains("libx265") {
            avail.libx265 = true;
        }
        if trimmed.contains("libx264") {
            avail.libx264 = true;
        }
        if trimmed.contains("hevc_nvenc") {
            avail.nvenc = true;
        }
        if trimmed.contains("hevc_nvmpi") {
            avail.nvmpi = true;
        }
        if trimmed.contains("hevc_qsv") {
            avail.qsv = true;
        }
        if trimmed.contains("hevc_vaapi") {
            avail.vaapi = true;
        }
        if trimmed.contains("hevc_amf") {
            avail.amf = true;
        }
        if trimmed.contains("hevc_videotoolbox") {
            avail.videotoolbox = true;
        }
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
    let mut has_xpsnr = false;
    let mut has_ssim = false;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let _flags = parts.next();
        if let Some(name) = parts.next() {
            if name == "xpsnr" {
                has_xpsnr = true;
            } else if name == "ssim" {
                has_ssim = true;
            }
        }
    }
    (has_xpsnr, has_ssim)
}

fn probe_x265_10bit() -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-h", "encoder=libx265"])
        .output()
        .map(|o| {
            o.status.success() && String::from_utf8_lossy(&o.stdout).contains("yuv420p10le")
        })
        .unwrap_or(false)
}

/// Run a tiny 2-frame lavfi encode to verify that an FFmpeg hardware encoder
/// actually initializes on real hardware (not just compiled in).
fn probe_ffmpeg_hw_encode(extra_args: &[&str]) -> bool {
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=256x256:d=0.07:r=30",
    ]);
    cmd.args(extra_args);
    cmd.args(["-f", "null", "-"]);
    cmd.output().map(|o| o.status.success()).unwrap_or(false)
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
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn probe_gst_element_exists(element: &str) -> bool {
    Command::new("gst-inspect-1.0")
        .arg(element)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
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
    for idx in 128..=135 {
        let candidate = format!("/dev/dri/renderD{}", idx);
        if Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    None
}

/// Check if running on NVIDIA Jetson (Tegra) platform.
pub fn is_jetson_platform() -> bool {
    if let Ok(release) = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        && release.contains("tegra")
    {
        return true;
    }
    Path::new("/dev/v4l2-nvenc").exists()
        || Path::new("/sys/devices/platform/17000000.gpu").exists()
}

/// Perform real runtime hardware capability probing once at startup.
pub fn detect_hardware_capabilities() -> Result<HardwareCapabilities> {
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

    let on_jetson = is_jetson_platform();

    if on_jetson {
        // Prefer official JetPack GStreamer nvv4l2h265enc (zero custom FFmpeg required!).
        if probe_gst_nvv4l2(true) {
            caps.jetson_gst_ok = true;
            caps.jetson_gst_10bit = true;
        } else if probe_gst_nvv4l2(false) {
            caps.jetson_gst_ok = true;
            caps.jetson_gst_10bit = false;
        }
        if caps.jetson_gst_ok {
            caps.jetson_gst_decoder = probe_gst_element_exists("nvv4l2decoder")
                && probe_gst_element_exists("matroskamux");
        }
        if enc.nvmpi {
            caps.jetson_nvmpi_ok = probe_ffmpeg_hw_encode(&["-c:v", "hevc_nvmpi", "-b:v", "1000k"]);
        }
        log_debug(&format!("Jetson硬件探测结果: {:?}", caps));
        return Ok(caps);
    }

    // Non-Jetson desktop / server platforms: runtime-verify available hardware encoders.
    if enc.nvenc {
        // Probe full modern NVENC feature set first (Turing/Ampere/Ada/Blackwell).
        if probe_ffmpeg_hw_encode(&[
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "28",
            "-b:v",
            "0",
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
            caps.nvenc_uhq = probe_ffmpeg_hw_encode(&[
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "p6",
                "-tune",
                "uhq",
                "-rc",
                "vbr",
                "-cq",
                "28",
                "-b:v",
                "0",
            ]);
            caps.nvenc_gpu_count = detect_nvidia_gpu_count();
        } else if probe_ffmpeg_hw_encode(&[
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "28",
            "-b:v",
            "0",
        ]) {
            // Older GPU (e.g. Pascal) — verify sub-features individually.
            caps.nvenc_ok = true;
            caps.nvenc_10bit = probe_ffmpeg_hw_encode(&[
                "-c:v",
                "hevc_nvenc",
                "-profile:v",
                "main10",
                "-pix_fmt",
                "p010le",
                "-rc",
                "vbr",
                "-cq",
                "28",
                "-b:v",
                "0",
            ]);
            caps.nvenc_bframes = probe_ffmpeg_hw_encode(&[
                "-c:v",
                "hevc_nvenc",
                "-bf",
                "4",
                "-b_ref_mode",
                "middle",
                "-rc",
                "vbr",
                "-cq",
                "28",
                "-b:v",
                "0",
            ]);
            caps.nvenc_multipass = probe_ffmpeg_hw_encode(&[
                "-c:v",
                "hevc_nvenc",
                "-multipass",
                "qres",
                "-rc",
                "vbr",
                "-cq",
                "28",
                "-b:v",
                "0",
            ]);
            caps.nvenc_gpu_count = detect_nvidia_gpu_count();
        }
    }

    if !caps.nvenc_ok && enc.qsv {
        caps.qsv_ok = probe_ffmpeg_hw_encode(&[
            "-c:v",
            "hevc_qsv",
            "-global_quality",
            "26",
            "-look_ahead",
            "0",
        ]);
    }

    if !caps.nvenc_ok && !caps.qsv_ok && enc.videotoolbox {
        caps.videotoolbox_ok = probe_ffmpeg_hw_encode(&["-c:v", "hevc_videotoolbox", "-q:v", "65"]);
    }

    if !caps.nvenc_ok && !caps.qsv_ok && !caps.videotoolbox_ok && enc.amf {
        caps.amf_ok = probe_ffmpeg_hw_encode(&[
            "-c:v",
            "hevc_amf",
            "-rc",
            "cqp",
            "-qp_i",
            "26",
            "-qp_p",
            "26",
        ]);
    }

    if !caps.nvenc_ok && !caps.qsv_ok && !caps.videotoolbox_ok && !caps.amf_ok && enc.vaapi
        && let Some(dev) = find_vaapi_device()
    {
        let ok = Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-v",
                "error",
                "-vaapi_device",
                &dev,
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.07:r=30",
                "-vf",
                "format=nv12,hwupload",
                "-c:v",
                "hevc_vaapi",
                "-rc_mode",
                "CQP",
                "-qp",
                "26",
                "-f",
                "null",
                "-",
            ])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            caps.vaapi_ok = true;
            caps.vaapi_device = Some(dev);
        }
    }

    log_debug(&format!("硬件探测结果: {:?}", caps));
    Ok(caps)
}

/// Resolve CLI `--encoder` string against runtime-verified hardware capabilities.
pub fn resolve_encoder_type(s: &str, caps: &HardwareCapabilities) -> Result<EncoderType> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(EncoderType::Cpu),
        "gpu" | "nvenc" => Ok(EncoderType::Gpu),
        "jetson" => {
            if caps.jetson_gst_ok {
                Ok(EncoderType::JetsonGst)
            } else if caps.jetson_nvmpi_ok {
                log_info("注意: 未检测到官方 GStreamer nvv4l2h265enc，回退到兼容模式 hevc_nvmpi");
                Ok(EncoderType::JetsonNvmpi)
            } else {
                Ok(EncoderType::JetsonGst)
            }
        }
        "jetson-gst" | "gstreamer" | "v4l2" => Ok(EncoderType::JetsonGst),
        "jetson-nvmpi" | "nvmpi" => Ok(EncoderType::JetsonNvmpi),
        "qsv" => Ok(EncoderType::Qsv),
        "vaapi" => Ok(EncoderType::Vaapi),
        "amf" => Ok(EncoderType::Amf),
        "videotoolbox" | "vt" => Ok(EncoderType::VideoToolbox),
        "auto" => {
            if caps.jetson_gst_ok {
                log_info("自动选择: Jetson 官方 GStreamer nvv4l2h265enc (零自编译FFmpeg)");
                Ok(EncoderType::JetsonGst)
            } else if caps.nvenc_ok {
                log_info("自动选择: NVIDIA GPU NVENC (hevc_nvenc)");
                Ok(EncoderType::Gpu)
            } else if caps.qsv_ok {
                log_info("自动选择: Intel QSV (hevc_qsv)");
                Ok(EncoderType::Qsv)
            } else if caps.videotoolbox_ok {
                log_info("自动选择: Apple VideoToolbox (hevc_videotoolbox)");
                Ok(EncoderType::VideoToolbox)
            } else if caps.amf_ok {
                log_info("自动选择: AMD AMF (hevc_amf)");
                Ok(EncoderType::Amf)
            } else if caps.vaapi_ok {
                log_info("自动选择: Linux VA-API (hevc_vaapi)");
                Ok(EncoderType::Vaapi)
            } else if caps.jetson_nvmpi_ok {
                log_info("自动选择: Jetson NVMPI (hevc_nvmpi, 兼容模式)");
                Ok(EncoderType::JetsonNvmpi)
            } else {
                log_info("注意: 未检测到可用硬件编码器，使用 CPU (libx265)");
                Ok(EncoderType::Cpu)
            }
        }
        _ => anyhow::bail!(
            "无效的编码器类型: {}。可选: auto, cpu, gpu, jetson, jetson-gst, nvmpi, qsv, vaapi, amf, videotoolbox",
            s
        ),
    }
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

    let version = String::from_utf8_lossy(&output.stdout);
    if let Some(line) = version.lines().next() {
        log_info(&format!("检测到: {}", line));
    }

    let metric_str = if caps.has_xpsnr {
        "XPSNR (感知加权)"
    } else if caps.has_ssim {
        "SSIM (结构相似)"
    } else {
        "仅码率估算"
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
        EncoderType::JetsonNvmpi => {
            if !caps.jetson_nvmpi_ok {
                anyhow::bail!("FFmpeg未提供可用的hevc_nvmpi编码器");
            }
            log_info("✓ Jetson hevc_nvmpi 可用 (建议优先使用官方 jetson-gst)");
        }
        EncoderType::Qsv => {
            if !caps.qsv_ok {
                anyhow::bail!("Intel QSV (hevc_qsv) 初始化失败");
            }
            log_info("✓ Intel QSV (hevc_qsv) 可用");
        }
        EncoderType::Vaapi => {
            if !caps.vaapi_ok {
                anyhow::bail!("Linux VA-API (hevc_vaapi) 初始化失败");
            }
            log_info(&format!(
                "✓ VA-API (hevc_vaapi) 可用 (设备: {})",
                caps.vaapi_device.as_deref().unwrap_or("default")
            ));
        }
        EncoderType::Amf => {
            if !caps.amf_ok {
                anyhow::bail!("AMD AMF (hevc_amf) 初始化失败");
            }
            log_info("✓ AMD AMF (hevc_amf) 可用");
        }
        EncoderType::VideoToolbox => {
            if !caps.videotoolbox_ok {
                anyhow::bail!("Apple VideoToolbox (hevc_videotoolbox) 初始化失败");
            }
            log_info("✓ Apple VideoToolbox (hevc_videotoolbox) 可用");
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
    if val.is_finite() && (1.0..=240.0).contains(&val) {
        Some(val)
    } else {
        None
    }
}

/// Parse `ffprobe -of default=noprint_wrappers=0` sectioned output containing
/// `[STREAM]` blocks (video + audio) and `[FORMAT]` block robustly.
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
    let mut cur_codec_type: Option<String> = None;
    let mut seen_video = false;
    let mut seen_audio = false;

    // Temporary fields for current [STREAM]
    let mut s_codec: Option<String> = None;
    let mut s_width: Option<u32> = None;
    let mut s_height: Option<u32> = None;
    let mut s_pix_fmt: Option<String> = None;
    let mut s_bitrate: Option<u64> = None;
    let mut s_avg_fps: Option<f64> = None;
    let mut s_r_fps: Option<f64> = None;
    let mut s_color_transfer: Option<String> = None;
    let mut s_color_primaries: Option<String> = None;
    let mut s_color_space: Option<String> = None;
    let mut s_channels: Option<u32> = None;
    let mut s_rotation: Option<i32> = None;

    let flush_stream = |info: &mut VideoInfo,
                        seen_video: &mut bool,
                        seen_audio: &mut bool,
                        cur_codec_type: &mut Option<String>,
                        s_codec: &mut Option<String>,
                        s_width: &mut Option<u32>,
                        s_height: &mut Option<u32>,
                        s_pix_fmt: &mut Option<String>,
                        s_bitrate: &mut Option<u64>,
                        s_avg_fps: &mut Option<f64>,
                        s_r_fps: &mut Option<f64>,
                        s_color_transfer: &mut Option<String>,
                        s_color_primaries: &mut Option<String>,
                        s_color_space: &mut Option<String>,
                        s_channels: &mut Option<u32>,
                        s_rotation: &mut Option<i32>| {
        match cur_codec_type.as_deref() {
            Some("video") if !*seen_video => {
                *seen_video = true;
                info.codec_name = s_codec.take();
                if let Some(w) = s_width.take() {
                    info.width = w;
                }
                if let Some(h) = s_height.take() {
                    info.height = h;
                }
                info.pix_fmt = s_pix_fmt.take();
                info.video_bitrate = s_bitrate.take().unwrap_or(0);
                info.fps = s_avg_fps.take().or(s_r_fps.take()).unwrap_or(30.0);
                info.color_transfer = s_color_transfer.take();
                info.color_primaries = s_color_primaries.take();
                info.color_space = s_color_space.take();
                info.rotation = s_rotation.take().unwrap_or(0);
            }
            Some("audio") if !*seen_audio => {
                *seen_audio = true;
                info.has_audio = true;
                info.audio_codec = s_codec.take();
                info.audio_bitrate = s_bitrate.take().unwrap_or(0);
                info.audio_channels = s_channels.take().unwrap_or(2);
            }
            _ => {}
        }
        *cur_codec_type = None;
        *s_codec = None;
        *s_width = None;
        *s_height = None;
        *s_pix_fmt = None;
        *s_bitrate = None;
        *s_avg_fps = None;
        *s_r_fps = None;
        *s_color_transfer = None;
        *s_color_primaries = None;
        *s_color_space = None;
        *s_channels = None;
        *s_rotation = None;
    };

    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line == "[STREAM]" {
            section = Section::Stream;
            continue;
        } else if line == "[/STREAM]" {
            flush_stream(
                &mut info,
                &mut seen_video,
                &mut seen_audio,
                &mut cur_codec_type,
                &mut s_codec,
                &mut s_width,
                &mut s_height,
                &mut s_pix_fmt,
                &mut s_bitrate,
                &mut s_avg_fps,
                &mut s_r_fps,
                &mut s_color_transfer,
                &mut s_color_primaries,
                &mut s_color_space,
                &mut s_channels,
                &mut s_rotation,
            );
            section = Section::None;
            continue;
        } else if line == "[SIDE_DATA]" {
            section = Section::SideData;
            continue;
        } else if line == "[/SIDE_DATA]" {
            section = Section::Stream;
            continue;
        } else if line == "[FORMAT]" {
            section = Section::Format;
            continue;
        } else if line == "[/FORMAT]" {
            section = Section::None;
            continue;
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
                "codec_type" => cur_codec_type = Some(val.to_ascii_lowercase()),
                "codec_name" => s_codec = Some(val.to_string()),
                "width" => s_width = val.parse().ok(),
                "height" => s_height = val.parse().ok(),
                "pix_fmt" => s_pix_fmt = Some(val.to_string()),
                "bit_rate" => s_bitrate = val.parse().ok(),
                "avg_frame_rate" => s_avg_fps = parse_fps_fraction(val),
                "r_frame_rate" => s_r_fps = parse_fps_fraction(val),
                "color_transfer" => s_color_transfer = Some(val.to_string()),
                "color_primaries" => s_color_primaries = Some(val.to_string()),
                "color_space" => s_color_space = Some(val.to_string()),
                "channels" => s_channels = val.parse().ok(),
                "TAG:rotate" => {
                    if let Ok(r) = val.parse::<i32>() {
                        s_rotation = Some(r);
                    }
                }
                _ => {}
            },
            Section::SideData => {
                if key.trim() == "rotation"
                    && let Ok(r) = val.parse::<f64>()
                {
                    s_rotation = Some(r.round() as i32);
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
                    if let Ok(b) = val.parse::<u64>() {
                        info.bitrate = b;
                    }
                }
                "format_name" => {
                    info.format_name = Some(val.to_string());
                }
                _ => {}
            },
            Section::None => {}
        }
    }

    // Backwards-compatible fallback for wrapperless key=value lines (e.g., older tests)
    if !seen_video && info.codec_name.is_none() {
        for raw_line in output.lines() {
            let Some((key, raw_val)) = raw_line.split_once('=') else {
                continue;
            };
            let val = raw_val.trim();
            if val.is_empty() || val.eq_ignore_ascii_case("N/A") {
                continue;
            }
            match key.trim() {
                "codec_name" if info.codec_name.is_none() => info.codec_name = Some(val.to_string()),
                "width" => {
                    if let Ok(w) = val.parse() {
                        info.width = w;
                    }
                }
                "height" => {
                    if let Ok(h) = val.parse() {
                        info.height = h;
                    }
                }
                "pix_fmt" if info.pix_fmt.is_none() => info.pix_fmt = Some(val.to_string()),
                "duration" => {
                    if let Ok(d) = val.parse::<f64>() {
                        info.duration = d;
                    }
                }
                "bit_rate" => {
                    if let Ok(b) = val.parse::<u64>() {
                        info.bitrate = b;
                    }
                }
                _ => {}
            }
        }
    }

    info
}

/// Probe video, primary audio, and container metadata in a single `ffprobe` call.
pub fn get_video_info(path: &Path) -> Result<VideoInfo> {
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,bit_rate,avg_frame_rate,r_frame_rate,color_transfer,color_primaries,color_space,channels:stream_tags=rotate:stream_side_data=rotation:format=duration,bit_rate,format_name",
        "-of",
        "default=noprint_wrappers=0",
    ])
    .arg(path);

    let output = run_command_cancellable(&mut cmd, "执行ffprobe失败")?;
    if !output.status.success() {
        anyhow::bail!("ffprobe执行失败");
    }

    let info_text = String::from_utf8_lossy(&output.stdout);
    Ok(parse_ffprobe_sections(&info_text))
}

/// Probe real duration (seconds) and verify a valid video stream exists.
pub fn probe_output_integrity(path: &Path) -> Option<f64> {
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type:format=duration",
        "-of",
        "default=noprint_wrappers=1",
    ])
    .arg(path);

    let output = run_command_cancellable(&mut cmd, "校验输出视频完整性失败").ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut has_video = false;
    let mut duration = None;
    for line in text.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "codec_type" if v.trim() == "video" => has_video = true,
            "duration" => {
                if let Ok(d) = v.trim().parse::<f64>()
                    && d > 0.0
                {
                    duration = Some(d);
                }
            }
            _ => {}
        }
    }

    if has_video { duration } else { None }
}

/// Probe the real duration (seconds) of a media file via ffprobe.
pub fn probe_duration(path: &Path) -> Option<f64> {
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
    ])
    .arg(path);

    let output = run_command_cancellable(&mut cmd, "执行ffprobe获取时长失败").ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<f64>().ok().filter(|d| *d > 0.0)
}

/// Parse XPSNR Y-channel value from FFmpeg stderr (e.g. `XPSNR  y: 42.9312  u: 46.1 ...`).
pub fn parse_xpsnr_y(stderr: &str) -> Option<f64> {
    for line in stderr.lines().rev() {
        if let Some(pos) = line.find("XPSNR") {
            let tail = &line[pos..];
            if let Some(ypos) = tail.find("y:") {
                let after = tail[ypos + 2..].trim_start();
                let token = after.split_whitespace().next()?;
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
    }
    None
}

/// Parse SSIM `All:` value from FFmpeg stderr (e.g. `All:0.984123 (17.99)`).
pub fn parse_ssim_all(stderr: &str) -> Option<f64> {
    for line in stderr.lines().rev() {
        if let Some(pos) = line.find("All:") {
            let after = &line[pos + 4..];
            let token = after.split_whitespace().next()?;
            if let Ok(val) = token.parse::<f64>()
                && val.is_finite()
            {
                return Some(val);
            }
        }
    }
    None
}

/// Fast 1/3-frame-subsampled XPSNR measurement between reference sample and encoded candidate.
/// Empirically validated on Jetson Orin: 2× faster than full-frame XPSNR with a consistent
/// ~+0.35 dB offset, which we subtract here to return the calibrated full-frame equivalent.
pub fn measure_xpsnr_fast(ref_path: &Path, dist_path: &Path) -> Result<f64> {
    const SUBSAMPLE_CALIBRATION_OFFSET: f64 = 0.35;
    let filt = "[0:v]select='not(mod(n,3))',setpts=N/FRAME_RATE/TB,format=yuv420p[r];\
                [1:v]select='not(mod(n,3))',setpts=N/FRAME_RATE/TB,format=yuv420p[d];\
                [r][d]xpsnr";
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-v", "info", "-i"])
        .arg(ref_path)
        .arg("-i")
        .arg(dist_path)
        .args(["-filter_complex", filt, "-f", "null", "-"]);

    let output = run_command_cancellable(&mut cmd, "执行XPSNR感知质量评估失败")?;
    if !output.status.success() {
        anyhow::bail!("XPSNR评估失败");
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let raw = parse_xpsnr_y(&stderr).context("无法解析FFmpeg XPSNR输出")?;
    Ok((raw - SUBSAMPLE_CALIBRATION_OFFSET).max(0.0))
}

/// Fast 1/3-frame-subsampled SSIM measurement fallback when FFmpeg lacks `xpsnr`.
pub fn measure_ssim_fast(ref_path: &Path, dist_path: &Path) -> Result<f64> {
    let filt = "[0:v]select='not(mod(n,3))',setpts=N/FRAME_RATE/TB,format=yuv420p[r];\
                [1:v]select='not(mod(n,3))',setpts=N/FRAME_RATE/TB,format=yuv420p[d];\
                [r][d]ssim";
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-v", "info", "-i"])
        .arg(ref_path)
        .arg("-i")
        .arg(dist_path)
        .args(["-filter_complex", filt, "-f", "null", "-"]);

    let output = run_command_cancellable(&mut cmd, "执行SSIM质量评估失败")?;
    if !output.status.success() {
        anyhow::bail!("SSIM评估失败");
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    parse_ssim_all(&stderr).context("无法解析FFmpeg SSIM输出")
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
        // 96k AAC <= 128k*1.15 -> copy (avoid generational loss & bloat!)
        assert_eq!(AudioPlan::decide(&info, 128), AudioPlan::Copy);

        // 5.1 PCM at 4.6 Mbps -> transcode to at least 256kbps AAC, NOT 128kbps!
        info.audio_codec = Some("pcm_s16le".into());
        info.audio_bitrate = 4_608_000;
        info.audio_channels = 6;
        assert_eq!(AudioPlan::decide(&info, 128), AudioPlan::TranscodeAac(256));

        // Explicit -a 0 -> always copy
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

        let s_stderr = "[Parsed_ssim_2 @ 0x123] SSIM Y:0.985 (18.2) U:0.991 V:0.992 All:0.987123 (18.90)\n";
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
        assert!(!avail.nvmpi);
    }
}
