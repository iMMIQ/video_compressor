use crate::config::EncoderType;
use crate::logging::log_info;
use crate::signal::run_command_cancellable;
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, PartialEq)]
pub struct VideoInfo {
    pub codec_name: Option<String>,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub bitrate: u64,
    #[allow(dead_code)]
    pub pix_fmt: Option<String>,
}

/// Summary of FFmpeg encoder capabilities detected in a single probe.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AvailableEncoders {
    pub libx265: bool,
    pub nvenc: bool,
    pub nvmpi: bool,
}

/// Query `ffmpeg -encoders` once and parse HEVC encoder availability.
pub fn detect_available_encoders() -> Result<AvailableEncoders> {
    let output = Command::new("ffmpeg")
        .arg("-encoders")
        .output()
        .context("无法执行ffmpeg命令")?;

    if !output.status.success() {
        return Ok(AvailableEncoders::default());
    }

    let encoders = String::from_utf8_lossy(&output.stdout);
    Ok(parse_ffmpeg_encoders(&encoders))
}

fn parse_ffmpeg_encoders(encoders_stdout: &str) -> AvailableEncoders {
    let mut avail = AvailableEncoders::default();
    for line in encoders_stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('V') {
            continue;
        }
        if trimmed.contains("libx265") {
            avail.libx265 = true;
        }
        if trimmed.contains("hevc_nvenc") {
            avail.nvenc = true;
        }
        if trimmed.contains("hevc_nvmpi") {
            avail.nvmpi = true;
        }
    }
    avail
}

pub fn check_ffmpeg(encoder_type: EncoderType, avail: AvailableEncoders) -> Result<()> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("无法找到ffmpeg命令，请确保已安装FFmpeg")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg执行失败");
    }

    let version = String::from_utf8_lossy(&output.stdout);
    if let Some(line) = version.lines().next() {
        log_info(&format!("检测到: {}", line));
    }

    if avail.libx265 || version.contains("libx265") {
        log_info("✓ H.265编码器可用");
    } else {
        anyhow::bail!("FFmpeg未编译libx265支持，无法进行H.265编码");
    }

    match encoder_type {
        EncoderType::Gpu => {
            if avail.nvenc {
                log_info("✓ NVENC编码器可用");
            } else {
                anyhow::bail!("FFmpeg未编译hevc_nvenc支持，无法使用GPU编码");
            }
        }
        EncoderType::Jetson => {
            if avail.nvmpi {
                log_info("✓ NVMPI编码器可用 (Jetson)");
            } else {
                anyhow::bail!(
                    "FFmpeg未编译hevc_nvmpi支持，无法使用Jetson GPU编码。请参考jetson-ffmpeg编译指南"
                );
            }
        }
        EncoderType::Cpu => {}
    }

    Ok(())
}

/// Parse `ffprobe -of default=noprint_wrappers=1` key=value output robustly.
/// Missing or `N/A` fields never shift subsequent fields.
pub fn parse_ffprobe_kv(output: &str) -> VideoInfo {
    let mut codec_name = None;
    let mut width = None;
    let mut height = None;
    let mut pix_fmt = None;
    let mut duration = None;
    let mut bitrate = None;

    for line in output.lines() {
        let Some((key, raw_val)) = line.split_once('=') else {
            continue;
        };
        let val = raw_val.trim();
        if val.is_empty() || val.eq_ignore_ascii_case("N/A") {
            continue;
        }
        match key.trim() {
            "codec_name" if codec_name.is_none() => codec_name = Some(val.to_string()),
            "width" if width.is_none() => width = val.parse::<u32>().ok(),
            "height" if height.is_none() => height = val.parse::<u32>().ok(),
            "pix_fmt" if pix_fmt.is_none() => pix_fmt = Some(val.to_string()),
            "duration" if duration.is_none() => duration = val.parse::<f64>().ok(),
            "bit_rate" if bitrate.is_none() => bitrate = val.parse::<u64>().ok(),
            _ => {}
        }
    }

    VideoInfo {
        codec_name,
        width: width.unwrap_or(1920),
        height: height.unwrap_or(1080),
        duration: duration.unwrap_or(0.0),
        bitrate: bitrate.unwrap_or(0),
        pix_fmt,
    }
}

pub fn get_video_info(path: &Path) -> Result<VideoInfo> {
    let mut cmd = Command::new("ffprobe");
    cmd.arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=codec_name,width,height,pix_fmt")
        .arg("-show_entries")
        .arg("format=duration,bit_rate")
        .arg("-of")
        .arg("default=noprint_wrappers=1")
        .arg(path);

    let output = run_command_cancellable(&mut cmd, "执行ffprobe失败")?;

    if !output.status.success() {
        anyhow::bail!("ffprobe执行失败");
    }

    let info = String::from_utf8_lossy(&output.stdout);
    Ok(parse_ffprobe_kv(&info))
}

/// Probe the real duration (seconds) of a media file via ffprobe.
/// Returns None if ffprobe fails or reports no usable duration.
pub fn probe_duration(path: &Path) -> Option<f64> {
    let mut cmd = Command::new("ffprobe");
    cmd.arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=nokey=1:noprint_wrappers=1")
        .arg(path);

    let output = run_command_cancellable(&mut cmd, "执行ffprobe获取时长失败").ok()?;
    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<f64>().ok().filter(|d| *d > 0.0)
}

/// Check if running on Jetson (Tegra) platform
fn is_jetson_platform() -> bool {
    if let Ok(release) = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        && release.contains("tegra")
    {
        return true;
    }
    std::path::Path::new("/sys/devices/platform/17000000.gpu/load").exists()
}

/// Resolve encoder type from string, handling "auto" detection using pre-detected encoders.
pub fn resolve_encoder_type(s: &str, avail: AvailableEncoders) -> Result<EncoderType> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(EncoderType::Cpu),
        "gpu" => Ok(EncoderType::Gpu),
        "jetson" => Ok(EncoderType::Jetson),
        "auto" => {
            if is_jetson_platform() {
                // On Jetson (Tegra) platform, use NVMPI only;
                // NVENC is compiled in but does NOT work at runtime.
                if avail.nvmpi {
                    log_info("注意: 使用Jetson GPU编码 (NVMPI可用)");
                    return Ok(EncoderType::Jetson);
                }
                log_info("注意: Jetson平台未找到NVMPI编码器，使用CPU编码");
                return Ok(EncoderType::Cpu);
            }
            if avail.nvenc {
                log_info("注意: 使用GPU编码 (NVENC可用)");
                Ok(EncoderType::Gpu)
            } else if avail.nvmpi {
                log_info("注意: 使用Jetson GPU编码 (NVMPI可用)");
                Ok(EncoderType::Jetson)
            } else {
                log_info("注意: 无GPU编码器可用，使用CPU编码");
                Ok(EncoderType::Cpu)
            }
        }
        _ => anyhow::bail!("Invalid encoder type: {}. Use: cpu, gpu, jetson, or auto", s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ffprobe_kv_complete() {
        let raw = "\
codec_name=h264
width=1280
height=720
pix_fmt=yuv420p
duration=123.456
bit_rate=2500000
";
        let info = parse_ffprobe_kv(raw);
        assert_eq!(info.codec_name.as_deref(), Some("h264"));
        assert_eq!(info.width, 1280);
        assert_eq!(info.height, 720);
        assert_eq!(info.pix_fmt.as_deref(), Some("yuv420p"));
        assert!((info.duration - 123.456).abs() < 1e-6);
        assert_eq!(info.bitrate, 2_500_000);
    }

    #[test]
    fn parse_ffprobe_kv_missing_pix_fmt_and_na_bitrate() {
        // Previously positional parsing shifted duration into pix_fmt here!
        let raw = "\
codec_name=wmv2
width=640
height=480
duration=42.5
bit_rate=N/A
";
        let info = parse_ffprobe_kv(raw);
        assert_eq!(info.codec_name.as_deref(), Some("wmv2"));
        assert_eq!(info.width, 640);
        assert_eq!(info.height, 480);
        assert_eq!(info.pix_fmt, None);
        assert!((info.duration - 42.5).abs() < 1e-6);
        assert_eq!(info.bitrate, 0);
    }

    #[test]
    fn parse_ffmpeg_encoders_list() {
        let raw = "\
 V..... libx265              libx265 H.265 / HEVC
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder
 A..... aac                  AAC (Advanced Audio Coding)
";
        let avail = parse_ffmpeg_encoders(raw);
        assert!(avail.libx265);
        assert!(avail.nvenc);
        assert!(!avail.nvmpi);
    }
}
