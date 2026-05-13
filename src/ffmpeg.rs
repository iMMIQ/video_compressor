use crate::config::EncoderType;
use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

#[derive(Debug)]
pub struct VideoInfo {
    pub codec_name: Option<String>,
    #[allow(dead_code)]
    pub width: u32,
    #[allow(dead_code)]
    pub height: u32,
    pub duration: f64,
    #[allow(dead_code)]
    pub bitrate: u64,
    pub pix_fmt: Option<String>,
}

/// Check if NVENC encoder is available (desktop GPUs)
pub fn check_nvenc_available() -> Result<bool> {
    let output = Command::new("ffmpeg")
        .arg("-encoders")
        .output()
        .context("无法执行ffmpeg命令")?;

    if !output.status.success() {
        return Ok(false);
    }

    let encoders = String::from_utf8_lossy(&output.stdout);
    let nvenc_available = encoders
        .lines()
        .any(|line| line.trim().starts_with('V') && line.contains("hevc_nvenc"));
    Ok(nvenc_available)
}

/// Check if NVMPI encoder is available (Jetson devices)
pub fn check_nvmpi_available() -> Result<bool> {
    let output = Command::new("ffmpeg")
        .arg("-encoders")
        .output()
        .context("无法执行ffmpeg命令")?;

    if !output.status.success() {
        return Ok(false);
    }

    let encoders = String::from_utf8_lossy(&output.stdout);
    let nvmpi_available = encoders
        .lines()
        .any(|line| line.trim().starts_with('V') && line.contains("hevc_nvmpi"));
    Ok(nvmpi_available)
}

pub fn check_ffmpeg(encoder_type: EncoderType) -> Result<()> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("无法找到ffmpeg命令，请确保已安装FFmpeg")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg执行失败");
    }

    let version = String::from_utf8_lossy(&output.stdout);
    let version_lines: Vec<&str> = version.lines().collect();
    if let Some(line) = version_lines.first() {
        println!("检测到: {}", line);
    }

    // Check for libx265 support
    if version.contains("libx265") {
        println!("✓ H.265编码器可用");
    } else {
        anyhow::bail!("FFmpeg未编译libx265支持，无法进行H.265编码");
    }

    match encoder_type {
        EncoderType::Gpu => {
            if check_nvenc_available()? {
                println!("✓ NVENC编码器可用");
            } else {
                anyhow::bail!("FFmpeg未编译hevc_nvenc支持，无法使用GPU编码");
            }
        }
        EncoderType::Jetson => {
            if check_nvmpi_available()? {
                println!("✓ NVMPI编码器可用 (Jetson)");
            } else {
                anyhow::bail!("FFmpeg未编译hevc_nvmpi支持，无法使用Jetson GPU编码。请参考jetson-ffmpeg编译指南");
            }
        }
        EncoderType::Cpu => {}
    }

    Ok(())
}

pub fn get_video_info(path: &Path) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=codec_name,width,height,pix_fmt")
        .arg("-show_entries")
        .arg("format=duration,bit_rate")
        .arg("-of")
        .arg("default=nokey=1:noprint_wrappers=1")
        .arg(path)
        .output()
        .context("执行ffprobe失败")?;

    if !output.status.success() {
        anyhow::bail!("ffprobe执行失败");
    }

    let info = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = info.lines().collect();

    let codec_name = lines.get(0).map(|s| s.to_string());
    let width = lines.get(1).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height = lines.get(2).and_then(|s| s.parse().ok()).unwrap_or(1080);
    let pix_fmt = lines.get(3).filter(|s| !s.is_empty()).map(|s| s.to_string());
    let duration = lines.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let bitrate = lines.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    Ok(VideoInfo {
        codec_name,
        width,
        height,
        duration,
        bitrate,
        pix_fmt,
    })
}

/// Check if running on Jetson (Tegra) platform
fn is_jetson_platform() -> bool {
    // Check kernel release for "tegra" suffix
    if let Ok(release) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
        if release.contains("tegra") {
            return true;
        }
    }
    // Check Jetson GPU sysfs path
    std::path::Path::new("/sys/devices/platform/17000000.gpu/load").exists()
}

/// Resolve encoder type from string, handling "auto" detection
pub fn resolve_encoder_type(s: &str) -> Result<EncoderType> {
    match s.to_lowercase().as_str() {
        "cpu" => Ok(EncoderType::Cpu),
        "gpu" => Ok(EncoderType::Gpu),
        "jetson" => Ok(EncoderType::Jetson),
        "auto" => {
            if is_jetson_platform() {
                // On Jetson (Tegra) platform, use NVMPI only;
                // NVENC is compiled in but does NOT work at runtime
                // (Host1x channel open failed, double free crash)
                if check_nvmpi_available()? {
                    println!("注意: 使用Jetson GPU编码 (NVMPI可用)");
                    return Ok(EncoderType::Jetson);
                }
                println!("注意: Jetson平台未找到NVMPI编码器，使用CPU编码");
                return Ok(EncoderType::Cpu);
            }
            if check_nvenc_available()? {
                println!("注意: 使用GPU编码 (NVENC可用)");
                Ok(EncoderType::Gpu)
            } else if check_nvmpi_available()? {
                println!("注意: 使用Jetson GPU编码 (NVMPI可用)");
                Ok(EncoderType::Jetson)
            } else {
                println!("注意: 无GPU编码器可用，使用CPU编码");
                Ok(EncoderType::Cpu)
            }
        }
        _ => anyhow::bail!("Invalid encoder type: {}. Use: cpu, gpu, jetson, or auto", s),
    }
}
