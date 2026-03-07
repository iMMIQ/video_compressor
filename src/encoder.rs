use crate::config::{EncoderConfig, EncoderType};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::NamedTempFile;

pub struct PreviewResult {
    pub output_size_per_second: f64, // Bytes per second after preview encoding
}

/// Build hardware decode arguments for FFmpeg (GPU mode only)
pub fn build_decode_args(config: &EncoderConfig) -> Vec<String> {
    match config.encoder_type {
        EncoderType::Gpu => vec!["-hwaccel".to_string(), "cuda".to_string()],
        // Jetson nvmpi: no hwaccel flag needed, decoder is specified separately if desired
        _ => vec![],
    }
}

/// Build video encoder arguments for FFmpeg
pub fn build_encode_args(crf: u8, preset: &str, config: &EncoderConfig) -> Vec<String> {
    match config.encoder_type {
        EncoderType::Cpu => vec![
            "-c:v".to_string(),
            "libx265".to_string(),
            "-crf".to_string(),
            crf.to_string(),
            "-preset".to_string(),
            preset.to_string(),
            "-x265-params".to_string(),
            "fast=1:log-level=error".to_string(),
        ],
        EncoderType::Gpu => vec![
            "-c:v".to_string(),
            "hevc_nvenc".to_string(),
            "-preset".to_string(),
            "p7".to_string(),
            "-tune".to_string(),
            "hq".to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-cq".to_string(),
            (crf + 7).to_string(),
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
        ],
        EncoderType::Jetson => vec![
            "-c:v".to_string(),
            "hevc_nvmpi".to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-qp".to_string(),
            (crf + 7).to_string(),
        ],
    }
}

pub fn preview_encode(
    input_path: &Path,
    crf: u8,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
    duration: u32,
    video_duration: f64,
) -> Result<PreviewResult> {
    // Create temp file for output
    let temp_output = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;
    let temp_output_path = temp_output.path();

    let segment_paths: Vec<PathBuf>;

    if video_duration <= duration as f64 * 1.5 {
        // Video too short, extract raw segment from beginning
        let temp_segment = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;

        let extract_cmd = Command::new("ffmpeg")
            .arg("-y")
            .arg("-i")
            .arg(input_path)
            .arg("-t")
            .arg(duration.to_string())
            .arg("-c")
            .arg("copy")
            .arg(temp_segment.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg提取预览片段失败")?;

        if !extract_cmd.status.success() {
            drop(temp_segment);
            anyhow::bail!("提取预览片段失败");
        }

        segment_paths = vec![temp_segment.path().to_path_buf()];
        // Keep temp_segment alive until encoding completes
        let _keep = temp_segment;

        // Encode raw segment to H.265 (with audio processing, same as full encode)
        let decode_args = build_decode_args(encoder_config);
        let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
        let mut encode_cmd = Command::new("ffmpeg");
        encode_cmd.arg("-y");
        for arg in &decode_args {
            encode_cmd.arg(arg);
        }
        encode_cmd.arg("-i").arg(&segment_paths[0]);
        for arg in video_args {
            encode_cmd.arg(arg);
        }
        encode_cmd.arg("-f").arg("mp4");

        // Audio processing (same as full encode)
        if audio_bitrate > 0 {
            encode_cmd
                .arg("-c:a")
                .arg("aac")
                .arg("-b:a")
                .arg(format!("{}k", audio_bitrate));
        } else {
            encode_cmd.arg("-c:a").arg("copy");
        }

        encode_cmd.arg(temp_output_path);

        let output = encode_cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg预览编码失败")?;

        if !output.status.success() {
            anyhow::bail!("预览编码失败");
        }

        drop(_keep);
    } else {
        // Five-segment uniform sampling: at 10%, 30%, 50%, 70%, 90% positions
        const NUM_SEGMENTS: u32 = 5;
        const SAMPLE_POSITIONS: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];

        let segment_duration = duration / NUM_SEGMENTS;
        let remaining = duration % NUM_SEGMENTS;

        // Calculate start time for each segment
        let seg_starts: Vec<f64> = SAMPLE_POSITIONS
            .iter()
            .map(|&pos| {
                (video_duration * pos - segment_duration as f64 / 2.0).max(0.0)
            })
            .collect();

        // Duration for each segment (last segment gets remainder)
        let seg_durs: Vec<u32> = (0..NUM_SEGMENTS)
            .map(|i| {
                if i == NUM_SEGMENTS - 1 {
                    segment_duration + remaining
                } else {
                    segment_duration
                }
            })
            .collect();

        // Extract five raw segments to temp files
        let temp_segs: Vec<NamedTempFile> = (0..NUM_SEGMENTS)
            .map(|i| {
                NamedTempFile::with_suffix(".mp4").context(format!("无法创建临时文件{}", i + 1))
            })
            .collect::<Result<Vec<_>>>()?;

        // Extract each segment
        let mut extract_success = true;
        for (i, temp_seg) in temp_segs.iter().enumerate() {
            let extract = Command::new("ffmpeg")
                .arg("-y")
                .arg("-ss")
                .arg(format!("{:.1}", seg_starts[i]))
                .arg("-i")
                .arg(input_path)
                .arg("-t")
                .arg(seg_durs[i].to_string())
                .arg("-c")
                .arg("copy")
                .arg(temp_seg.path())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .context(format!("提取片段{}失败", i + 1))?;

            if !extract.status.success() {
                extract_success = false;
                break;
            }
        }

        if !extract_success {
            drop(temp_segs);
            anyhow::bail!("提取多段预览片段失败");
        }

        // Create concat list file
        let concat_list = NamedTempFile::with_suffix(".txt").context("无法创建concat列表")?;
        let concat_content = temp_segs
            .iter()
            .map(|seg| format!("file '{}'\n", seg.path().display()))
            .collect::<String>();
        std::fs::write(concat_list.path(), concat_content).context("无法写入concat列表")?;

        // Use concat demuxer to merge and encode (with audio processing, same as full encode)
        let decode_args = build_decode_args(encoder_config);
        let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
        let mut encode_cmd = Command::new("ffmpeg");
        encode_cmd.arg("-y");
        for arg in &decode_args {
            encode_cmd.arg(arg);
        }
        encode_cmd
            .arg("-f")
            .arg("concat")
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(concat_list.path());
        for arg in video_args {
            encode_cmd.arg(arg);
        }
        encode_cmd.arg("-f").arg("mp4");

        // Audio processing (same as full encode)
        if audio_bitrate > 0 {
            encode_cmd
                .arg("-c:a")
                .arg("aac")
                .arg("-b:a")
                .arg(format!("{}k", audio_bitrate));
        } else {
            encode_cmd.arg("-c:a").arg("copy");
        }

        encode_cmd.arg(temp_output_path);

        let output = encode_cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg预览编码失败")?;

        drop(concat_list);
        drop(temp_segs);

        if !output.status.success() {
            anyhow::bail!("预览编码失败");
        }
    }

    // Get encoded size
    let output_size = std::fs::metadata(temp_output_path)
        .context("无法读取预览输出文件")?
        .len() as f64;

    drop(temp_output);

    // Return bytes per second for proportional estimation
    let output_size_per_second = output_size / duration as f64;

    Ok(PreviewResult {
        output_size_per_second,
    })
}

pub fn full_encode(
    input_path: &Path,
    output_path: &Path,
    crf: u8,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
) -> Result<u64> {
    let decode_args = build_decode_args(encoder_config);
    let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");
    for arg in decode_args {
        cmd.arg(arg);
    }
    cmd.arg("-i").arg(input_path);
    for arg in video_args {
        cmd.arg(arg);
    }

    // Audio processing
    if audio_bitrate > 0 {
        cmd.arg("-c:a")
            .arg("aac")
            .arg("-b:a")
            .arg(format!("{}k", audio_bitrate));
    } else {
        cmd.arg("-c:a").arg("copy");
    }

    cmd.arg(output_path);

    let output = cmd.output().context("执行FFmpeg失败")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg转换失败: {}", stderr);
    }

    Ok(std::fs::metadata(output_path)
        .context("无法读取输出文件")?
        .len())
}
