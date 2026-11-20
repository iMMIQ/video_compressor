use crate::model::{FileStats, ProcessResult};
use crate::signal::is_shutdown_requested;
use crate::state::StateCache;
use crate::utils::{
    atomic_replace_from_temp, create_temp_in_same_dir, file_mtime_secs, format_size,
    get_image_output_path,
};
use crate::webp_mux;
use anyhow::{Context, Result};
use std::io::Write;
use std::path::Path;

/// Minimum size reduction (percent) required to actually replace the original.
/// Below this the re-encode is considered not worth the CPU/risk.
const MIN_GAIN_PERCENT: f64 = 5.0;

/// Extract raw EXIF bytes from an image file (JPEG only).
fn extract_exif(data: &[u8]) -> Option<Vec<u8>> {
    let mut reader = std::io::Cursor::new(data);
    let exif_reader = exif::Reader::new();
    match exif_reader.read_from_container(&mut reader) {
        Ok(_) => extract_raw_exif(data),
        Err(_) => None,
    }
}

/// Extract raw EXIF bytes from JPEG APP1 segment.
/// Only JPEG is supported; other formats simply yield no EXIF.
fn extract_raw_exif(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 4 {
        return None;
    }

    if data[0] == 0xFF && data[1] == 0xD8 {
        let mut pos = 2;
        while pos + 4 <= data.len() {
            if data[pos] != 0xFF {
                break;
            }
            let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
            if marker == 0xFFE1 {
                let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                if pos + 2 + seg_len <= data.len() {
                    let seg_data = &data[pos + 4..pos + 2 + seg_len];
                    if seg_data.starts_with(b"Exif\x00\x00") {
                        return Some(seg_data[6..].to_vec());
                    }
                }
            }
            let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            if seg_len < 2 {
                break;
            }
            pos += 2 + seg_len;
            while pos < data.len() && data[pos] == 0xFF {
                pos += 1;
            }
        }
    }

    None
}

/// Whether the path is a GIF (case-insensitive extension).
#[inline]
fn is_gif(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gif"))
        .unwrap_or(false)
}

pub fn process_image(input_path: &Path, state_cache: Option<&StateCache>) -> Result<ProcessResult> {
    if is_shutdown_requested() {
        anyhow::bail!("收到中断信号，跳过处理");
    }

    let output_path = get_image_output_path(input_path);

    // Guard against clobbering a pre-existing target file BEFORE reading/decoding.
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

    if let Some(cache) = state_cache
        && let Some(reason) = cache.get_cached_skip(input_path, input_size, mtime_secs)
    {
        return Ok(ProcessResult::Skipped(format!("{} (缓存)", reason)));
    }

    let record_skip = |reason: String| -> Result<ProcessResult> {
        if let Some(cache) = state_cache {
            cache.record_permanent_skip(input_path, input_size, mtime_secs, &reason);
        }
        Ok(ProcessResult::Skipped(reason))
    };

    // Read file once; reuse for both EXIF extraction and decoding (avoids a
    // second read over slow NFS).
    let file_data = std::fs::read(input_path).context("无法读取图片文件")?;
    let raw_exif = extract_exif(&file_data);

    // Guard: single-frame encoder would flatten animated GIFs. Skip multi-frame GIFs.
    if is_gif(input_path)
        && let Ok(dec) = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(&file_data))
    {
        use image::AnimationDecoder;
        if let Ok(frames) = dec.into_frames().collect_frames()
            && frames.len() > 1
        {
            return record_skip(format!(
                "动画GIF({}帧)需多帧管线，跳过保留原文件",
                frames.len()
            ));
        }
    }

    let img = image::load_from_memory(&file_data).context("无法解码图片")?;
    let (width, height) = (img.width(), img.height());

    // WebP has a hard per-dimension cap of 16383px.
    const WEBP_MAX_DIM: u32 = 16383;
    if width > WEBP_MAX_DIM || height > WEBP_MAX_DIM {
        return record_skip(format!(
            "超过WebP尺寸上限 ({}x{} > {}px)，跳过保留原文件",
            width, height, WEBP_MAX_DIM
        ));
    }

    let mut config = webp::WebPConfig::new_with_preset(
        libwebp_sys::WebPPreset::WEBP_PRESET_DEFAULT,
        90.0,
    )
    .map_err(|_| anyhow::anyhow!("无法创建WebP编码配置"))?;
    config.lossless = 0;
    config.quality = 90.0;
    config.alpha_quality = 90;

    let has_alpha = matches!(
        img.color(),
        image::ColorType::La8
            | image::ColorType::Rgba8
            | image::ColorType::La16
            | image::ColorType::Rgba16
    );
    let webp_memory = if has_alpha {
        let rgba = img.to_rgba8();
        webp::Encoder::from_rgba(&rgba, width, height)
            .encode_advanced(&config)
            .map_err(|e| anyhow::anyhow!("WebP编码失败: {:?}", e))?
    } else {
        let rgb = img.to_rgb8();
        webp::Encoder::from_rgb(&rgb, width, height)
            .encode_advanced(&config)
            .map_err(|e| anyhow::anyhow!("WebP编码失败: {:?}", e))?
    };

    let mut webp_data = webp_memory.to_vec();

    // Inject EXIF metadata through libwebp's canonical VP8X muxer.
    if let Some(exif) = raw_exif {
        webp_data = webp_mux::add_exif(&webp_data, &exif)?;
    }

    let output_size = webp_data.len() as u64;
    let gain_percent = if input_size > 0 {
        (1.0 - output_size as f64 / input_size as f64) * 100.0
    } else {
        0.0
    };

    if output_size >= input_size {
        return record_skip(format!(
            "WebP更大 ({} >= {})",
            format_size(output_size),
            format_size(input_size)
        ));
    }

    if gain_percent < MIN_GAIN_PERCENT {
        return record_skip(format!(
            "收益过小 ({:.1}% < {:.0}%)",
            gain_percent, MIN_GAIN_PERCENT
        ));
    }

    let mut temp_file = create_temp_in_same_dir(&output_path, ".webp")?;
    temp_file
        .write_all(&webp_data)
        .context("无法写入WebP文件")?;
    temp_file.flush().context("无法刷新临时文件")?;

    atomic_replace_from_temp(temp_file, input_path, &output_path, &original_metadata)?;

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: 0,
    }))
}
