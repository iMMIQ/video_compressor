use crate::model::{FileStats, ProcessResult, savings_percent};
use crate::state::StateCache;
use crate::utils::{
    atomic_replace_from_temp, create_temp_in_same_dir, format_size, get_image_output_path,
    preflight_media_file,
};
use crate::webp_mux;
use anyhow::{Context, Result};
use std::io::Write;
use std::path::Path;

/// Minimum size reduction (percent) required to actually replace the original.
const MIN_GAIN_PERCENT: f64 = 5.0;

/// WebP has a hard per-dimension cap of 16383px.
const WEBP_MAX_DIM: u32 = 16383;

/// Extract raw EXIF TIFF bytes from JPEG/PNG containers using `kamadak-exif`.
/// TIFF files are intentionally excluded because `kamadak-exif` returns the entire
/// TIFF file (including uncompressed pixel strips) as `exif.buf()`.
fn extract_exif(data: &[u8]) -> Option<Vec<u8>> {
    let is_jpeg = data.starts_with(&[0xFF, 0xD8]);
    let is_png = data.starts_with(b"\x89PNG\r\n\x1a\n");
    if !is_jpeg && !is_png {
        return None;
    }
    exif::Reader::new()
        .read_from_container(&mut std::io::Cursor::new(data))
        .ok()
        .map(|exif| exif.buf().to_vec())
}

#[inline]
fn is_gif(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gif"))
}

pub fn process_image(input_path: &Path, state_cache: Option<&StateCache>) -> Result<ProcessResult> {
    let output_path = get_image_output_path(input_path);
    let preflight = match preflight_media_file(input_path, &output_path, state_cache, false)? {
        Ok(info) => info,
        Err(skipped) => return Ok(skipped),
    };
    let input_size = preflight.input_size;

    // Read file once; reuse for EXIF extraction, GIF check, and decoding (avoids second NFS read).
    let file_data = std::fs::read(input_path).context("无法读取图片文件")?;
    let raw_exif = extract_exif(&file_data);

    // Guard: single-frame encoder would flatten animated GIFs. Peek at most 2 frames (O(1) RAM).
    if is_gif(input_path)
        && let Ok(dec) = image::codecs::gif::GifDecoder::new(std::io::Cursor::new(&file_data))
    {
        use image::AnimationDecoder;
        if dec.into_frames().take(2).count() > 1 {
            return preflight.record_skip(
                input_path,
                state_cache,
                "动画GIF需多帧管线，跳过保留原文件".to_string(),
                true,
            );
        }
    }

    let img = image::load_from_memory(&file_data).context("无法解码图片")?;
    let (width, height) = (img.width(), img.height());

    if width > WEBP_MAX_DIM || height > WEBP_MAX_DIM {
        return preflight.record_skip(
            input_path,
            state_cache,
            format!(
                "超过WebP尺寸上限 ({}x{} > {}px)，跳过保留原文件",
                width, height, WEBP_MAX_DIM
            ),
            true,
        );
    }

    let mut config =
        webp::WebPConfig::new_with_preset(libwebp_sys::WebPPreset::WEBP_PRESET_DEFAULT, 90.0)
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
    if let Some(exif) = raw_exif {
        webp_data = webp_mux::add_exif(&webp_data, &exif)?;
    }

    let output_size = webp_data.len() as u64;
    if output_size >= input_size {
        return preflight.record_skip(
            input_path,
            state_cache,
            format!(
                "WebP更大 ({} >= {})",
                format_size(output_size),
                format_size(input_size)
            ),
            true,
        );
    }

    let gain_percent = savings_percent(input_size, output_size);
    if gain_percent < MIN_GAIN_PERCENT {
        return preflight.record_skip(
            input_path,
            state_cache,
            format!("收益过小 ({:.1}% < {:.0}%)", gain_percent, MIN_GAIN_PERCENT),
            true,
        );
    }

    let mut temp_file = create_temp_in_same_dir(&output_path, ".webp")?;
    temp_file
        .write_all(&webp_data)
        .context("无法写入WebP文件")?;
    temp_file.flush().context("无法刷新临时文件")?;

    atomic_replace_from_temp(temp_file, input_path, &output_path, &preflight.metadata)?;

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_exif_reads_jpeg_app1_and_skips_raw_tiff() {
        // Minimal valid little-endian TIFF header with 1 IFD entry (Orientation = 1).
        let tiff_payload: &[u8] = &[
            b'I', b'I', 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // TIFF header -> IFD0 at 8
            0x01, 0x00, // 1 entry
            0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
            0x00, // Orientation=1
            0x00, 0x00, 0x00, 0x00, // Next IFD = 0
        ];
        let seg_len = (2 + 6 + tiff_payload.len()) as u16;
        let mut jpeg = Vec::new();
        jpeg.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE1]);
        jpeg.extend_from_slice(&seg_len.to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(tiff_payload);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        assert_eq!(extract_exif(&jpeg).as_deref(), Some(tiff_payload));
        assert!(extract_exif(tiff_payload).is_none());
    }
}
