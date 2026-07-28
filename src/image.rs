use crate::utils::{format_size, get_image_output_path};
use crate::video::{FileStats, ProcessResult};
use crate::webp_mux;
use anyhow::{Context, Result};
use filetime::{set_file_times, FileTime};
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
/// NOTE: only JPEG is supported. The previous implementation returned the ENTIRE
/// file bytes for TIFF input (treating the whole TIFF as EXIF), which bloated
/// the output WebP with garbage — that branch is removed; other formats simply
/// yield no EXIF.
fn extract_raw_exif(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 4 {
        return None;
    }

    // JPEG: look for APP1 marker (0xFFE1)
    if data[0] == 0xFF && data[1] == 0xD8 {
        let mut pos = 2;
        while pos + 4 <= data.len() {
            if data[pos] != 0xFF {
                break;
            }
            let marker = u16::from_be_bytes([data[pos], data[pos + 1]]);
            if marker == 0xFFE1 {
                // APP1 segment
                let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                if pos + 2 + seg_len <= data.len() {
                    let seg_data = &data[pos + 4..pos + 2 + seg_len];
                    // Check for Exif header ("Exif\0\0")
                    if seg_data.starts_with(b"Exif\x00\x00") {
                        return Some(seg_data[6..].to_vec());
                    }
                }
            }
            let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            pos += 2 + seg_len;
            // Skip padding bytes
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

pub fn process_image(input_path: &Path) -> Result<ProcessResult> {
    // Read original file metadata
    let original_metadata =
        std::fs::metadata(input_path).context("无法读取输入文件元数据")?;
    let original_atime = FileTime::from_last_access_time(&original_metadata);
    let original_mtime = FileTime::from_last_modification_time(&original_metadata);
    let original_permissions = original_metadata.permissions();
    let input_size = original_metadata.len();

    // Read file once; reuse for both EXIF extraction and decoding (avoids a
    // second read over slow NFS).
    let file_data = std::fs::read(input_path).context("无法读取图片文件")?;
    let raw_exif = extract_exif(&file_data);

    // Guard: the encoder below is single-frame, so an animated GIF would be
    // silently flattened to its first frame (animation lost). Detect multi-
    // frame GIFs and skip them — they are preserved untouched, never destroyed.
    // Static (single-frame) GIFs proceed normally as ordinary images.
    if is_gif(input_path) {
        if let Ok(dec) =
            image::codecs::gif::GifDecoder::new(std::io::Cursor::new(&file_data))
        {
            use image::AnimationDecoder;
            if let Ok(frames) = dec.into_frames().collect_frames() {
                if frames.len() > 1 {
                    return Ok(ProcessResult::Skipped(format!(
                        "动画GIF({}帧)需多帧管线，跳过保留原文件",
                        frames.len()
                    )));
                }
            }
        }
    }

    // Decode image from the in-memory buffer
    let img = image::load_from_memory(&file_data).context("无法解码图片")?;
    let (width, height) = (img.width(), img.height());

    // WebP has a hard per-dimension cap of 16383px. Encoding beyond it fails
    // with VP8_ENC_ERROR_BAD_DIMENSION — skip up front (original preserved)
    // instead of logging a failure.
    const WEBP_MAX_DIM: u32 = 16383;
    if width > WEBP_MAX_DIM || height > WEBP_MAX_DIM {
        return Ok(ProcessResult::Skipped(format!(
            "超过WebP尺寸上限 ({}x{} > {}px)，跳过保留原文件",
            width, height, WEBP_MAX_DIM
        )));
    }

    // Lossy q90 WebP encoding config
    let mut config = webp::WebPConfig::new_with_preset(
        libwebp_sys::WebPPreset::WEBP_PRESET_DEFAULT,
        90.0,
    )
    .map_err(|_| anyhow::anyhow!("无法创建WebP编码配置"))?;
    config.lossless = 0;
    config.quality = 90.0;
    config.alpha_quality = 90;

    // Encode via RGB path when the source has no alpha → smaller output
    // (no useless fully-opaque alpha plane, which matters for opaque JPEGs).
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

    // Inject EXIF metadata if present
    if let Some(exif) = raw_exif {
        webp_data = webp_mux::add_exif(&webp_data, &exif)?;
    }

    let output_size = webp_data.len() as u64;
    let gain_percent = (1.0 - output_size as f64 / input_size as f64) * 100.0;

    // Skip if output is not smaller
    if output_size >= input_size {
        return Ok(ProcessResult::Skipped(format!(
            "WebP更大 ({} > {})",
            format_size(output_size),
            format_size(input_size)
        )));
    }

    // Skip if the gain is negligible — not worth a re-encode
    if gain_percent < MIN_GAIN_PERCENT {
        return Ok(ProcessResult::Skipped(format!(
            "收益过小 ({:.1}% < {:.0}%)",
            gain_percent, MIN_GAIN_PERCENT
        )));
    }

    let output_path = get_image_output_path(input_path);

    // Guard against clobbering a pre-existing target file
    // (e.g. both X.jpg and X.webp present — converting the jpg would overwrite
    // the existing webp via the atomic rename below).
    if output_path.exists() && output_path != input_path {
        return Ok(ProcessResult::Skipped(format!(
            "目标已存在，跳过避免覆盖: {}",
            output_path.display()
        )));
    }

    // Crash-safe replacement:
    // 1. write the new WebP to a temp file IN THE SAME directory as the output
    //    (same filesystem → the move is an atomic rename, never a copy);
    // 2. atomically persist temp -> output_path;
    // 3. only AFTER the new file is durable, delete the original.
    // The original is never removed before its replacement exists on disk.
    let parent = output_path.parent().unwrap_or(Path::new(""));
    let mut temp_file = tempfile::NamedTempFile::with_suffix_in(".webp", parent)
        .context("无法创建临时文件")?;
    temp_file
        .write_all(&webp_data)
        .context("无法写入WebP文件")?;
    temp_file.flush().context("无法刷新临时文件")?;

    temp_file
        .persist(&output_path)
        .map_err(|e| anyhow::anyhow!("无法移动输出文件: {}", e.error))?;

    // New file is durable — now safe to remove the original (different name)
    if output_path != input_path {
        let _ = std::fs::remove_file(input_path);
    }

    // Restore timestamps and permissions
    set_file_times(&output_path, original_atime, original_mtime)
        .context("无法设置文件时间戳")?;
    std::fs::set_permissions(&output_path, original_permissions)
        .context("无法设置文件权限")?;

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: 0,
    }))
}
