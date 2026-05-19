use crate::utils::{format_size, get_image_output_path, safe_move_file};
use crate::video::{FileStats, ProcessResult};
use anyhow::{Context, Result};
use filetime::{set_file_times, FileTime};
use std::io::Write;
use std::path::Path;

/// Extract raw EXIF bytes from an image file
fn extract_exif(data: &[u8]) -> Option<Vec<u8>> {
    let mut reader = std::io::Cursor::new(data);
    let exif_reader = exif::Reader::new();
    match exif_reader.read_from_container(&mut reader) {
        Ok(_exif_data) => {
            // kamadak-exif parses EXIF but doesn't give raw bytes directly.
            // Extract raw EXIF from JPEG/TIFF manually.
            extract_raw_exif(data)
        }
        Err(_) => None,
    }
}

/// Extract raw EXIF bytes from JPEG or TIFF file data
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
        return None;
    }

    // TIFF/PNG/etc: use kamadak-exif's InboundData to get raw TIFF/EXIF
    // For TIFF files, the entire file IS the EXIF data structure
    let little_endian = data.len() >= 2 && data[0] == b'I' && data[1] == b'I';
    let big_endian = data.len() >= 2 && data[0] == b'M' && data[1] == b'M';
    if little_endian || big_endian {
        return Some(data.to_vec());
    }

    None
}

/// Inject EXIF data into a WebP file's RIFF container
fn inject_exif_into_webp(webp_data: &[u8], exif_data: &[u8]) -> Vec<u8> {
    // WebP RIFF structure:
    //   RIFF <file_size:u32LE> WEBP <chunks...>
    // Each chunk: <fourcc:4bytes> <chunk_size:u32LE> <data...> [padding]
    if webp_data.len() < 12 || &webp_data[0..4] != b"RIFF" || &webp_data[8..12] != b"WEBP" {
        return webp_data.to_vec();
    }

    // Build EXIF chunk: "EXIF" + size (LE) + data (padded to even)
    let exif_chunk_data = exif_data;
    let exif_chunk_size = exif_chunk_data.len() as u32;
    let padding = if exif_chunk_data.len() % 2 == 1 {
        vec![0u8]
    } else {
        vec![]
    };

    let mut exif_chunk = Vec::with_capacity(8 + exif_chunk_data.len() + padding.len());
    exif_chunk.extend_from_slice(b"EXIF");
    exif_chunk.extend_from_slice(&exif_chunk_size.to_le_bytes());
    exif_chunk.extend_from_slice(exif_chunk_data);
    exif_chunk.extend_from_slice(&padding);

    // Insert EXIF chunk after the WEBP header (at offset 12), before VP8/VP8L chunk
    let mut result = Vec::with_capacity(webp_data.len() + exif_chunk.len());
    result.extend_from_slice(&webp_data[..12]); // "RIFF" + size + "WEBP"
    result.extend_from_slice(&exif_chunk);
    result.extend_from_slice(&webp_data[12..]);

    // Update RIFF file size
    let new_size = (result.len() - 8) as u32;
    result[4..8].copy_from_slice(&new_size.to_le_bytes());

    result
}

pub fn process_image(input_path: &Path) -> Result<ProcessResult> {
    println!("处理图片: {}", input_path.display());

    // Read original file metadata
    let original_metadata =
        std::fs::metadata(input_path).context("无法读取输入文件元数据")?;
    let original_atime = FileTime::from_last_access_time(&original_metadata);
    let original_mtime = FileTime::from_last_modification_time(&original_metadata);
    let original_permissions = original_metadata.permissions();
    let input_size = original_metadata.len();

    // Read file data for EXIF extraction
    let file_data = std::fs::read(input_path).context("无法读取图片文件")?;
    let raw_exif = extract_exif(&file_data);

    // Decode image using image crate
    let img = image::open(input_path).context("无法解码图片")?;
    let (width, height) = (img.width(), img.height());

    // Convert to RGBA for webp encoding
    let rgba = img.to_rgba8();
    let raw_pixels = rgba.as_raw();

    // Encode to WebP lossy quality 90 using webp crate
    let encoder = webp::Encoder::from_rgba(raw_pixels, width, height);
    let mut config = webp::WebPConfig::new_with_preset(libwebp_sys::WebPPreset::WEBP_PRESET_DEFAULT, 90.0)
        .map_err(|_| anyhow::anyhow!("无法创建WebP编码配置"))?;
    config.lossless = 0;
    config.quality = 90.0;
    config.alpha_quality = 90;
    let webp_memory = encoder
        .encode_advanced(&config)
        .map_err(|e| anyhow::anyhow!("WebP编码失败: {:?}", e))?;

    let mut webp_data = webp_memory.to_vec();

    // Inject EXIF metadata if present
    if let Some(exif) = raw_exif {
        webp_data = inject_exif_into_webp(&webp_data, &exif);
    }

    let output_size = webp_data.len() as u64;

    println!(
        "  {} -> {} ({:.1}%)",
        format_size(input_size),
        format_size(output_size),
        (1.0 - output_size as f64 / input_size as f64) * 100.0
    );

    // Skip if output is larger or equal
    if output_size >= input_size {
        return Ok(ProcessResult::Skipped(format!(
            "WebP更大 ({} > {})",
            format_size(output_size),
            format_size(input_size)
        )));
    }

    // Write to temp file, then atomically replace
    let output_path = get_image_output_path(input_path);
    let mut temp_file = tempfile::NamedTempFile::with_suffix(".webp").context("无法创建临时文件")?;
    temp_file
        .write_all(&webp_data)
        .context("无法写入WebP文件")?;
    temp_file.flush().context("无法刷新临时文件")?;

    let temp_path = temp_file.path().to_path_buf();

    // Delete original
    std::fs::remove_file(input_path).context("无法删除原文件")?;

    // Move temp to output
    safe_move_file(&temp_path, &output_path).context("无法移动输出文件")?;

    // Restore timestamps and permissions
    set_file_times(&output_path, original_atime, original_mtime)
        .context("无法设置文件时间戳")?;
    std::fs::set_permissions(&output_path, original_permissions)
        .context("无法设置文件权限")?;

    // Consume temp_file to avoid cleanup
    let _ = temp_file;

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: 0,
    }))
}
