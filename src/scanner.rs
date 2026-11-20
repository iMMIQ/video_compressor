use crate::model::MediaItem;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const VIDEO_EXTENSIONS: &[&str] = &[
    "3g2", "3gp", "asf", "avi", "divx", "f4v", "flv", "m2ts", "m4v", "mkv", "mov", "mp4", "mp4v",
    "mpe", "mpeg", "mpg", "mts", "ogv", "rm", "rmvb", "tp", "trp", "ts", "vob", "webm", "wmv",
];

// NOTE:
// - "webp" is intentionally NOT included (avoid re-encoding or flattening animated WebP).
// - "heic"/"heif" are intentionally NOT included because the `image` crate does not decode them.
// - "gif" is included for static GIFs; animated GIFs are detected and skipped in `image.rs`.
const IMAGE_EXTENSIONS: &[&str] = &["bmp", "gif", "jpg", "jpeg", "png", "tiff", "tif"];

const WEBP_EXTENSIONS: &[&str] = &["webp"];

#[inline]
fn matches_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let lower = ext.to_ascii_lowercase();
            extensions.contains(&lower.as_str())
        })
        .unwrap_or(false)
}

#[inline]
fn classify_path(path: &Path) -> Option<MediaItem> {
    if matches_extension(path, VIDEO_EXTENSIONS) {
        Some(MediaItem::Video(path.to_path_buf()))
    } else if matches_extension(path, IMAGE_EXTENSIONS) {
        Some(MediaItem::Image(path.to_path_buf()))
    } else {
        None
    }
}

/// Single-pass streaming media scanner: discovers both videos and images in one
/// `WalkDir` traversal and classifies them into `MediaItem`.
/// Uses `DirEntry::file_type().is_file()` to avoid extra `stat` RPCs on NFS.
pub fn scan_media_streaming<F>(dir: &Path, mut callback: F)
where
    F: FnMut(MediaItem),
{
    if dir.is_file() {
        if let Some(item) = classify_path(dir) {
            callback(item);
        }
        return;
    }

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        if let Some(item) = classify_path(entry.path()) {
            callback(item);
        }
    }
}

/// Stream scan WebP files for the one-time container repair mode.
pub fn scan_webp_streaming<F>(dir: &Path, mut callback: F)
where
    F: FnMut(PathBuf),
{
    if dir.is_file() {
        if matches_extension(dir, WEBP_EXTENSIONS) {
            callback(dir.to_path_buf());
        }
        return;
    }

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| matches_extension(e.path(), WEBP_EXTENSIONS))
        .filter(|e| e.file_type().is_file())
    {
        callback(entry.path().to_path_buf());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_pass_scan_classifies_media_and_ignores_unsupported() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.MP4"), b"").unwrap();
        std::fs::write(dir.path().join("b.jpg"), b"").unwrap();
        std::fs::write(dir.path().join("c.webp"), b"").unwrap();
        std::fs::write(dir.path().join("d.heic"), b"").unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"").unwrap();

        let mut videos = 0;
        let mut images = 0;
        scan_media_streaming(dir.path(), |item| match item {
            MediaItem::Video(_) => videos += 1,
            MediaItem::Image(_) => images += 1,
        });
        assert_eq!(videos, 1);
        assert_eq!(images, 1);

        let mut webps = 0;
        scan_webp_streaming(dir.path(), |_| webps += 1);
        assert_eq!(webps, 1);
    }
}
