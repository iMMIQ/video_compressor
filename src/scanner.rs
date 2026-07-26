use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const VIDEO_EXTENSIONS: &[&str] = &[
    "3g2",
    "3gp",
    "asf",
    "avi",
    "divx",
    "f4v",
    "flv",
    "m2ts",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mp4v",
    "mpe",
    "mpeg",
    "mpg",
    "mts",
    "ogv",
    "rm",
    "rmvb",
    "tp",
    "trp",
    "ts",
    "vob",
    "webm",
    "wmv",
];

// NOTE: "webp" is intentionally NOT included. Existing WebP files (including
// animated ugoira webp) are left untouched — re-encoding them would destroy
// animations and/or needlessly degrade already-compressed images.
// "gif" is included: GIFs are treated as images (converted to single-frame
// WebP). The image path guards against animated GIFs (skips them rather than
// flattening to frame 0), and was previously misrouted to the ffmpeg video
// path where it always failed.
const IMAGE_EXTENSIONS: &[&str] = &[
    "bmp",
    "gif",
    "heic",
    "heif",
    "jpg",
    "jpeg",
    "png",
    "tiff",
    "tif",
];

#[inline]
fn matches_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| extensions.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Stream scan video files (callback-based for memory efficiency)
pub fn scan_videos_streaming<F>(dir: &Path, mut callback: F)
where
    F: FnMut(PathBuf),
{
    if dir.is_file() {
        if matches_extension(dir, VIDEO_EXTENSIONS) {
            callback(dir.to_path_buf());
        }
        return;
    }

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter(|e| matches_extension(e.path(), VIDEO_EXTENSIONS))
    {
        callback(entry.path().to_path_buf());
    }
}

/// Stream scan image files (callback-based for memory efficiency)
pub fn scan_images_streaming<F>(dir: &Path, mut callback: F)
where
    F: FnMut(PathBuf),
{
    if dir.is_file() {
        if matches_extension(dir, IMAGE_EXTENSIONS) {
            callback(dir.to_path_buf());
        }
        return;
    }

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter(|e| matches_extension(e.path(), IMAGE_EXTENSIONS))
    {
        callback(entry.path().to_path_buf());
    }
}
