use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Stream scan video files (callback-based for memory efficiency)
pub fn scan_videos_streaming<F>(dir: &Path, mut callback: F)
where
    F: FnMut(PathBuf),
{
    let video_extensions = [
        "mp4", "mkv", "avi", "mov", "flv", "wmv", "webm", "m4v", "mpg", "mpeg", "3gp",
    ];

    // If input is a single file, process it directly
    if dir.is_file() {
        let is_video = dir
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| video_extensions.contains(&ext.to_lowercase().as_str()))
            .unwrap_or(false);
        if is_video {
            callback(dir.to_path_buf());
        }
        return;
    }

    // Use min_depth(1) to avoid processing root directory itself
    let mut entries: Vec<_> = WalkDir::new(dir)
        .follow_links(true)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| video_extensions.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    // Sort for consistent processing order
    entries.sort();

    for path in entries {
        callback(path);
    }
}
