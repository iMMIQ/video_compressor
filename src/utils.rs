use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Format bytes to human readable string
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Generate output path: same directory as input, with .mp4 extension
pub fn get_output_path(input_path: &Path) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or(std::ffi::OsStr::new(""));
    let parent = input_path.parent().unwrap_or(Path::new(""));
    parent.join(format!("{}.mp4", stem.to_string_lossy()))
}

/// Cross-filesystem safe file move
/// Try rename first, fallback to copy+remove
pub fn safe_move_file(from: &Path, to: &Path) -> Result<()> {
    // Try direct rename first (fastest for same filesystem)
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }

    // rename failed (possibly cross-filesystem), use copy+remove
    std::fs::copy(from, to)
        .context(format!("无法复制文件从 {} 到 {}", from.display(), to.display()))?;
    std::fs::remove_file(from)
        .context(format!("无法删除源文件 {}", from.display()))?;

    Ok(())
}
