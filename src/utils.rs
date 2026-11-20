use anyhow::{Context, Result};
use filetime::{FileTime, set_file_times};
use std::fs::Metadata;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

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

/// Generate output path for image: same directory as input, with .webp extension
pub fn get_image_output_path(input_path: &Path) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or(std::ffi::OsStr::new(""));
    let parent = input_path.parent().unwrap_or(Path::new(""));
    parent.join(format!("{}.webp", stem.to_string_lossy()))
}

/// Extract modification time in Unix seconds for state caching.
pub fn file_mtime_secs(metadata: &Metadata) -> i64 {
    FileTime::from_last_modification_time(metadata).unix_seconds()
}

/// Create a temporary file in the same directory as `target_path` so that
/// persisting it to `target_path` is an atomic same-filesystem rename.
pub fn create_temp_in_same_dir(target_path: &Path, suffix: &str) -> Result<NamedTempFile> {
    let parent = target_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    NamedTempFile::with_suffix_in(suffix, parent)
        .with_context(|| format!("无法在 {} 创建临时文件", parent.display()))
}

/// Crash-safe file replacement:
/// 1. Refuses to clobber a pre-existing distinct target file.
/// 2. Atomically persists `temp_file` (created in the same directory) to `output_path`.
/// 3. Only AFTER `output_path` is durable, deletes `input_path` (when its name differs).
/// 4. Restores original atime, mtime, and permissions.
pub fn atomic_replace_from_temp(
    temp_file: NamedTempFile,
    input_path: &Path,
    output_path: &Path,
    original_metadata: &Metadata,
) -> Result<()> {
    if output_path.exists() && output_path != input_path {
        anyhow::bail!("目标已存在，拒绝覆盖: {}", output_path.display());
    }

    let original_atime = FileTime::from_last_access_time(original_metadata);
    let original_mtime = FileTime::from_last_modification_time(original_metadata);
    let original_permissions = original_metadata.permissions();

    temp_file
        .persist(output_path)
        .map_err(|e| anyhow::anyhow!("无法原子替换输出文件 {}: {}", output_path.display(), e.error))?;

    if output_path != input_path {
        std::fs::remove_file(input_path)
            .with_context(|| format!("新文件已就位，但删除原文件失败: {}", input_path.display()))?;
    }

    set_file_times(output_path, original_atime, original_mtime)
        .context("无法设置文件时间戳")?;
    std::fs::set_permissions(output_path, original_permissions)
        .context("无法设置文件权限")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn format_size_units() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(2048), "2.00 KB");
        assert_eq!(format_size(3 * 1024 * 1024), "3.00 MB");
        assert_eq!(format_size(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn output_paths_replace_extension() {
        assert_eq!(
            get_output_path(Path::new("/a/b/movie.avi")),
            PathBuf::from("/a/b/movie.mp4")
        );
        assert_eq!(
            get_image_output_path(Path::new("/a/b/photo.JPEG")),
            PathBuf::from("/a/b/photo.webp")
        );
    }

    #[test]
    fn atomic_replace_preserves_times_and_removes_distinct_source() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("clip.avi");
        let dst = dir.path().join("clip.mp4");
        std::fs::write(&src, b"original-avi-bytes").unwrap();

        let atime = FileTime::from_unix_time(1_700_000_001, 0);
        let mtime = FileTime::from_unix_time(1_700_000_002, 0);
        set_file_times(&src, atime, mtime).unwrap();
        let meta = std::fs::metadata(&src).unwrap();

        let mut tmp = create_temp_in_same_dir(&dst, ".mp4").unwrap();
        tmp.write_all(b"new-mp4").unwrap();
        tmp.flush().unwrap();

        atomic_replace_from_temp(tmp, &src, &dst, &meta).unwrap();

        assert!(!src.exists());
        let after = std::fs::metadata(&dst).unwrap();
        assert_eq!(FileTime::from_last_access_time(&after), atime);
        assert_eq!(FileTime::from_last_modification_time(&after), mtime);
        assert_eq!(std::fs::read(&dst).unwrap(), b"new-mp4");
    }

    #[test]
    fn atomic_replace_refuses_to_clobber_existing_target() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("clip.avi");
        let dst = dir.path().join("clip.mp4");
        std::fs::write(&src, b"avi").unwrap();
        std::fs::write(&dst, b"existing-mp4").unwrap();
        let meta = std::fs::metadata(&src).unwrap();

        let mut tmp = create_temp_in_same_dir(&dst, ".mp4").unwrap();
        tmp.write_all(b"new-mp4").unwrap();

        let err = atomic_replace_from_temp(tmp, &src, &dst, &meta).unwrap_err();
        assert!(err.to_string().contains("拒绝覆盖"));
        assert_eq!(std::fs::read(&src).unwrap(), b"avi");
        assert_eq!(std::fs::read(&dst).unwrap(), b"existing-mp4");
    }
}
