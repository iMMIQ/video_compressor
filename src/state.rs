use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CachedSkip {
    size: u64,
    mtime_secs: i64,
    fingerprint: String,
    reason: String,
}

/// Optional persistent cache of skipped files so repeated runs on large
/// directories avoid redundant ffprobe / perceptual search work on unchanged files.
/// Entries are scoped by `config_fingerprint` so changing encoder, target XPSNR,
/// or minimum saving ratio automatically invalidates stale video skip decisions.
pub struct StateCache {
    entries: Mutex<HashMap<PathBuf, CachedSkip>>,
    writer: Mutex<BufWriter<File>>,
    fingerprint: String,
}

impl StateCache {
    /// Open or create a state file at `path`, scoped to `fingerprint`.
    pub fn open_with_fingerprint(path: &Path, fingerprint: impl Into<String>) -> Result<Self> {
        let mut map = HashMap::new();

        if path.exists() {
            let file = File::open(path)
                .with_context(|| format!("无法打开状态缓存文件: {}", path.display()))?;
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                if let Some((file_path, entry)) = parse_line(&line) {
                    map.insert(file_path, entry);
                }
            }
        }

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("无法创建状态缓存目录: {}", parent.display()))?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("无法写入状态缓存文件: {}", path.display()))?;

        Ok(Self {
            entries: Mutex::new(map),
            writer: Mutex::new(BufWriter::new(file)),
            fingerprint: fingerprint.into(),
        })
    }

    /// Number of loaded entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if `path` with `(size, mtime_secs)` was previously skipped under the current
    /// config fingerprint (or for fingerprint-independent reasons like images / already-HEVC).
    pub fn get_cached_skip(&self, path: &Path, size: u64, mtime_secs: i64) -> Option<String> {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries
            .get(path)
            .filter(|e| {
                e.size == size
                    && e.mtime_secs == mtime_secs
                    && (e.fingerprint == "*" || e.fingerprint == self.fingerprint)
            })
            .map(|e| e.reason.clone())
    }

    /// Record a skipped file scoped to the current config fingerprint.
    pub fn record_skip(&self, path: &Path, size: u64, mtime_secs: i64, reason: &str) {
        self.record_skip_with_scope(path, size, mtime_secs, &self.fingerprint, reason);
    }

    /// Record a skipped file valid across all video config fingerprints (e.g. already HEVC/AV1 or images).
    pub fn record_permanent_skip(&self, path: &Path, size: u64, mtime_secs: i64, reason: &str) {
        self.record_skip_with_scope(path, size, mtime_secs, "*", reason);
    }

    fn record_skip_with_scope(
        &self,
        path: &Path,
        size: u64,
        mtime_secs: i64,
        scope: &str,
        reason: &str,
    ) {
        {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            entries.insert(
                path.to_path_buf(),
                CachedSkip {
                    size,
                    mtime_secs,
                    fingerprint: scope.to_string(),
                    reason: reason.to_string(),
                },
            );
        }

        let mut writer = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let line = format_line(path, size, mtime_secs, scope, reason);
        if writeln!(writer, "{}", line).is_ok() {
            let _ = writer.flush();
        }
    }
}

fn escape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out
}

fn unescape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn format_line(path: &Path, size: u64, mtime_secs: i64, fingerprint: &str, reason: &str) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}",
        escape_field(&path.to_string_lossy()),
        size,
        mtime_secs,
        escape_field(fingerprint),
        escape_field(reason)
    )
}

fn parse_line(line: &str) -> Option<(PathBuf, CachedSkip)> {
    let mut parts = line.splitn(5, '\t');
    let path_str = parts.next()?;
    let size: u64 = parts.next()?.parse().ok()?;
    let mtime_secs: i64 = parts.next()?.parse().ok()?;
    let fourth = parts.next()?;
    let (fingerprint, reason_str) = match parts.next() {
        Some(fifth) => (unescape_field(fourth), fifth),
        None => (String::new(), fourth),
    };

    Some((
        PathBuf::from(unescape_field(path_str)),
        CachedSkip {
            size,
            mtime_secs,
            fingerprint,
            reason: unescape_field(reason_str),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_cache_roundtrip_and_fingerprint_invalidation() {
        let dir = tempfile::tempdir().unwrap();
        let state_path = dir.path().join("state.tsv");
        let media_path = PathBuf::from("/mnt/nas/videos/foo\tbar\\baz.mp4");
        let hevc_path = PathBuf::from("/mnt/nas/videos/already.mp4");

        {
            let cache =
                StateCache::open_with_fingerprint(&state_path, "jetson-gst|x=42|m=15").unwrap();
            assert_eq!(cache.len(), 0);
            cache.record_skip(&media_path, 12345, 1_700_000_000, "预估节省不足");
            cache.record_permanent_skip(&hevc_path, 999, 1_700_000_000, "已是H.265编码");
            assert_eq!(
                cache
                    .get_cached_skip(&media_path, 12345, 1_700_000_000)
                    .as_deref(),
                Some("预估节省不足")
            );
            assert!(
                cache
                    .get_cached_skip(&media_path, 99999, 1_700_000_000)
                    .is_none()
            );
            assert!(
                cache
                    .get_cached_skip(&media_path, 12345, 1_700_000_001)
                    .is_none()
            );
        }

        let reloaded_same =
            StateCache::open_with_fingerprint(&state_path, "jetson-gst|x=42|m=15").unwrap();
        assert_eq!(reloaded_same.len(), 2);
        assert_eq!(
            reloaded_same
                .get_cached_skip(&media_path, 12345, 1_700_000_000)
                .as_deref(),
            Some("预估节省不足")
        );

        let reloaded_new =
            StateCache::open_with_fingerprint(&state_path, "jetson-gst|x=40|m=10").unwrap();
        assert!(
            reloaded_new
                .get_cached_skip(&media_path, 12345, 1_700_000_000)
                .is_none()
        );
        assert_eq!(
            reloaded_new
                .get_cached_skip(&hevc_path, 999, 1_700_000_000)
                .as_deref(),
            Some("已是H.265编码")
        );
    }
}
