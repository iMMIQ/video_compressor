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
    reason: String,
}

/// Optional persistent cache of skipped files so repeated runs on large
/// directories avoid redundant ffprobe / preview-encode work on files that
/// have not changed.
pub struct StateCache {
    entries: Mutex<HashMap<PathBuf, CachedSkip>>,
    writer: Mutex<Option<BufWriter<File>>>,
}

impl StateCache {
    /// Open or create a state file at `path`. Existing valid lines are loaded.
    pub fn open(path: &Path) -> Result<Self> {
        let mut map = HashMap::new();

        if path.exists() {
            let file = File::open(path)
                .with_context(|| format!("无法打开状态缓存文件: {}", path.display()))?;
            let reader = BufReader::new(file);
            for line in reader.lines().map_while(Result::ok) {
                if let Some((file_path, entry)) = parse_line(&line) {
                    map.insert(file_path, entry);
                }
            }
        }

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("无法创建状态缓存目录: {}", parent.display())
            })?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("无法写入状态缓存文件: {}", path.display()))?;

        Ok(Self {
            entries: Mutex::new(map),
            writer: Mutex::new(Some(BufWriter::new(file))),
        })
    }

    /// Number of loaded entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if `path` with `(size, mtime_secs)` was previously skipped.
    pub fn get_cached_skip(&self, path: &Path, size: u64, mtime_secs: i64) -> Option<String> {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries
            .get(path)
            .filter(|e| e.size == size && e.mtime_secs == mtime_secs)
            .map(|e| e.reason.clone())
    }

    /// Record a skipped file into memory and append it to the state file.
    pub fn record_skip(&self, path: &Path, size: u64, mtime_secs: i64, reason: &str) {
        let entry = CachedSkip {
            size,
            mtime_secs,
            reason: reason.to_string(),
        };

        {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            entries.insert(path.to_path_buf(), entry);
        }

        let mut writer_guard = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(writer) = writer_guard.as_mut() {
            let line = format_line(path, size, mtime_secs, reason);
            if writeln!(writer, "{}", line).is_ok() {
                let _ = writer.flush();
            }
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

fn format_line(path: &Path, size: u64, mtime_secs: i64, reason: &str) -> String {
    format!(
        "{}\t{}\t{}\t{}",
        escape_field(&path.to_string_lossy()),
        size,
        mtime_secs,
        escape_field(reason)
    )
}

fn parse_line(line: &str) -> Option<(PathBuf, CachedSkip)> {
    let mut parts = line.splitn(4, '\t');
    let path_str = parts.next()?;
    let size: u64 = parts.next()?.parse().ok()?;
    let mtime_secs: i64 = parts.next()?.parse().ok()?;
    let reason_str = parts.next()?;

    Some((
        PathBuf::from(unescape_field(path_str)),
        CachedSkip {
            size,
            mtime_secs,
            reason: unescape_field(reason_str),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_cache_roundtrip_and_invalidation() {
        let dir = tempfile::tempdir().unwrap();
        let state_path = dir.path().join("state.tsv");
        let media_path = PathBuf::from("/mnt/nas/videos/foo\tbar\\baz.mp4");

        {
            let cache = StateCache::open(&state_path).unwrap();
            assert_eq!(cache.len(), 0);
            cache.record_skip(&media_path, 12345, 1_700_000_000, "需要CRF=38 超出限制(35)");
            assert_eq!(
                cache.get_cached_skip(&media_path, 12345, 1_700_000_000).as_deref(),
                Some("需要CRF=38 超出限制(35)")
            );
            // Size or mtime change invalidates the entry
            assert!(cache.get_cached_skip(&media_path, 99999, 1_700_000_000).is_none());
            assert!(cache.get_cached_skip(&media_path, 12345, 1_700_000_001).is_none());
        }

        // Reload from disk
        let reloaded = StateCache::open(&state_path).unwrap();
        assert_eq!(reloaded.len(), 1);
        assert_eq!(
            reloaded.get_cached_skip(&media_path, 12345, 1_700_000_000).as_deref(),
            Some("需要CRF=38 超出限制(35)")
        );
    }
}
