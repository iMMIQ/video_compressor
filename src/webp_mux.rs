use anyhow::{Context, Result, bail};
use filetime::{FileTime, set_file_times};
use libwebp_sys::{
    WEBP_MUX_ABI_VERSION, WebPData, WebPDataClear, WebPMux, WebPMuxAssemble, WebPMuxCreateInternal,
    WebPMuxDelete, WebPMuxError, WebPMuxSetChunk,
};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const RIFF_HEADER_LEN: usize = 12;
const CHUNK_HEADER_LEN: usize = 8;

struct MuxHandle(*mut WebPMux);

impl Drop for MuxHandle {
    fn drop(&mut self) {
        unsafe { WebPMuxDelete(self.0) };
    }
}

struct OwnedWebPData(WebPData);

impl OwnedWebPData {
    fn empty() -> Self {
        Self(WebPData::default())
    }
}

impl Drop for OwnedWebPData {
    fn drop(&mut self) {
        unsafe { WebPDataClear(&mut self.0) };
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RepairResult {
    NotAffected,
    WouldRepair { old_size: u64, new_size: u64 },
    Repaired { old_size: u64, new_size: u64 },
}

/// Add EXIF metadata through libwebp's muxer. The muxer creates or updates the
/// VP8X chunk, sets the EXIF feature flag, and emits chunks in canonical order.
pub fn add_exif(webp_data: &[u8], exif_data: &[u8]) -> Result<Vec<u8>> {
    let input = WebPData {
        bytes: webp_data.as_ptr(),
        size: webp_data.len(),
    };
    let mux = unsafe { WebPMuxCreateInternal(&input, 1, WEBP_MUX_ABI_VERSION as std::ffi::c_int) };
    if mux.is_null() {
        bail!("libwebp 无法解析编码后的 WebP");
    }
    let mux = MuxHandle(mux);

    let exif = WebPData {
        bytes: exif_data.as_ptr(),
        size: exif_data.len(),
    };
    let status = unsafe { WebPMuxSetChunk(mux.0, c"EXIF".as_ptr(), &exif, 1) };
    ensure_mux_ok(status, "写入 EXIF")?;

    let mut assembled = OwnedWebPData::empty();
    let status = unsafe { WebPMuxAssemble(mux.0, &mut assembled.0) };
    ensure_mux_ok(status, "组装 WebP")?;
    if assembled.0.bytes.is_null() || assembled.0.size == 0 {
        bail!("libwebp 返回了空的 WebP");
    }

    let output =
        unsafe { std::slice::from_raw_parts(assembled.0.bytes, assembled.0.size).to_vec() };
    Ok(output)
}

fn ensure_mux_ok(status: WebPMuxError, operation: &str) -> Result<()> {
    if status == WebPMuxError::WEBP_MUX_OK {
        Ok(())
    } else {
        bail!("libwebp {}失败: {:?}", operation, status)
    }
}

fn is_riff_webp(data: &[u8]) -> bool {
    data.len() >= RIFF_HEADER_LEN && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP"
}

fn is_broken_exif_first(data: &[u8]) -> bool {
    is_riff_webp(data)
        && data.len() >= RIFF_HEADER_LEN + CHUNK_HEADER_LEN
        && &data[RIFF_HEADER_LEN..RIFF_HEADER_LEN + 4] == b"EXIF"
}

fn chunk_end(data: &[u8], offset: usize) -> Result<usize> {
    if offset
        .checked_add(CHUNK_HEADER_LEN)
        .is_none_or(|end| end > data.len())
    {
        bail!("WebP chunk 头被截断，偏移 {}", offset);
    }
    let size = u32::from_le_bytes(
        data[offset + 4..offset + 8]
            .try_into()
            .expect("chunk size is four bytes"),
    ) as usize;
    let padded_size = size.checked_add(size & 1).context("WebP chunk 长度溢出")?;
    offset
        .checked_add(CHUNK_HEADER_LEN)
        .and_then(|start| start.checked_add(padded_size))
        .filter(|end| *end <= data.len())
        .context("WebP chunk 数据被截断")
}

/// Repair the exact invalid layout emitted by the old implementation:
/// `RIFF WEBP EXIF ... VP8/VP8L` (or `EXIF` before an existing `VP8X`).
/// All chunk payloads are preserved; libwebp only rebuilds the container.
fn repair_broken_data(data: &[u8]) -> Result<Option<Vec<u8>>> {
    if !is_broken_exif_first(data) {
        return Ok(None);
    }

    let declared_size = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize + 8;
    if declared_size != data.len() {
        bail!(
            "RIFF 声明长度 {} 与实际长度 {} 不一致，拒绝自动修复",
            declared_size,
            data.len()
        );
    }

    let first_end = chunk_end(data, RIFF_HEADER_LEN)?;
    let exif_size = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let exif_start = RIFF_HEADER_LEN + CHUNK_HEADER_LEN;
    let exif_end = exif_start
        .checked_add(exif_size)
        .filter(|end| *end <= first_end)
        .context("EXIF chunk 长度无效")?;
    let exif = &data[exif_start..exif_end];

    let mut clean = Vec::with_capacity(data.len() - (first_end - RIFF_HEADER_LEN));
    clean.extend_from_slice(&data[..RIFF_HEADER_LEN]);

    let mut offset = first_end;
    let mut has_image = false;
    while offset < data.len() {
        let end = chunk_end(data, offset)?;
        let fourcc = &data[offset..offset + 4];
        if matches!(fourcc, b"VP8 " | b"VP8L") {
            has_image = true;
        }
        // The buggy writer emitted one EXIF chunk first. Drop any duplicate
        // EXIF chunks so WebPMuxSetChunk can add exactly one canonical chunk.
        if fourcc != b"EXIF" {
            clean.extend_from_slice(&data[offset..end]);
        }
        offset = end;
    }
    if !has_image {
        bail!("未找到 VP8/VP8L 图像 chunk，拒绝自动修复");
    }

    let clean_riff_size = u32::try_from(clean.len() - 8).context("WebP 文件过大")?;
    clean[4..8].copy_from_slice(&clean_riff_size.to_le_bytes());
    add_exif(&clean, exif).map(Some)
}

fn open_without_atime(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOATIME);
        match options.open(path) {
            Ok(file) => return Ok(file),
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied => {
                // O_NOATIME requires ownership or CAP_FOWNER. Fall back to a
                // normal read; repaired files have their timestamps restored.
            }
            Err(error) => return Err(error).context("无法打开 WebP 文件"),
        }
    }

    File::open(path).context("无法打开 WebP 文件")
}

fn write_in_place(path: &Path, data: &[u8], original: &[u8]) -> Result<()> {
    let write = |bytes: &[u8]| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .open(path)
            .context("无法打开 WebP 文件进行写入")?;
        file.seek(SeekFrom::Start(0))
            .context("无法定位 WebP 文件")?;
        file.write_all(bytes).context("无法写入 WebP 文件")?;
        file.set_len(bytes.len() as u64)
            .context("无法调整 WebP 文件长度")?;
        file.sync_all().context("无法同步 WebP 文件")?;
        Ok(())
    };

    if let Err(write_error) = write(data) {
        if let Err(rollback_error) = write(original) {
            bail!(
                "修复写入失败且回滚失败；写入错误: {:#}；回滚错误: {:#}",
                write_error,
                rollback_error
            );
        }
        return Err(write_error).context("修复写入失败，已恢复原文件内容");
    }
    Ok(())
}

/// Inspect and optionally repair a WebP file while preserving its inode,
/// ownership, permissions, xattrs, atime, and mtime. Linux ctime necessarily
/// changes when file contents are written and cannot be restored by userspace.
pub fn repair_file(path: &Path, dry_run: bool) -> Result<RepairResult> {
    let metadata = std::fs::metadata(path).context("无法读取 WebP 文件元数据")?;
    let original_atime = FileTime::from_last_access_time(&metadata);
    let original_mtime = FileTime::from_last_modification_time(&metadata);

    let mut file = open_without_atime(path)?;
    let mut prefix = [0u8; RIFF_HEADER_LEN + CHUNK_HEADER_LEN];
    if file.read_exact(&mut prefix).is_err() || !is_broken_exif_first(&prefix) {
        return Ok(RepairResult::NotAffected);
    }

    let mut original = prefix.to_vec();
    file.read_to_end(&mut original)
        .context("无法读取 WebP 文件")?;
    let Some(repaired) = repair_broken_data(&original)? else {
        return Ok(RepairResult::NotAffected);
    };

    let old_size = original.len() as u64;
    let new_size = repaired.len() as u64;
    if dry_run {
        return Ok(RepairResult::WouldRepair { old_size, new_size });
    }

    if let Err(write_error) = write_in_place(path, &repaired, &original) {
        let restore_result = set_file_times(path, original_atime, original_mtime);
        return match restore_result {
            Ok(()) => Err(write_error),
            Err(time_error) => Err(write_error.context(format!(
                "原内容已回滚，但恢复访问/修改时间失败: {}",
                time_error
            ))),
        };
    }
    set_file_times(path, original_atime, original_mtime)
        .context("WebP 已修复，但恢复访问/修改时间失败")?;

    Ok(RepairResult::Repaired { old_size, new_size })
}

#[cfg(test)]
mod tests {
    use super::*;
    use libwebp_sys::{WebPFeatureFlags, WebPMuxGetChunk, WebPMuxGetFeatures};

    fn make_webp() -> Vec<u8> {
        let pixels = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        webp::Encoder::from_rgb(&pixels, 2, 2).encode(90.0).to_vec()
    }

    fn sample_exif() -> Vec<u8> {
        // Minimal TIFF header used as an opaque EXIF payload by WebPMux.
        b"II*\0\x08\0\0\0\0\0".to_vec()
    }

    fn old_broken_injection(webp: &[u8], exif: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::new();
        chunk.extend_from_slice(b"EXIF");
        chunk.extend_from_slice(&(exif.len() as u32).to_le_bytes());
        chunk.extend_from_slice(exif);
        if exif.len() & 1 == 1 {
            chunk.push(0);
        }

        let mut broken = Vec::new();
        broken.extend_from_slice(&webp[..RIFF_HEADER_LEN]);
        broken.extend_from_slice(&chunk);
        broken.extend_from_slice(&webp[RIFF_HEADER_LEN..]);
        let size = (broken.len() - 8) as u32;
        broken[4..8].copy_from_slice(&size.to_le_bytes());
        broken
    }

    fn mux_features(data: &[u8]) -> u32 {
        let input = WebPData {
            bytes: data.as_ptr(),
            size: data.len(),
        };
        let mux =
            unsafe { WebPMuxCreateInternal(&input, 1, WEBP_MUX_ABI_VERSION as std::ffi::c_int) };
        assert!(!mux.is_null());
        let mux = MuxHandle(mux);
        let mut flags = 0;
        let status = unsafe { WebPMuxGetFeatures(mux.0, &mut flags) };
        assert_eq!(status, WebPMuxError::WEBP_MUX_OK);
        flags
    }

    fn mux_chunk(data: &[u8], fourcc: &'static std::ffi::CStr) -> Vec<u8> {
        let input = WebPData {
            bytes: data.as_ptr(),
            size: data.len(),
        };
        let mux =
            unsafe { WebPMuxCreateInternal(&input, 1, WEBP_MUX_ABI_VERSION as std::ffi::c_int) };
        assert!(!mux.is_null());
        let mux = MuxHandle(mux);
        let mut chunk = WebPData::default();
        let status = unsafe { WebPMuxGetChunk(mux.0, fourcc.as_ptr(), &mut chunk) };
        assert_eq!(status, WebPMuxError::WEBP_MUX_OK);
        unsafe { std::slice::from_raw_parts(chunk.bytes, chunk.size).to_vec() }
    }

    #[test]
    fn mux_adds_canonical_vp8x_and_decodable_exif() {
        let output = add_exif(&make_webp(), &sample_exif()).unwrap();

        assert_eq!(&output[12..16], b"VP8X");
        assert_ne!(
            mux_features(&output) & WebPFeatureFlags::EXIF_FLAG as u32,
            0
        );
        assert_eq!(mux_chunk(&output, c"EXIF"), sample_exif());
        ::image::load_from_memory(&output).expect("muxed WebP should decode");
    }

    #[test]
    fn mux_preserves_existing_alpha_feature() {
        let pixels = [
            255, 0, 0, 128, 0, 255, 0, 64, 0, 0, 255, 32, 255, 255, 255, 0,
        ];
        let base = webp::Encoder::from_rgba(&pixels, 2, 2)
            .encode(90.0)
            .to_vec();
        let output = add_exif(&base, &sample_exif()).unwrap();
        let flags = mux_features(&output);

        assert_ne!(flags & WebPFeatureFlags::EXIF_FLAG as u32, 0);
        assert_ne!(flags & WebPFeatureFlags::ALPHA_FLAG as u32, 0);
        ::image::load_from_memory(&output).expect("alpha WebP should decode");
    }

    #[test]
    fn repairs_exif_first_layout_without_reencoding_pixels() {
        let base = make_webp();
        let exif = sample_exif();
        let broken = old_broken_injection(&base, &exif);

        assert_eq!(&broken[12..16], b"EXIF");
        let repaired = repair_broken_data(&broken).unwrap().unwrap();
        assert_eq!(&repaired[12..16], b"VP8X");
        assert_ne!(
            mux_features(&repaired) & WebPFeatureFlags::EXIF_FLAG as u32,
            0
        );
        ::image::load_from_memory(&repaired).expect("repaired WebP should decode");
        assert!(repair_broken_data(&repaired).unwrap().is_none());
    }

    #[test]
    #[cfg(unix)]
    fn file_repair_preserves_inode_mode_and_timestamps() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.webp");
        let broken = old_broken_injection(&make_webp(), &sample_exif());
        std::fs::write(&path, &broken).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
        let atime = FileTime::from_unix_time(1_650_000_001, 123_456_789);
        let mtime = FileTime::from_unix_time(1_650_000_002, 987_654_321);
        set_file_times(&path, atime, mtime).unwrap();
        let before = std::fs::metadata(&path).unwrap();

        let result = repair_file(&path, false).unwrap();
        assert!(matches!(result, RepairResult::Repaired { .. }));

        let after = std::fs::metadata(&path).unwrap();
        assert_eq!(before.ino(), after.ino());
        assert_eq!(before.uid(), after.uid());
        assert_eq!(before.gid(), after.gid());
        assert_eq!(before.mode(), after.mode());
        assert_eq!(FileTime::from_last_access_time(&after), atime);
        assert_eq!(FileTime::from_last_modification_time(&after), mtime);
        let repaired = std::fs::read(&path).unwrap();
        ::image::load_from_memory(&repaired).expect("repaired file should decode");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn dry_run_does_not_change_timestamps() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.webp");
        let broken = old_broken_injection(&make_webp(), &sample_exif());
        std::fs::write(&path, &broken).unwrap();
        let atime = FileTime::from_unix_time(1_650_000_003, 111_222_333);
        let mtime = FileTime::from_unix_time(1_650_000_004, 444_555_666);
        set_file_times(&path, atime, mtime).unwrap();

        let result = repair_file(&path, true).unwrap();
        assert!(matches!(result, RepairResult::WouldRepair { .. }));

        let after = std::fs::metadata(&path).unwrap();
        assert_eq!(FileTime::from_last_access_time(&after), atime);
        assert_eq!(FileTime::from_last_modification_time(&after), mtime);
        assert_eq!(std::fs::read(&path).unwrap(), broken);
    }
}
