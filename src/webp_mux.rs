use anyhow::{Result, bail};
use libwebp_sys::{
    WEBP_MUX_ABI_VERSION, WebPData, WebPDataClear, WebPMux, WebPMuxAssemble, WebPMuxCreateInternal,
    WebPMuxDelete, WebPMuxError, WebPMuxSetChunk,
};

struct MuxHandle(*mut WebPMux);

impl Drop for MuxHandle {
    fn drop(&mut self) {
        unsafe { WebPMuxDelete(self.0) };
    }
}

struct OwnedWebPData(WebPData);

impl Drop for OwnedWebPData {
    fn drop(&mut self) {
        unsafe { WebPDataClear(&mut self.0) };
    }
}

fn ensure_mux_ok(status: WebPMuxError, operation: &str) -> Result<()> {
    if status == WebPMuxError::WEBP_MUX_OK {
        Ok(())
    } else {
        bail!("libwebp {}失败: {:?}", operation, status)
    }
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

    let mut assembled = OwnedWebPData(WebPData::default());
    let status = unsafe { WebPMuxAssemble(mux.0, &mut assembled.0) };
    ensure_mux_ok(status, "组装 WebP")?;
    if assembled.0.bytes.is_null() || assembled.0.size == 0 {
        bail!("libwebp 返回了空的 WebP");
    }

    Ok(unsafe { std::slice::from_raw_parts(assembled.0.bytes, assembled.0.size).to_vec() })
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
        b"II*\0\x08\0\0\0\0\0".to_vec()
    }

    fn open_mux(data: &[u8]) -> MuxHandle {
        let input = WebPData {
            bytes: data.as_ptr(),
            size: data.len(),
        };
        let mux =
            unsafe { WebPMuxCreateInternal(&input, 1, WEBP_MUX_ABI_VERSION as std::ffi::c_int) };
        assert!(!mux.is_null());
        MuxHandle(mux)
    }

    fn mux_features(data: &[u8]) -> u32 {
        let mux = open_mux(data);
        let mut flags = 0;
        assert_eq!(
            unsafe { WebPMuxGetFeatures(mux.0, &mut flags) },
            WebPMuxError::WEBP_MUX_OK
        );
        flags
    }

    fn mux_chunk(data: &[u8], fourcc: &'static std::ffi::CStr) -> Vec<u8> {
        let mux = open_mux(data);
        let mut chunk = WebPData::default();
        assert_eq!(
            unsafe { WebPMuxGetChunk(mux.0, fourcc.as_ptr(), &mut chunk) },
            WebPMuxError::WEBP_MUX_OK
        );
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
}
