/// Video encoder backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    /// Software encoding via libx265 (10-bit when supported).
    Cpu,
    /// Desktop NVIDIA NVENC (hevc_nvenc).
    Gpu,
    /// Official JetPack GStreamer NVENC (nvv4l2h265enc + nvv4l2decoder + standard ffmpeg remux).
    /// Requires NO custom-compiled FFmpeg.
    JetsonGst,
    /// Legacy Jetson jetson-ffmpeg wrapper (hevc_nvmpi), kept as opt-in/fallback.
    JetsonNvmpi,
    /// Intel Quick Sync Video (hevc_qsv).
    Qsv,
    /// Linux VA-API (hevc_vaapi, AMD/Intel).
    Vaapi,
    /// AMD AMF (hevc_amf).
    Amf,
    /// Apple VideoToolbox (hevc_videotoolbox).
    VideoToolbox,
}

impl EncoderType {
    pub fn display_name(self) -> &'static str {
        match self {
            EncoderType::Cpu => "CPU (libx265)",
            EncoderType::Gpu => "NVIDIA GPU (hevc_nvenc)",
            EncoderType::JetsonGst => "Jetson 官方硬件 (GStreamer nvv4l2h265enc)",
            EncoderType::JetsonNvmpi => "Jetson NVMPI (hevc_nvmpi, 兼容模式)",
            EncoderType::Qsv => "Intel QSV (hevc_qsv)",
            EncoderType::Vaapi => "VA-API (hevc_vaapi)",
            EncoderType::Amf => "AMD AMF (hevc_amf)",
            EncoderType::VideoToolbox => "Apple VideoToolbox (hevc_videotoolbox)",
        }
    }

    pub fn short_id(self) -> &'static str {
        match self {
            EncoderType::Cpu => "cpu",
            EncoderType::Gpu => "nvenc",
            EncoderType::JetsonGst => "jetson-gst",
            EncoderType::JetsonNvmpi => "jetson-nvmpi",
            EncoderType::Qsv => "qsv",
            EncoderType::Vaapi => "vaapi",
            EncoderType::Amf => "amf",
            EncoderType::VideoToolbox => "videotoolbox",
        }
    }

    /// Whether this backend runs primarily on dedicated video hardware
    /// (leaving CPU cores free for concurrent image encoding and XPSNR evaluation).
    pub fn is_hardware(self) -> bool {
        !matches!(self, EncoderType::Cpu)
    }

    /// Whether this backend can run heterogeneous image processing concurrently with video encoding.
    pub fn allows_concurrent_images(self) -> bool {
        !matches!(self, EncoderType::Cpu)
    }
}

/// Runtime-probed encoder and filter capabilities on the current machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareCapabilities {
    // Compiled-in FFmpeg encoders
    pub has_libx265: bool,
    pub x265_10bit: bool,
    pub has_libx264: bool,

    // Runtime-verified hardware encoders
    pub nvenc_ok: bool,
    pub nvenc_10bit: bool,
    pub nvenc_bframes: bool,
    pub nvenc_multipass: bool,
    pub nvenc_uhq: bool,
    pub nvenc_gpu_count: usize,

    pub jetson_gst_ok: bool,
    pub jetson_gst_10bit: bool,
    pub jetson_gst_decoder: bool,

    pub jetson_nvmpi_ok: bool,

    pub qsv_ok: bool,
    pub vaapi_ok: bool,
    pub vaapi_device: Option<String>,
    pub amf_ok: bool,
    pub videotoolbox_ok: bool,

    // Perceptual quality filters available in standard FFmpeg
    pub has_xpsnr: bool,
    pub has_ssim: bool,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            has_libx265: false,
            x265_10bit: false,
            has_libx264: false,
            nvenc_ok: false,
            nvenc_10bit: false,
            nvenc_bframes: false,
            nvenc_multipass: false,
            nvenc_uhq: false,
            nvenc_gpu_count: 1,
            jetson_gst_ok: false,
            jetson_gst_10bit: false,
            jetson_gst_decoder: false,
            jetson_nvmpi_ok: false,
            qsv_ok: false,
            vaapi_ok: false,
            vaapi_device: None,
            amf_ok: false,
            videotoolbox_ok: false,
            has_xpsnr: false,
            has_ssim: false,
        }
    }
}

/// Global encoder configuration passed from CLI & runtime detection.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub encoder_type: EncoderType,
    pub preset: String,
    pub capabilities: HardwareCapabilities,
    /// Target perceptual XPSNR threshold in dB (higher = closer to visually lossless).
    pub target_xpsnr: f64,
}

/// Quality search bounds native to an encoder backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualityProfile {
    /// Highest quality (lowest QP/CRF) allowed during perceptual search.
    pub min_q: u8,
    /// Starting anchor QP/CRF for first probe.
    pub anchor_q: u8,
    /// Lowest quality (highest QP/CRF) allowed during perceptual search.
    pub max_q: u8,
    /// Step size used when probing below anchor for noisy/complex footage.
    pub low_probe_delta: u8,
}

/// Concrete encoding parameters resolved for a single encode invocation.
#[derive(Debug, Clone)]
pub struct EncodeParams {
    /// Encoder-native quality value (CRF for x265, CQ for NVENC/QSV/VAAPI/AMF, QP for Jetson GStreamer).
    pub quality: u8,
    pub preset: String,
    /// True during perceptual search samples (uses faster hardware preset where available).
    pub fast_preview: bool,
    /// Prefer 10-bit output (Main10 / P010_10LE / yuv420p10le) for better R-D efficiency & HDR safety.
    pub ten_bit: bool,
    pub pix_fmt: Option<String>,
    /// Target bitrate in kbps (used only by legacy Jetson NVMPI).
    pub target_bitrate_kbps: Option<u32>,
    /// Optional VBV soft peak bitrate cap in kbps (never equal to target_bitrate to avoid pseudo-CBR).
    pub maxrate_kbps: Option<u32>,
    /// Optional GPU index for multi-GPU NVENC machines.
    pub gpu_index: Option<usize>,
}

impl EncodeParams {
    pub fn for_quality(quality: u8, preset: &str, ten_bit: bool, fast_preview: bool) -> Self {
        Self {
            quality,
            preset: preset.to_string(),
            fast_preview,
            ten_bit,
            pix_fmt: None,
            target_bitrate_kbps: None,
            maxrate_kbps: None,
            gpu_index: None,
        }
    }
}
