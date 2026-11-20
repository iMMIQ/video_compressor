/// Encoder type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    Cpu,
    Gpu,    // Standard NVENC (desktop GPUs)
    Jetson, // Jetson NVMPI (Jetson devices)
}

/// Global encoder configuration passed from CLI.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub encoder_type: EncoderType,
    pub preset: String,
}

/// Concrete encoding parameters resolved for a single encode invocation.
#[derive(Debug, Clone)]
pub struct EncodeParams {
    pub crf: u8,
    pub preset: String,
    pub pix_fmt: Option<String>,
    /// Target bitrate in kbps (used by Jetson bitrate-based encoding).
    pub target_bitrate_kbps: Option<u32>,
    /// Optional maxrate cap in kbps (used by Jetson when capped to 70% of source).
    pub maxrate_kbps: Option<u32>,
}

impl EncodeParams {
    pub fn from_crf(crf: u8, preset: &str) -> Self {
        Self {
            crf,
            preset: preset.to_string(),
            pix_fmt: None,
            target_bitrate_kbps: None,
            maxrate_kbps: None,
        }
    }
}
