/// Encoder type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    Cpu,
    Gpu,      // Standard NVENC (desktop GPUs)
    Jetson,   // Jetson NVMPI (Jetson devices)
}

/// Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub encoder_type: EncoderType,
    pub preset: String,
    /// Target bitrate in kbps (for Jetson bitrate-based encoding)
    /// None = use CRF-based default mapping
    pub target_bitrate_kbps: Option<u32>,
    /// Use 2-pass encoding (for Jetson when bitrate is capped)
    pub use_2pass: bool,
}

