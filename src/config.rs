/// Encoder type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    Cpu,
    Gpu,
}

/// Encoder configuration
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub encoder_type: EncoderType,
    pub preset: String,
}

