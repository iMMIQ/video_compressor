# GPU Encoding Support Design

## Overview
Add NVIDIA NVENC GPU encoding support to the video compressor, with CPU libx265 fallback.

## Requirements
- Support NVIDIA NVENC (hevc_nvenc) with p6 preset
- Support CPU libx265 as fallback option
- User selects encoder via `--encoder` flag (cpu, gpu, auto)
- Default to auto-detect GPU availability

## Architecture

### Encoder Type Enum
```rust
enum EncoderType {
    Cpu,
    Gpu,
    Auto,
}

struct EncoderConfig {
    encoder_type: EncoderType,
    preset: String,  // Only used for CPU
}
```

### CLI Changes
- Add `--encoder` flag: `cpu`, `gpu`, or `auto` (default: `auto`)
- Existing `--preset` flag only applies to CPU encoding
- GPU always uses `p6` preset (fixed)

### Runtime Detection
- `cpu` mode: Skip NVENC check
- `gpu` mode: Require NVENC, error if unavailable
- `auto` mode: Detect NVENC, fall back to CPU if unavailable

## FFmpeg Command Changes

### GPU Encoding (hevc_nvenc)
```
-c:v hevc_nvenc -preset p7 -tune hq -rc lookahead -rc-lookahead 32 -b_ref_mode middle -b 4 -init_qpP 21 -init_qpB 23 -init_qpI 21 -no-scenecut 0 -spatial_aq 1 -temporal_aq 1 -aq-strength 8
```

### CPU Encoding (libx265)
```
-c:v libx265 -crf <value> -preset <preset> -x265-params "fast=1:log-level=error"
```

### CRF Mapping for GPU
- GPU uses fixed quality parameters (init_qpP 21, init_qpB 23, init_qpI 21)
- CRF extrapolation logic still applies but GPU quality is controlled by init_qp values

## Data Structures

### Updated Args
```rust
struct Args {
    // ... existing fields ...
    #[arg(long, default_value = "auto")]
    encoder: String,
}
```

### Function Signatures
- `process_video()` takes `&EncoderConfig`
- `preview_encode()` takes `&EncoderConfig`
- `full_encode()` takes `&EncoderConfig`

## Error Handling

### NVENC Detection
- `check_nvenc_available()` runs `ffmpeg -encoders`
- Parses output for `hevc_nvenc`
- Returns `bool`

### Validation
- `gpu` without NVENC → error
- `auto` without NVENC → warn, use CPU
- GPU encoding failure → return error (no silent fallback)

## Implementation Functions

### New Functions
- `check_nvenc_available() -> Result<bool>`
- `resolve_encoder_type(EncoderType) -> Result<EncoderType>`
- `build_encode_args(crf, preset, config) -> Vec<String>`

### Modified Functions
- `check_ffmpeg()` - optionally check NVENC
- `process_video()` - accept `EncoderConfig`
- `preview_encode()` - use `build_encode_args()`
- `full_encode()` - use `build_encode_args()`

## Testing
1. CPU mode - existing behavior unchanged
2. GPU mode with NVENC - verify hevc_nvenc works
3. GPU mode without NVENC - verify error message
4. Auto mode with NVENC - verify GPU selected
5. Auto mode without NVENC - verify CPU fallback
6. CRF extrapolation - verify CQ calculation
