# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Compressor - A Rust tool that recursively converts videos to H.265/HEVC format with intelligent compression logic:
- Fixed CRF=23 preview encoding (10% of video duration, 5-30 seconds)
- If predicted compression >= 30%: use CRF=23 for full encode (best quality)
- If predicted compression < 30%: extrapolate higher CRF using exponential formula
- Maximum CRF limit of 35 (skip video if extrapolated CRF > 35)
- H.264 sources must achieve 30% compression or conversion is skipped

## Build Commands

### Standard build (dynamic linking):
```bash
cargo build --release
```

### Static build (musl target, requires musl-tools):
```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

Or use Makefile:
```bash
make musl    # Static build
make build   # Standard build
```

Static binary location: `target/x86_64-unknown-linux-musl/release/video_compressor`

### Run tests:
```bash
cargo test
```

### Run directly (with args):
```bash
cargo run -- --input /path/to/videos --max-crf 35 --min-compression-ratio 30
```

### Run with GPU encoding:
```bash
cargo run -- --input /path/to/videos --encoder gpu
```

### Run with auto-detection (default):
```bash
cargo run -- --input /path/to/videos --encoder auto
```

### Run with CPU encoding:
```bash
cargo run -- --input /path/to/videos --encoder cpu
```

## Architecture

## Architecture

The application is a single-binary CLI tool:

1. **CLI parsing** (`Args` struct) - clap derive API for argument parsing
2. **Encoder detection** (`check_nvenc_available`) - Verifies NVENC support for GPU encoding
3. **FFmpeg detection** (`check_ffmpeg`) - Verifies FFmpeg and libx265/hevc_nvenc support
4. **Video scanning** (`scan_videos`) - walkdir for recursive file discovery
5. **Video info** (`get_video_info`) - ffprobe for codec, resolution, duration
6. **Preview encoding** (`preview_encode`) - Encodes 10% duration at CRF=23
7. **CRF extrapolation** - Uses exponential formula if compression insufficient
8. **Full encoding** (`full_encode`) - Complete conversion with selected CRF

### Key Algorithm (process_video)

```
for each video:
    select encoder (CPU/GPU based on --encoder flag)
    detect source codec
    if HEVC: skip (already compressed)

    preview_duration = 10% of video (capped at 5-30 seconds)
    preview_result = preview_encode(CRF=23, preview_duration)

    # Apply correction factor (preview encode is ~1.8x smaller than full)
    predicted_ratio = 1 - (preview_output_per_sec * duration * 1.8) / input_size

    if predicted_ratio >= 30%:
        final_crf = 23  # Best quality that meets target
    else:
        # Exponential extrapolation: CRF +6 ≈ bitrate halved
        extrapolated_crf = 23 - 6 * log2(target_output_ratio / predicted_output_ratio)

        if extrapolated_crf > 35:
            skip video  # Would require too much quality loss
        else:
            final_crf = extrapolated_crf

    full_encode(final_crf)
```

## Static Linking Configuration

Located in `.cargo/config.toml`:
- Uses `x86_64-unknown-linux-musl` target
- `link-self-contained=yes` and `-static` flag

## Dependencies

- `clap` - CLI argument parsing with derive macros
- `anyhow` - Error handling
- `walkdir` - Recursive directory traversal
- `tempfile` - Temporary files for preview encoding
