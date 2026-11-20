# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Compressor — A Rust CLI tool for hardware-first, perceptually transparent H.265/HEVC video compression and WebP image conversion with a strict zero-bloat guarantee (`output_size < input_size` always):

- **On-demand runtime hardware probing** (`detect_hardware_capabilities`): probes real encoders at startup (never relies solely on `ffmpeg -encoders` text); explicit `--encoder` probes only the requested backend.
- **Jetson official hardware stack (zero custom FFmpeg)**: uses JetPack's preinstalled GStreamer `nvv4l2decoder ! nvvidconv ! nvv4l2h265enc` (10-bit Main10 `P010_10LE`, `maxperf-enable=1`, `iframeinterval=240`, `qp-range`) with **single-pass parallel audio demux/transcode (`mp4mux`)** + standard system `ffmpeg` pure `-c copy` remux (`-tag:v hvc1 -movflags +faststart`). Non-NVDEC formats (e.g. WMV2) fall back seamlessly via single-pass multi-output `ffmpeg | gst-launch-1.0` pipe.
- **Network filesystem (NFS/SMB/FUSE) optimization**: detects network mounts via `statfs`, stages `+faststart` moov relocation and `ffprobe` integrity checks on local `/tmp` when headroom allows, then streams a single 1 MiB-buffered sequential write to same-directory temp before atomic rename.
- **Desktop NVENC (`hevc_nvenc`)**: conditional zero-copy `-hwaccel cuda -hwaccel_output_format cuda`, 10-bit `main10`, `-bf 4 -b_ref_mode middle`, `-b:v 0` pure CQ-VBR, `-multipass qres`, multi-GPU round-robin, and automatic session-limit slot shrinking & retry.
- **Cross-platform hardware**: Intel QSV (`hevc_qsv`), Linux VA-API (`hevc_vaapi`), AMD AMF (`hevc_amf`), Apple VideoToolbox (`hevc_videotoolbox`), and 10-bit `libx265` CPU fallback.
- **Perceptual quality search + 4-layer zero-bloat pipeline**:
  1. Single-call `ffprobe` pre-filter (skip HEVC/AV1/VP9, ultra-low BPP H.264, smart `AudioPlan` copy vs channel-aware AAC, HDR 10-bit protection).
  2. Single-command 6s pure-video (`-an`) reference sample + 3-step noise-knee-aware `XPSNR` (1/3 subsampled, fallback `SSIM`) search + pure video/audio separated budget pruning.
  3. Full encode with real-time `SizeWatchdog` early abort if temp output reaches original file size.
  4. Final gate verifying strict `output_size < input_size`, net saving >= `min_compression_ratio`, and `ffprobe` stream + duration consistency before atomic same-directory replacement.

## Build & Test Commands

```bash
cargo test
cargo build --release
make musl
```

## Architecture

```
src/
├── main.rs       # Entry point, CLI parsing, runtime capability probe, unified streaming pipeline
├── config.rs     # EncoderType, HardwareCapabilities, EncoderConfig, VideoConfig, QualityProfile, EncodeParams
├── model.rs      # Shared MediaItem, ProcessResult, FileStats, MediaStats, ConversionStats
├── encoder.rs    # Table-driven Encoder<'a> backends, shrinkable HwSlotPool, single-cmd sample extractor
├── ffmpeg.rs     # Single-call ffprobe parser, AudioPlan, on-demand HW probe, unified XPSNR/SSIM metrics
├── scheduler.rs  # Unified video pool (HW-slot decoupling, multi-GPU round-robin & session-limit backoff)
├── scanner.rs    # Single-pass streaming media scanner
├── video.rs      # 4-layer zero-bloat perceptual compression pipeline
├── image.rs      # Image-to-WebP conversion with EXIF preservation & O(1) animated GIF guard
├── webp_mux.rs   # libwebp canonical VP8X/EXIF muxing
├── state.rs      # Config-fingerprinted persistent skip-cache (--state-file)
├── signal.rs     # SIGINT/SIGTERM shutdown, SizeWatchdog, shell-free OS pipeline runner
├── logging.rs    # Thread-safe leveled logging
└── utils.rs      # Shared media preflight, crash-safe atomic_replace_from_temp, NFS staging helpers
```
