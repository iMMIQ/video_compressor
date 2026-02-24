# GPU Encoding Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add NVIDIA NVENC GPU encoding support to the video compressor with CPU fallback.

**Architecture:** Introduce `EncoderType` enum and `EncoderConfig` struct, add NVENC detection, modify FFmpeg command builders to support both libx265 (CPU) and hevc_nvenc (GPU) encoders.

**Tech Stack:** Rust, clap, FFmpeg with hevc_nvenc support

---

### Task 1: Add EncoderType enum and EncoderConfig struct

**Files:**
- Modify: `src/main.rs` (after `use` statements, before `Args` struct)

**Step 1: Add the enum and config struct**

```rust
/// Encoder type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderType {
    Cpu,
    Gpu,
    Auto,
}

/// Encoder configuration
#[derive(Debug, Clone)]
struct EncoderConfig {
    encoder_type: EncoderType,
    preset: String,
}

impl EncoderType {
    /// Parse from string
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "auto" => Ok(Self::Auto),
            _ => anyhow::bail!("Invalid encoder type: {}. Use: cpu, gpu, or auto", s),
        }
    }

    /// Resolve Auto to Cpu or Gpu based on NVENC availability
    fn resolve(self) -> Result<Self> {
        match self {
            Self::Auto => {
                if check_nvenc_available()? {
                    Ok(Self::Gpu)
                } else {
                    Ok(Self::Cpu)
                }
            }
            other => Ok(other),
        }
    }
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: Error - `check_nvenc_available` not defined yet (we'll add it next)

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add EncoderType enum and EncoderConfig struct"
```

---

### Task 2: Add NVENC availability check function

**Files:**
- Modify: `src/main.rs` (before `check_ffmpeg` function, around line 617)

**Step 1: Write the NVENC check function**

```rust
/// Check if NVENC encoder is available
fn check_nvenc_available() -> Result<bool> {
    let output = Command::new("ffmpeg")
        .arg("-encoders")
        .output()
        .context("无法执行ffmpeg命令")?;

    if !output.status.success() {
        return Ok(false);
    }

    let encoders = String::from_utf8_lossy(&output.stdout);
    Ok(encoders.contains("hevc_nvenc"))
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: OK

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add check_nvenc_available function"
```

---

### Task 3: Update Args struct to include encoder flag

**Files:**
- Modify: `src/main.rs` (in `Args` struct, after `preset` field around line 26)

**Step 1: Add encoder field**

Find this section:
```rust
    /// 视频编码器预设
    #[arg(long, default_value = "medium")]
    preset: String,

    /// 音频编码比特率（kbps），设为0保持原音频
```

Add the encoder field:
```rust
    /// 视频编码器预设
    #[arg(long, default_value = "medium")]
    preset: String,

    /// 视频编码器类型 (cpu, gpu, auto)
    #[arg(long, default_value = "auto")]
    encoder: String,

    /// 音频编码比特率（kbps），设为0保持原音频
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: OK

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add --encoder CLI argument"
```

---

### Task 4: Build encoder arguments helper function

**Files:**
- Modify: `src/main.rs` (after `check_ffmpeg` function, around line 641)

**Step 1: Add the build_encode_args function**

```rust
/// Build video encoder arguments for FFmpeg
fn build_encode_args(crf: u8, preset: &str, config: &EncoderConfig) -> Vec<String> {
    match config.encoder_type {
        EncoderType::Cpu => vec![
            "-c:v".to_string(),
            "libx265".to_string(),
            "-crf".to_string(),
            crf.to_string(),
            "-preset".to_string(),
            preset.to_string(),
            "-x265-params".to_string(),
            "fast=1:log-level=error".to_string(),
        ],
        EncoderType::Gpu => vec![
            "-c:v".to_string(),
            "hevc_nvenc".to_string(),
            "-preset".to_string(),
            "p7".to_string(),
            "-tune".to_string(),
            "hq".to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-rc-lookahead".to_string(),
            "32".to_string(),
            "-b_ref_mode".to_string(),
            "middle".to_string(),
            "-qmin".to_string(),
            "24".to_string(),
            "-qmax".to_string(),
            "28".to_string(),
            "-init_qpP".to_string(),
            "24".to_string(),
            "-init_qpB".to_string(),
            "26".to_string(),
            "-init_qpI".to_string(),
            "24".to_string(),
            "-no-scenecut".to_string(),
            "0".to_string(),
            "-spatial_aq".to_string(),
            "1".to_string(),
            "-temporal_aq".to_string(),
            "1".to_string(),
            "-aq-strength".to_string(),
            "8".to_string(),
        ],
        EncoderType::Auto => unreachable!("Auto should be resolved before calling build_encode_args"),
    }
}
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: Error - `EncoderType` is not accessible (needs to be public or we need to adjust scope)

**Step 3: Verify EncoderType is accessible before main function**

The `EncoderType` enum is already defined before `main()` so it should be accessible. Verify compilation again:
Run: `cargo check`

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: add build_encode_args helper function"
```

---

### Task 5: Update main function to create EncoderConfig

**Files:**
- Modify: `src/main.rs` (in `main` function, after line 60-61 where we scan videos)

**Step 1: Create EncoderConfig and update startup output**

Find this section in `main()`:
```rust
    println!("视频压缩工具 (智能压缩模式)");
    println!("输入目录: {}", args.input.display());
    println!("输出模式: 原地替换（后缀名改为.mp4）");
```

Add encoder parsing and config creation after line 56:
```rust
    // 解析编码器类型
    let encoder_type = EncoderType::from_str(&args.encoder)?;
    let encoder_type = encoder_type.resolve()?;
    let encoder_name = match encoder_type {
        EncoderType::Cpu => "CPU (libx265)",
        EncoderType::Gpu => "GPU (hevc_nvenc)",
        EncoderType::Auto => unreachable!("Auto should be resolved"),
    };

    println!("视频压缩工具 (智能压缩模式)");
    println!("输入目录: {}", args.input.display());
    println!("输出模式: 原地替换（后缀名改为.mp4）");
    println!("编码器: {}", encoder_name);
    println!("预览CRF: 23 | 最大CRF: {}", args.max_crf);
```

**Step 2: Create EncoderConfig**

Add after the encoder_type resolution:
```rust
    let encoder_type = encoder_type.resolve()?;
    let encoder_config = EncoderConfig {
        encoder_type,
        preset: args.preset.clone(),
    };
    let encoder_name = match encoder_config.encoder_type {
```

**Step 3: Update FFmpeg check for GPU mode**

Find `check_ffmpeg()?` call around line 60, modify to:
```rust
    // 检查FFmpeg是否可用
    check_ffmpeg(encoder_config.encoder_type)?;
```

**Step 4: Pass encoder_config to process_video**

Find the `process_video` call around line 83-89, modify to:
```rust
        match process_video(
            video_path,
            &encoder_config,
            args.max_crf,
            args.min_compression_ratio,
            args.audio_bitrate,
        ) {
```

**Step 5: Verify compilation**

Run: `cargo check`
Expected: Errors - function signatures don't match yet (we'll fix them)

**Step 6: Commit**

```bash
git add src/main.rs
git commit -m "feat: integrate EncoderConfig into main function"
```

---

### Task 6: Update check_ffmpeg to accept EncoderType

**Files:**
- Modify: `src/main.rs` (modify `check_ffmpeg` function around line 617)

**Step 1: Update function signature**

Change from:
```rust
fn check_ffmpeg() -> Result<()> {
```

To:
```rust
fn check_ffmpeg(encoder_type: EncoderType) -> Result<()> {
```

**Step 2: Add NVENC check for GPU mode**

Add at the end of the function, before the final `Ok(())`:
```rust
    // 检查是否支持libx265
    if version.contains("libx265") {
        println!("✓ H.265编码器可用");
    } else {
        anyhow::bail!("FFmpeg未编译libx265支持，无法进行H.265编码");
    }

    // 对于GPU模式，检查NVENC
    if matches!(encoder_type, EncoderType::Gpu) {
        if version.contains("hevc_nvenc") {
            println!("✓ NVENC编码器可用");
        } else {
            anyhow::bail!("FFmpeg未编译hevc_nvenc支持，无法使用GPU编码");
        }
    }

    Ok(())
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: OK

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: add NVENC check to check_ffmpeg"
```

---

### Task 7: Update process_video signature

**Files:**
- Modify: `src/main.rs` (modify `process_video` function around line 179)

**Step 1: Update function signature**

Change from:
```rust
fn process_video(
    input_path: &Path,
    max_crf: u8,
    min_compression_ratio: u8,
    preset: &str,
    audio_bitrate: u32,
) -> Result<ProcessResult> {
```

To:
```rust
fn process_video(
    input_path: &Path,
    encoder_config: &EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
) -> Result<ProcessResult> {
```

**Step 2: Update preview_encode call**

Find `preview_encode` call around line 236-243, change from:
```rust
    let preview_result = preview_encode(
        input_path,
        PREVIEW_CRF,
        preset,
        audio_bitrate,
        preview_duration,
        duration,
    )?;
```

To:
```rust
    let preview_result = preview_encode(
        input_path,
        PREVIEW_CRF,
        encoder_config,
        audio_bitrate,
        preview_duration,
        duration,
    )?;
```

**Step 3: Update full_encode call**

Find `full_encode` call around line 304, change from:
```rust
    let output_size = full_encode(input_path, temp_path, final_crf, preset, audio_bitrate)?;
```

To:
```rust
    let output_size = full_encode(input_path, temp_path, final_crf, encoder_config, audio_bitrate)?;
```

**Step 4: Verify compilation**

Run: `cargo check`
Expected: Errors - preview_encode and full_encode signatures don't match

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "refactor: update process_video to use EncoderConfig"
```

---

### Task 8: Update preview_encode function

**Files:**
- Modify: `src/main.rs` (modify `preview_encode` function around line 345)

**Step 1: Update function signature**

Change from:
```rust
fn preview_encode(
    input_path: &Path,
    crf: u8,
    preset: &str,
    audio_bitrate: u32,
    duration: u32,
    video_duration: f64,
) -> Result<PreviewResult> {
```

To:
```rust
fn preview_encode(
    input_path: &Path,
    crf: u8,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
    duration: u32,
    video_duration: f64,
) -> Result<PreviewResult> {
```

**Step 2: Replace hardcoded libx265 with build_encode_args**

Find the first encoding section (single segment) around lines 387-400, change from:
```rust
        encode_cmd
            .arg("-y")
            .arg("-i")
            .arg(&segment_paths[0])
            .arg("-c:v")
            .arg("libx265")
            .arg("-crf")
            .arg(crf.to_string())
            .arg("-preset")
            .arg(preset)
            .arg("-x265-params")
            .arg("fast=1:log-level=error")
            .arg("-f")
            .arg("mp4");
```

To:
```rust
        let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
        encode_cmd
            .arg("-y")
            .arg("-i")
            .arg(&segment_paths[0]);
        for arg in video_args {
            encode_cmd.arg(arg);
        }
        encode_cmd.arg("-f").arg("mp4");
```

**Step 3: Replace second libx265 section (multi-segment)**

Find the multi-segment encoding around lines 513-531, change from:
```rust
        encode_cmd
            .arg("-y")
            .arg("-f")
            .arg("concat")
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(concat_list.path())
            .arg("-c:v")
            .arg("libx265")
            .arg("-crf")
            .arg(crf.to_string())
            .arg("-preset")
            .arg(preset)
            .arg("-x265-params")
            .arg("fast=1:log-level=error")
            .arg("-f")
            .arg("mp4");
```

To:
```rust
        let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
        encode_cmd
            .arg("-y")
            .arg("-f")
            .arg("concat")
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(concat_list.path());
        for arg in video_args {
            encode_cmd.arg(arg);
        }
        encode_cmd.arg("-f").arg("mp4");
```

**Step 4: Verify compilation**

Run: `cargo check`
Expected: OK

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "refactor: update preview_encode to use EncoderConfig"
```

---

### Task 9: Update full_encode function

**Files:**
- Modify: `src/main.rs` (modify `full_encode` function around line 573)

**Step 1: Update function signature**

Change from:
```rust
fn full_encode(
    input_path: &Path,
    output_path: &Path,
    crf: u8,
    preset: &str,
    audio_bitrate: u32,
) -> Result<u64> {
```

To:
```rust
fn full_encode(
    input_path: &Path,
    output_path: &Path,
    crf: u8,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
) -> Result<u64> {
```

**Step 2: Replace hardcoded libx265 with build_encode_args**

Find the FFmpeg command building section around lines 580-591, change from:
```rust
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-i")
        .arg(input_path)
        .arg("-c:v")
        .arg("libx265")
        .arg("-crf")
        .arg(crf.to_string())
        .arg("-preset")
        .arg(preset)
        .arg("-x265-params")
        .arg("fast=1:log-level=error");
```

To:
```rust
    let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y")
        .arg("-i")
        .arg(input_path);
    for arg in video_args {
        cmd.arg(arg);
    }
```

**Step 3: Verify compilation**

Run: `cargo check`
Expected: OK

**Step 4: Commit**

```bash
git add src/main.rs
git commit -m "refactor: update full_encode to use EncoderConfig"
```

---

### Task 10: Update EncoderType resolve to handle Auto without NVENC warning

**Files:**
- Modify: `src/main.rs` (modify `EncoderType::resolve` method)

**Step 1: Update resolve method to print warning for Auto fallback**

Change the resolve implementation to:
```rust
    /// Resolve Auto to Cpu or Gpu based on NVENC availability
    fn resolve(self) -> Result<Self> {
        match self {
            Self::Auto => {
                if check_nvenc_available()? {
                    Ok(Self::Gpu)
                } else {
                    println!("注意: NVENC不可用，使用CPU编码");
                    Ok(Self::Cpu)
                }
            }
            other => Ok(other),
        }
    }
```

**Step 2: Verify compilation**

Run: `cargo check`
Expected: OK

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "feat: add warning when Auto mode falls back to CPU"
```

---

### Task 11: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add GPU encoding section to Build Commands**

After the static build section, add:
```markdown
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
```

**Step 2: Update Architecture section**

Add to the application architecture:
```
3. **Encoder detection** (`check_nvenc_available`) - Verifies NVENC support
```

**Step 3: Update Key Algorithm section**

Add note about encoder selection:
```
for each video:
    select encoder (CPU/GPU based on --encoder flag)
    ...
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add GPU encoding documentation"
```

---

### Task 12: Final build and test

**Files:**
- Build: `target/release/video_compressor`

**Step 1: Build release version**

Run: `cargo build --release`
Expected: Success, binary created

**Step 2: Test CPU mode**

Run: `./target/release/video_compressor --input . --encoder cpu --dry-run`
Expected: Lists videos, shows "CPU (libx265)" as encoder

**Step 3: Test GPU mode**

Run: `./target/release/video_compressor --input . --encoder gpu --dry-run`
Expected: Either shows "GPU (hevc_nvenc)" or error if NVENC not available

**Step 4: Test Auto mode**

Run: `./target/release/video_compressor --input . --encoder auto --dry-run`
Expected: Shows "GPU (hevc_nvenc)" if available, or warning + "CPU (libx265)"

**Step 5: Commit final changes**

```bash
git add -A
git commit -m "feat: complete GPU encoding support implementation"
```

---

## Summary

This implementation adds NVIDIA NVENC GPU encoding support with the following changes:

1. **New types**: `EncoderType` enum and `EncoderConfig` struct
2. **NVENC detection**: `check_nvenc_available()` function
3. **CLI flag**: `--encoder` with values `cpu`, `gpu`, `auto`
4. **Encoding args**: `build_encode_args()` helper for both CPU and GPU
5. **Updated flow**: All encoding functions now use `EncoderConfig`

GPU encoding uses:
- `hevc_nvenc` encoder
- `-preset p7 -tune hq -rc vbr -rc-lookahead 32 -b_ref_mode middle -qmin 24 -qmax 28 -init_qpP 24 -init_qpB 26 -init_qpI 24 -no-scenecut 0 -spatial_aq 1 -temporal_aq 1 -aq-strength 8` parameters
- VBR rate control with qmin/qmax quality bounds
