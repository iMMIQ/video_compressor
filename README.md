# Video Compressor

用 Rust 编写的硬件优先、感知无损视频/图片压缩工具：尽可能调用本机最快的硬件编码器，在保证人眼不易察觉画质损失的前提下最大化减小体积，并严格保证**转换后文件绝不大于原文件**。

## 核心特性

- **硬件优先 & 零自编译 FFmpeg**：
  - **Jetson Orin / JetPack**：默认采用官方原生 **GStreamer `nvv4l2h265enc` (10-bit Main10) + `nvv4l2decoder` NVMM 零拷贝**，并在**同一次顺序读盘中并行完成音频分流/转码**，随后由系统标准 `ffmpeg` 纯 `-c copy` 毫秒级合流封装（彻底告别自编译 `jetson-ffmpeg`；冷门格式自动走标准 `ffmpeg` 多输出软解管道喂给硬件编码器；仍保留 `--encoder nvmpi` 兼容选项）
  - **NFS / 网络存储 I/O 优化**：自动检测 NFS/SMB/FUSE 等网络挂载目录，本地临时盘空间充足时在本地 SSD 完成 `+faststart` 重排与完整性校验，再以 1 MiB 缓冲**单次顺序回写**并原子替换（消除二次通读大文件与 `+faststart` 三倍网络读写）
  - **桌面 NVIDIA GPU**：`hevc_nvenc`（10-bit Main10 + 条件 `-hwaccel_output_format cuda` 零拷贝 + `-bf 4 -b:v 0` + `-multipass qres` + 多 GPU 轮询 + 会话数上限自动降并发重试）
  - **跨平台硬件**：自动运行时探测并支持 **Intel QSV (`hevc_qsv`)**、**Linux VA-API (`hevc_vaapi`)**、**AMD AMF (`hevc_amf`)**、**Apple VideoToolbox (`hevc_videotoolbox`)**
  - **CPU 兜底**：`libx265` 10-bit + 心理视觉优化 (`aq-mode=3:limit-sao=1:deblock=-1,-1`)
- **感知无损搜索 + 四层“绝不大于原片”硬防线**：
  1. **零编码预筛**：自动跳过已是 HEVC / AV1 / VP9 的视频与超低 BPP 紧压 H.264；智能音频策略（低码率兼容音频直拷 `-c:a copy`，多声道自动保护码率）；HDR 10-bit 防降级
  2. **硬件加速感知质量搜索**：单命令提取 6s 纯视频样本，基于 FFmpeg 内置 `XPSNR`（1/3 抽帧提速，无 `xpsnr` 时自动退回 `SSIM`）执行 3 步噪点拐点感知搜索，在满足视觉透明前提下选中最小体积参数，若预估未达净节省阈值直接跳过
  3. **编码期实时体积熔断**：全量编码期间实时监控输出文件大小，一旦触及原片体积立即终止并保留原文件
  4. **落盘前硬门禁**：严格校验 `output_size < input_size`、净节省达标、视频流完整且时长一致后才原子替换（带 `-tag:v hvc1` 苹果兼容标签与 `-movflags +faststart`）
- **异构并行流水线**：
  - 硬件模式下（含 Jetson 与桌面 GPU）采用硬件编码槽解耦调度，视频硬件编码与 CPU 图片 WebP 转换并发执行
  - 单次流式扫描、SIGINT/SIGTERM 优雅退出、带配置指纹的增量跳过缓存（`--state-file`）

## 依赖

- Rust (2024 edition)
- 标准系统 **FFmpeg / ffprobe**（无需任何第三方补丁重编译）
- **Jetson 平台**：JetPack 官方预装的 `gstreamer1.0` + NVIDIA L4T `nvv4l2` 插件

## 构建

```bash
cargo build --release
# 或静态构建
make musl
```

## 使用

```bash
# 自动运行时探测最快可用硬件（Jetson 自动使用官方 GStreamer 10-bit nvv4l2h265enc）
./video_compressor -i /path/to/videos

# 调整感知质量目标（XPSNR 越高越接近无损，推荐 41.0-43.5 dB）
./video_compressor -i /path/to/videos --target-xpsnr 42.5

# 调整最小净节省门槛（默认 15%；无论设多少均严格保证输出小于原文件）
./video_compressor -i /path/to/videos --min-compression-ratio 20

# 显式指定后端
./video_compressor -i /path/to/videos --encoder jetson-gst    # Jetson 官方硬件
./video_compressor -i /path/to/videos --encoder gpu           # 桌面 NVIDIA NVENC
./video_compressor -i /path/to/videos --encoder qsv           # Intel QSV
./video_compressor -i /path/to/videos --encoder vaapi         # AMD/Intel VA-API
./video_compressor -i /path/to/videos --encoder cpu           # CPU libx265 10-bit

# 控制并发与增量缓存
./video_compressor -i /path/to/videos --jobs 2 --state-file ~/.cache/vc.state

# 仅扫描不执行
./video_compressor -i /path/to/videos --dry-run

# WebP 容器修复模式
./video_compressor -i /path/to/pictures --repair-webp
```

## 参数说明

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `-i, --input` | 输入目录或单个文件 | 必需 |
| `--encoder` | 编码器 (`auto`/`gpu`/`jetson`/`jetson-gst`/`nvmpi`/`qsv`/`vaapi`/`amf`/`videotoolbox`/`cpu`) | `auto` |
| `--target-xpsnr` | 目标感知质量阈值 (dB，推荐 41.0~43.5) | `42.0` |
| `--max-crf` | 感知搜索允许的最高质量步进上限 (CRF/CQ/QP) | `34` |
| `--min-compression-ratio` | 最小净节省比例 (%)；同时硬保证输出严格小于原片 | `15` |
| `--preset` | 编码预设 (`fast`/`medium`/`slow`/`p1`..`p7`) | `medium` |
| `-a, --audio-bitrate` | 音频目标码率 (kbps)，`0` 强制直拷；兼容且码率未超标时自动直拷 | `128` |
| `-j, --jobs` | 最大硬件并发数 (`0`=自动：桌面GPU动态至8，Jetson/HW为2，CPU为1) | `0` |
| `--serial` | 强制串行处理 | `false` |
| `--dry-run` | 仅扫描不执行 | `false` |
| `--repair-webp` | 修复旧版 EXIF-first WebP 后退出 | `false` |
| `--log-level` | 日志级别 (`quiet`/`info`/`debug`) | `info` |
| `--state-file` | 带配置指纹的增量跳过缓存文件 | 无 |
| `-f, --force` (`--reencode-hevc`) | 强制不预跳过已是 HEVC/AV1/VP9 或低 BPP 的视频（仍严格保证输出小于原片） | `false` |

## 许可

MIT License
