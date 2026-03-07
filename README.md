# Video Compressor

用Rust编写的高效视频压缩工具，将目录中的所有视频递归转换为H.265格式以减小体积。

## 功能特性

- **智能压缩算法**：
  - 预览编码：先编码10%视频（5-30秒）预估压缩比
  - 动态CRF：CRF=23优先，不足时自动外推更高CRF
  - H.264源必须达到30%压缩比才执行转换
  - 质量上限：CRF最大35，超过则跳过
- **GPU硬件加速**：
  - NVIDIA NVENC (hevc_nvenc)
  - Jetson Orin NVMPI (hevc_nvmpi)
  - 自动检测可用编码器
- **GPU并行处理**：支持多视频并行编码
- **流式扫描**：边扫描边处理，无需等待扫描完成
- 自动跳过已是H.265编码的视频
- 原地替换（保留原文件名，扩展名改为.mp4）
- 详细统计信息输出

## 依赖

- Rust (2024 edition)
- FFmpeg (需要libx265支持，GPU编码需hevc_nvenc或hevc_nvmpi支持)

### 安装依赖

#### Arch Linux
```bash
sudo pacman -S ffmpeg
```

#### Ubuntu/Debian
```bash
sudo apt install ffmpeg
```

#### Fedora
```bash
sudo dnf install ffmpeg
```

## 构建

### 标准构建
```bash
cargo build --release
```

### 静态链接构建 (需要musl-tools)
```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

或使用Makefile:
```bash
make musl    # 静态构建
make build   # 标准构建
```

静态链接的二进制文件位于: `target/x86_64-unknown-linux-musl/release/video_compressor`

## 使用

```bash
# 基本使用（自动检测编码器）
./video_compressor -i /path/to/videos

# 指定编码器
./video_compressor -i /path/to/videos --encoder cpu    # 强制CPU编码
./video_compressor -i /path/to/videos --encoder gpu    # NVIDIA GPU编码
./video_compressor -i /path/to/videos --encoder jetson # Jetson编码

# 调整压缩要求（H264源必须达到50%压缩）
./video_compressor -i /path/to/videos --min-compression-ratio 50

# 调整质量上限
./video_compressor -i /path/to/videos --max-crf 30

# 设置编码预设
./video_compressor -i /path/to/videos --preset fast

# 调整音频比特率（设为0保留原音频）
./video_compressor -i /path/to/videos --audio-bitrate 192
./video_compressor -i /path/to/videos -a 0

# GPU并行处理（设置并发任务数）
./video_compressor -i /path/to/videos --jobs 4

# 强制串行处理（禁用GPU并行）
./video_compressor -i /path/to/videos --serial

# 仅扫描不转换
./video_compressor -i /path/to/videos --dry-run
```

## 智能压缩算法

### 工作流程

1. **流式扫描**：递归查找目录中的视频文件，边扫描边处理
2. **检测编码**：使用ffprobe检测源视频编码
3. **跳过HEVC**：已是H.265编码的视频直接跳过
4. **预览编码**：编码10%时长视频（限制5-30秒），使用CRF=23
5. **预测压缩比**：根据预览结果推算完整视频压缩比
6. **动态调整CRF**：
   - 如果预测压缩比 >= 30%: 使用CRF=23（最佳质量）
   - 如果预测压缩比 < 30%: 指数外推更高CRF
   - 如果外推CRF > max-crf: 跳过该视频
7. **执行转换**：使用确定的CRF进行完整编码

### 为什么这样设计

- **CRF=23预览**：保证画质不低于"高质量"水平
- **30%压缩比**：H.265比H.264效率高约50%，30%是合理下限
- **指数外推**：CRF每增加6，码率约减半，用于推算达到目标压缩比所需CRF
- **原地替换**：简化输出管理，保持目录结构

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入目录 | 必需 |
| `--encoder` | 编码器类型 (cpu/gpu/jetson/auto) | auto |
| `--max-crf` | 最大CRF（质量下限） | 35 |
| `--min-compression-ratio` | H264源最小压缩比(%) | 30 |
| `--preset` | 编码预设 | medium |
| `-a, --audio-bitrate` | 音频比特率(kbps)，0保留原音频 | 128 |
| `-j, --jobs` | 并发任务数 | 1 |
| `--serial` | 强制串行处理 | false |
| `--dry-run` | 仅扫描不执行 | false |

### CRF 参考值

- 18-23: 高质量到视觉无损
- 23-28: 中等质量（推荐范围）
- 28-35: 较低质量
- 35+: 低质量

### 编码预设

从快到慢: `ultrafast` -> `superfast` -> `veryfast` -> `faster` -> `fast` -> `medium` -> `slow` -> `slower` -> `veryslow`

较慢的预设会产生更小的文件，但需要更长的处理时间。

## GPU编码支持

### NVIDIA NVENC

需要NVIDIA GPU和驱动支持，FFmpeg需编译时包含`--enable-nvenc`。

检测支持：
```bash
ffmpeg -encoders | grep nvenc
```

### Jetson Orin NVMPI

适用于NVIDIA Jetson设备，使用硬件视频编码引擎。

检测支持：
```bash
ffmpeg -encoders | grep nvmpi
```

## 支持的视频格式

- MP4, MKV, AVI, MOV, FLV, WMV, WebM, M4V, MPG, MPEG, 3GP

## 项目结构

```
.
├── src/
│   ├── main.rs       # 程序入口，CLI参数解析
│   ├── config.rs     # 编码器配置
│   ├── encoder.rs    # 编码逻辑
│   ├── ffmpeg.rs     # FFmpeg/ffprobe调用
│   ├── gpu.rs        # GPU编码器检测
│   ├── scheduler.rs  # GPU并行调度器
│   ├── scanner.rs    # 视频文件扫描
│   ├── utils.rs      # 工具函数
│   └── video.rs      # 视频处理逻辑
├── Cargo.toml        # 项目配置
├── .cargo/
│   └── config.toml   # 静态链接配置
├── Makefile          # 构建脚本
├── CLAUDE.md         # 项目说明（Claude Code）
└── README.md         # 本文件
```

## 许可

MIT License
