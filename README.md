# Video Compressor

用Rust编写的视频压缩工具，将目录中的所有视频递归转换为H.265格式以减小体积。

## 功能特性

- **智能压缩算法**：
  - 预览编码：先编码几秒视频预估压缩比
  - 动态CRF：自动调整CRF以满足压缩目标
  - H.264源必须达到30%压缩比才执行转换
  - 质量下限：CRF不低于23，保证画质
- 递归扫描目录中的所有视频文件
- 自动跳过已是H.265编码的视频
- 支持跳过已存在文件
- 详细统计信息输出

## 依赖

- Rust (2024 edition)
- FFmpeg (需要libx265支持)
- musl-tools (用于静态链接)

### 安装依赖

#### Arch Linux
```bash
sudo pacman -S ffmpeg musl
```

#### Ubuntu/Debian
```bash
sudo apt install ffmpeg musl-tools musl-dev
```

#### Fedora
```bash
sudo dnf install ffmpeg musl-gcc
```

## 构建

### 标准构建
```bash
cargo build --release
```

### 静态链接构建
```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

或使用Makefile:
```bash
make musl
```

静态链接的二进制文件位于: `target/x86_64-unknown-linux-musl/release/video_compressor`

## 使用

```bash
# 基本使用（使用智能压缩）
./video_compressor -i /path/to/videos

# 指定输出目录
./video_compressor -i /path/to/videos -o /path/to/output

# 调整压缩要求（H264源必须达到50%压缩）
./video_compressor -i /path/to/videos --min-compression-ratio 50

# 调整质量范围
./video_compressor -i /path/to/videos --min-crf 20 --max-crf 30

# 调整预览时长（默认10秒）
./video_compressor -i /path/to/videos --preview-duration 15

# 跳过已存在的文件
./video_compressor -i /path/to/videos --skip-existing

# 仅扫描不转换
./video_compressor -i /path/to/videos --dry-run
```

## 智能压缩算法

### 工作流程

1. **扫描视频**：递归查找目录中的视频文件
2. **检测编码**：使用ffprobe检测源视频编码
3. **跳过HEVC**：已是H.265编码的视频直接跳过
4. **预览编码**：编码前N秒视频，预估压缩比
5. **动态调整CRF**：
   - 从初始CRF（默认28）开始
   - 如果H.264源压缩比不足30%，提高CRF
   - CRF不超过max-crf（默认35）
   - CRF不低于min-crf（默认23）
6. **执行转换**：使用确定的CRF进行完整编码
7. **最终验证**：确保实际压缩比满足要求

### 为什么这样设计

- **预览编码**：避免完整编码后发现压缩比不足
- **CRF >= 23**：保证画质不低于"高质量"水平
- **30%压缩比**：H.265比H.264效率高约50%，30%是合理下限
- **自动跳过**：已压缩好的视频不需要重复处理

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i, --input` | 输入目录 | 必需 |
| `-o, --output` | 输出目录 | 输入目录/compressed |
| `-c, --crf` | 初始CRF值 | 28 |
| `--min-crf` | 最小CRF（质量下限） | 23 |
| `--max-crf` | 最大CRF（质量上限） | 35 |
| `--min-compression-ratio` | H264源最小压缩比(%) | 30 |
| `--preview-duration` | 预览编码时长(秒) | 10 |
| `--preset` | 编码预设 | medium |
| `-a, --audio-bitrate` | 音频比特率(kbps) | 128 |
| `--skip-existing` | 跳过已存在文件 | false |
| `--dry-run` | 仅扫描不执行 | false |

### CRF 参考值

- 18-23: 高质量到视觉无损
- 23-28: 中等质量（推荐范围）
- 28-35: 较低质量
- 35+: 低质量

### 编码预设

从快到慢: `ultrafast` -> `superfast` -> `veryfast` -> `faster` -> `fast` -> `medium` -> `slow` -> `slower` -> `veryslow`

较慢的预设会产生更小的文件，但需要更长的处理时间。

## 支持的视频格式

- MP4, MKV, AVI, MOV, FLV, WMV, WebM, M4V, MPG, MPEG, 3GP

## 项目结构

```
.
├── src/
│   └── main.rs       # 主程序
├── Cargo.toml        # 项目配置
├── .cargo/
│   └── config.toml   # 静态链接配置
├── Makefile          # 构建脚本
└── README.md         # 本文件
```

## 许可

MIT License
