use anyhow::{Context, Result};
use clap::Parser;
use crossbeam_channel::{Receiver, Sender, unbounded};
use filetime::{set_file_times, FileTime};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;
use walkdir::WalkDir;

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
                    println!("注意: NVENC不可用，使用CPU编码");
                    Ok(Self::Cpu)
                }
            }
            other => Ok(other),
        }
    }
}

// ============================================================================
// GPU 监控与并发控制
// ============================================================================

const MIN_CONCURRENT: usize = 1;
const MAX_CONCURRENT: usize = 8;
const MARGINAL_GAIN_THRESHOLD: u8 = 5;
const UTILIZATION_HIGH: u8 = 95;

/// GPU 利用率监控器
struct GpuMonitor {
    last_check: Instant,
    cache_duration: Duration,
    cached_utilization: u8,
}

impl GpuMonitor {
    fn new() -> Self {
        Self {
            last_check: Instant::now() - Duration::from_secs(10),
            cache_duration: Duration::from_secs(2),
            cached_utilization: 0,
        }
    }

    /// 获取 GPU 利用率（带缓存）
    fn get_utilization(&mut self) -> Result<u8> {
        if self.last_check.elapsed() < self.cache_duration {
            return Ok(self.cached_utilization);
        }

        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
            .output()
            .context("无法执行 nvidia-smi")?;

        if !output.status.success() {
            anyhow::bail!("nvidia-smi 执行失败");
        }

        let util_str = String::from_utf8_lossy(&output.stdout);
        let util = util_str
            .trim()
            .parse::<u8>()
            .unwrap_or(0);

        self.cached_utilization = util;
        self.last_check = Instant::now();
        Ok(util)
    }

    /// 强制刷新缓存并获取最新值
    fn force_refresh(&mut self) -> Result<u8> {
        self.last_check = Instant::now() - Duration::from_secs(10);
        self.get_utilization()
    }
}

/// 并发调整动作
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConcurrencyAction {
    Increase,
    Decrease,
    Maintain,
}

/// 并发控制器
struct ConcurrencyController {
    current_max: usize,
    history: Vec<(usize, u8)>,  // (并发数, GPU利用率)
}

impl ConcurrencyController {
    fn new(initial: usize) -> Self {
        Self {
            current_max: initial.clamp(MIN_CONCURRENT, MAX_CONCURRENT),
            history: Vec::new(),
        }
    }

    /// 根据当前 GPU 利用率决定下一步操作
    fn decide(&mut self, current_gpu_util: u8, current_running: usize) -> ConcurrencyAction {
        self.history.push((current_running, current_gpu_util));

        // 如果正在运行的任务数还不到当前最大值，先填满
        if current_running < self.current_max {
            return ConcurrencyAction::Maintain;
        }

        // GPU 过载，减少并发
        if current_gpu_util > UTILIZATION_HIGH {
            if self.current_max > MIN_CONCURRENT {
                self.current_max -= 1;
            }
            return ConcurrencyAction::Decrease;
        }

        // 计算边际增益
        let marginal_gain = self.calculate_marginal_gain();

        if marginal_gain < MARGINAL_GAIN_THRESHOLD {
            // 边际增益太小，保持现状
            return ConcurrencyAction::Maintain;
        }

        // 还有提升空间，尝试增加
        if self.current_max < MAX_CONCURRENT {
            self.current_max += 1;
            return ConcurrencyAction::Increase;
        }

        ConcurrencyAction::Maintain
    }

    /// 计算最近两次的边际增益
    fn calculate_marginal_gain(&self) -> u8 {
        if self.history.len() < 2 {
            return 100; // 数据不足，假设有提升空间
        }

        let len = self.history.len();
        // 找最近两个不同并发数的记录
        let mut prev_idx = len - 1;
        while prev_idx > 0 && self.history[prev_idx].0 == self.history[len - 1].0 {
            prev_idx -= 1;
        }

        if prev_idx == 0 && self.history[0].0 == self.history[len - 1].0 {
            return 100; // 没有可比较的数据
        }

        let (prev_jobs, prev_util) = self.history[prev_idx];
        let (curr_jobs, curr_util) = self.history[len - 1];

        if curr_jobs <= prev_jobs {
            return 100;
        }

        // 每增加一个任务的增益
        let gain = curr_util.saturating_sub(prev_util);
        gain
    }

    fn current_max(&self) -> usize {
        self.current_max
    }
}

/// 探测结果
struct ProbeResult {
    gpu_impact: u8,  // 单任务对 GPU 利用率的影响
}

/// 对第一个视频进行探测，评估单任务 GPU 占用
fn probe_gpu_impact(
    first_video: &Path,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
) -> Result<ProbeResult> {
    let mut monitor = GpuMonitor::new();

    // 记录基线
    let baseline_util = monitor.force_refresh()?;
    println!("  探测: GPU 基线利用率 {}%", baseline_util);

    // 执行一次简短的预览编码
    let start = Instant::now();
    let _ = preview_encode(first_video, 23, encoder_config, audio_bitrate, 5, 0.0);
    let elapsed = start.elapsed();

    // 获取编码期间的 GPU 利用率
    let peak_util = monitor.force_refresh()?;
    let gpu_impact = peak_util.saturating_sub(baseline_util).max(10); // 至少假设 10%

    println!("  探测完成: 耗时 {:?}, GPU 影响 {}%", elapsed, gpu_impact);

    Ok(ProbeResult { gpu_impact })
}

/// 根据探测结果估算初始并发数
fn estimate_initial_concurrency(_probe: &ProbeResult) -> usize {
    // 从 1 个线程开始，让并发控制器根据 GPU 利用率动态调整
    1
}

// ============================================================================
// 任务调度器
// ============================================================================

/// 任务执行结果
#[derive(Debug)]
enum TaskResult {
    Completed { path: PathBuf, stats: FileStats },
    Skipped { path: PathBuf, reason: String },
    Failed { path: PathBuf, error: String },
}

/// 任务调度器（支持动态添加任务）
struct TaskScheduler {
    pending: Mutex<Vec<PathBuf>>,
    running_count: Arc<AtomicUsize>,
    result_sender: Sender<TaskResult>,
    result_receiver: Receiver<TaskResult>,
    encoder_config: EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
    scanning_done: Arc<AtomicBool>,
}

impl TaskScheduler {
    fn new(
        encoder_config: EncoderConfig,
        max_crf: u8,
        min_compression_ratio: u8,
        audio_bitrate: u32,
    ) -> Self {
        let (sender, receiver) = unbounded();
        Self {
            pending: Mutex::new(Vec::new()),
            running_count: Arc::new(AtomicUsize::new(0)),
            result_sender: sender,
            result_receiver: receiver,
            encoder_config,
            max_crf,
            min_compression_ratio,
            audio_bitrate,
            scanning_done: Arc::new(AtomicBool::new(false)),
        }
    }

    /// 标记扫描完成
    fn mark_scanning_done(&self) {
        self.scanning_done.store(true, Ordering::SeqCst);
    }

    /// 动态添加任务
    fn add_task(&self, video_path: PathBuf) {
        self.pending.lock().unwrap().push(video_path);
    }

    /// 获取当前运行中的任务数
    fn running_count(&self) -> usize {
        self.running_count.load(Ordering::SeqCst)
    }

    /// 是否还有未完成的工作
    fn has_pending_or_running(&self) -> bool {
        let pending_empty = self.pending.lock().unwrap().is_empty();
        let running = self.running_count() > 0;
        let done = self.scanning_done.load(Ordering::SeqCst);

        // 如果扫描未完成，总是返回 true（可能还有新任务）
        // 如果扫描完成，检查是否还有待处理或运行中的任务
        !done || !pending_empty || running
    }

    /// 尝试启动新任务，直到达到指定并发数
    fn spawn_up_to(&self, max_concurrent: usize) {
        loop {
            let running = self.running_count();
            if running >= max_concurrent {
                break;
            }

            let video_path = {
                let mut pending = self.pending.lock().unwrap();
                if pending.is_empty() {
                    break;
                }
                pending.remove(0)
            };

            let sender = self.result_sender.clone();
            let running_count = self.running_count.clone();
            let config = self.encoder_config.clone();
            let max_crf = self.max_crf;
            let min_ratio = self.min_compression_ratio;
            let audio_bitrate = self.audio_bitrate;

            running_count.fetch_add(1, Ordering::SeqCst);

            thread::spawn(move || {
                let result = process_video(
                    &video_path,
                    &config,
                    max_crf,
                    min_ratio,
                    audio_bitrate,
                );

                let task_result = match result {
                    Ok(ProcessResult::Converted(stats)) => TaskResult::Completed {
                        path: video_path,
                        stats,
                    },
                    Ok(ProcessResult::Skipped(reason)) => TaskResult::Skipped {
                        path: video_path,
                        reason,
                    },
                    Err(e) => TaskResult::Failed {
                        path: video_path,
                        error: e.to_string(),
                    },
                };

                running_count.fetch_sub(1, Ordering::SeqCst);
                let _ = sender.send(task_result);
            });
        }
    }

    /// 尝试接收已完成的任务结果（非阻塞）
    fn try_recv_result(&self) -> Option<TaskResult> {
        self.result_receiver.try_recv().ok()
    }

    /// 剩余待处理数量
    #[allow(dead_code)]
    fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }
}

/// GPU 并行处理主函数（流式扫描）
fn run_gpu_parallel(
    input_dir: &Path,
    encoder_config: &EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
    results: &mut ConversionStats,
) -> Result<()> {
    // 创建调度器和控制器
    let scheduler = Arc::new(TaskScheduler::new(
        encoder_config.clone(),
        max_crf,
        min_compression_ratio,
        audio_bitrate,
    ));

    // 启动扫描线程
    let scheduler_clone = scheduler.clone();
    let input_dir_clone = input_dir.to_path_buf();
    let video_count = Arc::new(AtomicUsize::new(0));
    let video_count_clone = video_count.clone();

    let scanner_thread = thread::spawn(move || {
        scan_videos_streaming(&input_dir_clone, |path| {
            scheduler_clone.add_task(path);
            video_count_clone.fetch_add(1, Ordering::SeqCst);
        });
        scheduler_clone.mark_scanning_done();
    });

    // 等待第一个视频出现用于探测
    while scheduler.pending_count() == 0 && !scheduler.scanning_done.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(50));
    }

    // 探测阶段：对第一个视频评估 GPU 影响
    let first_video = {
        let pending = scheduler.pending.lock().unwrap();
        pending.first().cloned()
    };

    if first_video.is_none() {
        scanner_thread.join().ok();
        return Ok(());
    }

    println!("=== GPU 并行探测阶段 ===");
    let probe_result = probe_gpu_impact(first_video.as_ref().unwrap(), encoder_config, audio_bitrate)
        .unwrap_or(ProbeResult { gpu_impact: 30 });

    let initial_concurrency = estimate_initial_concurrency(&probe_result);
    println!("初始并发数: {}", initial_concurrency);
    println!("=== 开始并行处理 ===\n");

    let mut controller = ConcurrencyController::new(initial_concurrency);
    let mut monitor = GpuMonitor::new();

    let mut completed_count = 0;

    // 主循环
    while scheduler.has_pending_or_running() {
        // 1. 收集已完成的结果
        while let Some(task_result) = scheduler.try_recv_result() {
            completed_count += 1;
            let total = video_count.load(Ordering::SeqCst);
            match task_result {
                TaskResult::Completed { path, stats } => {
                    results.successful += 1;
                    results.total_input_size += stats.input_size;
                    results.total_output_size += stats.output_size;
                    println!(
                        "[{}/{}] ✓ {} -> {} ({:.1}% 压缩)",
                        completed_count,
                        total,
                        path.display(),
                        format_size(stats.output_size),
                        (1.0 - stats.output_size as f64 / stats.input_size as f64) * 100.0
                    );
                }
                TaskResult::Skipped { path, reason } => {
                    results.skipped += 1;
                    println!("[{}/{}] - 跳过: {} ({})", completed_count, total, path.display(), reason);
                }
                TaskResult::Failed { path, error } => {
                    results.failed += 1;
                    println!("[{}/{}] ✗ 失败: {} ({})", completed_count, total, path.display(), error);
                }
            }
        }

        // 2. 获取 GPU 状态并决定是否调整并发
        let gpu_util = monitor.get_utilization().unwrap_or(0);
        let running = scheduler.running_count();
        let action = controller.decide(gpu_util, running);

        // 3. 根据决策启动任务
        let max_concurrent = controller.current_max();
        scheduler.spawn_up_to(max_concurrent);

        // 打印状态（仅在调整时）
        if action != ConcurrencyAction::Maintain || running == 0 {
            let total = video_count.load(Ordering::SeqCst);
            println!(
                "  [状态] 运行: {} | 并发上限: {} | GPU: {}% | 已发现: {}",
                running, max_concurrent, gpu_util, total
            );
        }

        // 4. 短暂休眠避免忙等待
        thread::sleep(Duration::from_millis(500));
    }

    // 确保扫描线程结束
    scanner_thread.join().ok();

    // 处理剩余的结果（确保所有任务都被统计）
    while let Some(task_result) = scheduler.try_recv_result() {
        completed_count += 1;
        let total = video_count.load(Ordering::SeqCst);
        match task_result {
            TaskResult::Completed { path, stats } => {
                results.successful += 1;
                results.total_input_size += stats.input_size;
                results.total_output_size += stats.output_size;
                println!(
                    "[{}/{}] ✓ {} -> {} ({:.1}% 压缩)",
                    completed_count,
                    total,
                    path.display(),
                    format_size(stats.output_size),
                    (1.0 - stats.output_size as f64 / stats.input_size as f64) * 100.0
                );
            }
            TaskResult::Skipped { path, reason } => {
                results.skipped += 1;
                println!("[{}/{}] - 跳过: {} ({})", completed_count, total, path.display(), reason);
            }
            TaskResult::Failed { path, error } => {
                results.failed += 1;
                println!("[{}/{}] ✗ 失败: {} ({})", completed_count, total, path.display(), error);
            }
        }
    }

    Ok(())
}

/// 视频压缩工具 - 将视频递归转换为H.265格式以减小体积
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// 输入目录路径
    #[arg(short, long)]
    input: PathBuf,

    /// 最大CRF值（质量下限），当CRF=23无法达到压缩目标时，外推的最大CRF限制
    #[arg(long, default_value = "35")]
    max_crf: u8,

    /// 最小压缩比（百分比），必须达到此压缩比才会执行转换
    #[arg(long, default_value = "30")]
    min_compression_ratio: u8,

    /// 视频编码器预设
    #[arg(long, default_value = "medium")]
    preset: String,

    /// 视频编码器类型 (cpu, gpu, auto)
    #[arg(long, default_value = "auto")]
    encoder: String,

    /// 音频编码比特率（kbps），设为0保持原音频
    #[arg(short, long, default_value = "128")]
    audio_bitrate: u32,

    /// 仅扫描不执行转换
    #[arg(long, default_value = "false")]
    dry_run: bool,

    /// 并发处理任务数
    #[arg(short, long, default_value = "1")]
    jobs: usize,

    /// 强制串行处理（禁用GPU并行编码）
    #[arg(long, default_value = "false")]
    serial: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 验证输入目录
    if !args.input.exists() {
        anyhow::bail!("输入目录不存在: {}", args.input.display());
    }

    // 解析编码器类型
    let encoder_type = EncoderType::from_str(&args.encoder)?;
    let encoder_type = encoder_type.resolve()?;
    let encoder_config = EncoderConfig {
        encoder_type,
        preset: args.preset.clone(),
    };
    let encoder_name = match encoder_config.encoder_type {
        EncoderType::Cpu => "CPU (libx265)",
        EncoderType::Gpu => "GPU (hevc_nvenc)",
        EncoderType::Auto => unreachable!("Auto should be resolved"),
    };

    println!("视频压缩工具 (智能压缩模式)");
    println!("输入目录: {}", args.input.display());
    println!("输出模式: 原地替换（后缀名改为.mp4）");
    println!("编码器: {}", encoder_name);
    println!("预览CRF: 23 | 最大CRF: {}", args.max_crf);
    println!("最小压缩比: {}%", args.min_compression_ratio);
    println!("编码预设: {}", args.preset);
    println!("音频比特率: {}kbps", args.audio_bitrate);

    // 检查FFmpeg是否可用
    check_ffmpeg(encoder_config.encoder_type)?;

    // 执行转换
    let mut results = ConversionStats::default();

    if args.dry_run {
        // dry run 模式：流式扫描，只显示不执行
        println!("扫描视频中...\n");
        let mut count = 0;
        scan_videos_streaming(&args.input, |path| {
            count += 1;
            println!("[{}] 处理: {}", count, path.display());
            let output_path = get_output_path(&path);
            println!("  -> (dry run) {}", output_path.display());
        });
        println!("\n共找到 {} 个视频文件", count);
    } else if matches!(encoder_config.encoder_type, EncoderType::Gpu) && !args.serial {
        // GPU 模式：使用并行调度器（流式扫描）
        println!("开始流式扫描和处理...\n");
        run_gpu_parallel(
            &args.input,
            &encoder_config,
            args.max_crf,
            args.min_compression_ratio,
            args.audio_bitrate,
            &mut results,
        )?;
    } else {
        // CPU 模式 或 GPU 串行模式：流式顺序处理
        println!("开始流式扫描和处理...\n");
        let mut count = 0;
        let (sender, receiver) = unbounded::<PathBuf>();

        // 扫描线程
        let input_clone = args.input.clone();
        let scan_thread = thread::spawn(move || {
            scan_videos_streaming(&input_clone, |path| {
                let _ = sender.send(path);
            });
        });

        // 主线程处理
        while let Ok(video_path) = receiver.recv() {
            count += 1;
            println!("[{}] 处理: {}", count, video_path.display());

            match process_video(
                &video_path,
                &encoder_config,
                args.max_crf,
                args.min_compression_ratio,
                args.audio_bitrate,
            ) {
                Ok(ProcessResult::Converted(stats)) => {
                    results.successful += 1;
                    results.total_input_size += stats.input_size;
                    results.total_output_size += stats.output_size;
                    println!(
                        "  ✓ {} -> {} ({:.1}% 压缩)",
                        format_size(stats.input_size),
                        format_size(stats.output_size),
                        (1.0 - stats.output_size as f64 / stats.input_size as f64) * 100.0
                    );
                }
                Ok(ProcessResult::Skipped(reason)) => {
                    results.skipped += 1;
                    println!("  - 跳过: {}", reason);
                }
                Err(e) => {
                    results.failed += 1;
                    println!("  ✗ 失败: {}", e);
                }
            }
        }

        scan_thread.join().ok();
    }

    // 输出统计信息
    println!("\n{}", "=".repeat(50));
    println!("转换完成!");
    println!(
        "成功: {} | 跳过: {} | 失败: {}",
        results.successful, results.skipped, results.failed
    );
    if results.total_input_size > 0 {
        println!(
            "总大小: {} -> {} ({:.1}% 节省)",
            format_size(results.total_input_size),
            format_size(results.total_output_size),
            (1.0 - results.total_output_size as f64 / results.total_input_size as f64) * 100.0
        );
    }
    println!("{}", "=".repeat(50));

    Ok(())
}

#[derive(Debug)]
enum ProcessResult {
    Converted(FileStats),
    Skipped(String),
}

#[derive(Debug, Default)]
struct FileStats {
    input_size: u64,
    output_size: u64,
    #[allow(dead_code)]
    used_crf: u8,
}

#[derive(Debug, Default)]
struct ConversionStats {
    successful: usize,
    skipped: usize,
    failed: usize,
    total_input_size: u64,
    total_output_size: u64,
}

/// 生成输出路径：与输入文件同目录，后缀名改为.mp4
fn get_output_path(input_path: &Path) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or(std::ffi::OsStr::new(""));
    let parent = input_path.parent().unwrap_or(Path::new(""));
    parent.join(format!("{}.mp4", stem.to_string_lossy()))
}

/// 跨文件系统安全的文件移动
/// 先尝试rename，失败则使用copy+remove
fn safe_move_file(from: &Path, to: &Path) -> Result<()> {
    // 先尝试直接重命名（同文件系统最快）
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }

    // rename失败（可能是跨文件系统），使用copy+remove
    std::fs::copy(from, to)
        .context(format!("无法复制文件从 {} 到 {}", from.display(), to.display()))?;
    std::fs::remove_file(from)
        .context(format!("无法删除源文件 {}", from.display()))?;

    Ok(())
}

fn process_video(
    input_path: &Path,
    encoder_config: &EncoderConfig,
    max_crf: u8,
    min_compression_ratio: u8,
    audio_bitrate: u32,
) -> Result<ProcessResult> {
    const PREVIEW_CRF: u8 = 23;  // 固定预览CRF
    const PREVIEW_RATIO: f64 = 0.10;  // 预览时长为视频的10%
    const MIN_PREVIEW_SECONDS: u32 = 5;  // 最短预览5秒

    // 获取输入文件元数据（用于后续恢复）
    let original_metadata = std::fs::metadata(input_path)
        .context("无法读取输入文件元数据")?;
    let original_atime = FileTime::from_last_access_time(&original_metadata);
    let original_mtime = FileTime::from_last_modification_time(&original_metadata);
    let original_permissions = original_metadata.permissions();

    // 获取输入文件大小
    let input_size = original_metadata.len();

    // 获取视频信息
    let video_info = get_video_info(input_path)?;
    let duration = video_info.duration;

    // 判断源视频编码类型
    let is_hevc = video_info
        .codec_name
        .as_deref()
        .map(|c| c.to_lowercase().contains("hevc") || c.to_lowercase().contains("h265"))
        .unwrap_or(false);

    println!(
        "  源编码: {} | 分辨率: {}x{} | 时长: {:.1}s",
        video_info.codec_name.as_deref().unwrap_or("unknown"),
        video_info.width,
        video_info.height,
        duration
    );

    // 如果已经是HEVC，直接跳过
    if is_hevc {
        return Ok(ProcessResult::Skipped("已是H.265编码".to_string()));
    }

    // 计算预览时长（10%时长，最短5秒）
    let preview_duration = if duration > 0.0 {
        let preview_from_ratio = (duration * PREVIEW_RATIO) as u32;
        preview_from_ratio.max(MIN_PREVIEW_SECONDS).min(duration as u32)
    } else {
        MIN_PREVIEW_SECONDS
    };

    println!("  预览编码 ({}秒, CRF={})...", preview_duration, PREVIEW_CRF);

    // 执行预览编码
    let preview_result = preview_encode(
        input_path,
        PREVIEW_CRF,
        encoder_config,
        audio_bitrate,
        preview_duration,
        duration,
    )?;

    // 计算预估压缩比
    // 使用预览输出的每秒大小推算完整视频大小
    // 应用修正因子（预览编码通常比完整编码小约1.8倍）
    const CORRECTION_FACTOR: f64 = 1.8;
    let expected_output_size = preview_result.output_size_per_second * duration * CORRECTION_FACTOR;
    let predicted_ratio = if input_size > 0 {
        1.0 - expected_output_size / input_size as f64
    } else {
        0.0
    };

    let target_ratio = min_compression_ratio as f64 / 100.0;

    println!("  预估压缩比: {:.1}% (目标: {}%)", predicted_ratio * 100.0, min_compression_ratio);

    // 决定使用的CRF
    let final_crf = if predicted_ratio >= target_ratio {
        // 预估压缩比达标，直接使用 CRF=23
        PREVIEW_CRF
    } else {
        // 预估压缩比不达标，外推更高的CRF
        // 使用指数公式: CRF +6 ≈ 比特率减半
        // new_output_ratio = old_output_ratio * 2^((old_crf - new_crf) / 6)
        let predicted_output_ratio = 1.0 - predicted_ratio;  // 输出/输入 比例
        let target_output_ratio = 1.0 - target_ratio;

        if predicted_output_ratio <= 0.0 || target_output_ratio <= 0.0 {
            return Ok(ProcessResult::Skipped("无法计算有效CRF".to_string()));
        }

        // target_output_ratio = predicted_output_ratio * 2^((PREVIEW_CRF - new_crf) / 6)
        // 2^((PREVIEW_CRF - new_crf) / 6) = target_output_ratio / predicted_output_ratio
        // new_crf = PREVIEW_CRF - 6 * log2(target_output_ratio / predicted_output_ratio)
        let ratio = target_output_ratio / predicted_output_ratio;
        let log2_ratio = ratio.log2();
        let crf_delta = 6.0 * log2_ratio;
        let extrapolated_crf = (PREVIEW_CRF as f64 - crf_delta).round() as i32;

        println!("  外推CRF: {} (需要减少输出至 {:.1}%)", extrapolated_crf, target_output_ratio * 100.0);

        // 检查是否超出最大CRF限制
        if extrapolated_crf > max_crf as i32 {
            return Ok(ProcessResult::Skipped(format!(
                "需要CRF={} 超出限制({})",
                extrapolated_crf, max_crf
            )));
        }

        // 确保CRF在有效范围内
        extrapolated_crf.max(PREVIEW_CRF as i32) as u8
    };

    println!("  使用 CRF={} 进行完整编码...", final_crf);

    // 创建临时文件用于编码
    let temp_file = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;
    let temp_path = temp_file.path();

    // 执行完整编码
    let output_size = full_encode(input_path, temp_path, final_crf, encoder_config, audio_bitrate)?;

    let actual_ratio = 1.0 - output_size as f64 / input_size as f64;
    println!(
        "  实际压缩: {} -> {} ({:.1}%)",
        format_size(input_size),
        format_size(output_size),
        actual_ratio * 100.0
    );

    // 生成最终输出路径
    let output_path = get_output_path(input_path);

    // 删除原文件
    std::fs::remove_file(input_path)
        .context("无法删除原文件")?;

    // 将临时文件移动到输出路径
    safe_move_file(temp_path, &output_path)
        .context("无法移动输出文件")?;

    // 恢复时间戳和权限
    set_file_times(&output_path, original_atime, original_mtime)
        .context("无法设置文件时间戳")?;
    std::fs::set_permissions(&output_path, original_permissions)
        .context("无法设置文件权限")?;

    // 消费temp_file避免它被删除
    let _ = temp_file;

    Ok(ProcessResult::Converted(FileStats {
        input_size,
        output_size,
        used_crf: final_crf,
    }))
}

struct PreviewResult {
    output_size_per_second: f64,   // 预览编码后的每秒大小（字节/秒）
}

fn preview_encode(
    input_path: &Path,
    crf: u8,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
    duration: u32,
    video_duration: f64,
) -> Result<PreviewResult> {
    // 创建临时文件
    let temp_output = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;
    let temp_output_path = temp_output.path();

    let segment_paths: Vec<PathBuf>;

    if video_duration <= duration as f64 * 1.5 {
        // 视频太短，从开头直接提取原始片段
        let temp_segment = NamedTempFile::with_suffix(".mp4").context("无法创建临时文件")?;

        let extract_cmd = Command::new("ffmpeg")
            .arg("-y")
            .arg("-i")
            .arg(input_path)
            .arg("-t")
            .arg(duration.to_string())
            .arg("-c")
            .arg("copy")
            .arg(temp_segment.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg提取预览片段失败")?;

        if !extract_cmd.status.success() {
            drop(temp_segment);
            anyhow::bail!("提取预览片段失败");
        }

        segment_paths = vec![temp_segment.path().to_path_buf()];
        // 保持 temp_segment 存活直到编码完成
        let _keep = temp_segment;

        // 对原始片段进行H.265编码（包含音频处理，与完整编码一致）
        let decode_args = build_decode_args(encoder_config);
        let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
        let mut encode_cmd = Command::new("ffmpeg");
        encode_cmd.arg("-y");
        for arg in &decode_args {
            encode_cmd.arg(arg);
        }
        encode_cmd.arg("-i").arg(&segment_paths[0]);
        for arg in video_args {
            encode_cmd.arg(arg);
        }
        encode_cmd.arg("-f").arg("mp4");

        // 音频处理（与完整编码保持一致）
        if audio_bitrate > 0 {
            encode_cmd.arg("-c:a").arg("aac").arg("-b:a").arg(format!("{}k", audio_bitrate));
        } else {
            encode_cmd.arg("-c:a").arg("copy");
        }

        encode_cmd.arg(temp_output_path);

        let output = encode_cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg预览编码失败")?;

        if !output.status.success() {
            anyhow::bail!("预览编码失败");
        }

        drop(_keep);
    } else {
        // 五段均匀采样：在10%、30%、50%、70%、90%位置采样
        const NUM_SEGMENTS: u32 = 5;
        // 采样位置（相对于视频时长的比例）
        const SAMPLE_POSITIONS: [f64; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];

        let segment_duration = duration / NUM_SEGMENTS;
        let remaining = duration % NUM_SEGMENTS;

        // 计算每段的起始时间
        let seg_starts: Vec<f64> = SAMPLE_POSITIONS
            .iter()
            .map(|&pos| (video_duration * pos - segment_duration as f64 / 2.0).max(0.0))
            .collect();

        // 每段时长（最后一段加上余数）
        let seg_durs: Vec<u32> = (0..NUM_SEGMENTS)
            .map(|i| {
                if i == NUM_SEGMENTS - 1 {
                    segment_duration + remaining
                } else {
                    segment_duration
                }
            })
            .collect();

        // 提取五个原始片段到临时文件
        let temp_segs: Vec<NamedTempFile> = (0..NUM_SEGMENTS)
            .map(|i| NamedTempFile::with_suffix(".mp4").context(format!("无法创建临时文件{}", i + 1)))
            .collect::<Result<Vec<_>>>()?;

        // 提取每一段
        let mut extract_success = true;
        for (i, temp_seg) in temp_segs.iter().enumerate() {
            let extract = Command::new("ffmpeg")
                .arg("-y")
                .arg("-ss")
                .arg(format!("{:.1}", seg_starts[i]))
                .arg("-i")
                .arg(input_path)
                .arg("-t")
                .arg(seg_durs[i].to_string())
                .arg("-c")
                .arg("copy")
                .arg(temp_seg.path())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .context(format!("提取片段{}失败", i + 1))?;

            if !extract.status.success() {
                extract_success = false;
                break;
            }
        }

        if !extract_success {
            drop(temp_segs);
            anyhow::bail!("提取多段预览片段失败");
        }

        // 创建concat列表文件
        let concat_list = NamedTempFile::with_suffix(".txt").context("无法创建concat列表")?;
        let concat_content = temp_segs
            .iter()
            .map(|seg| format!("file '{}'\n", seg.path().display()))
            .collect::<String>();
        std::fs::write(concat_list.path(), concat_content).context("无法写入concat列表")?;

        // 使用concat demuxer拼接并编码（包含音频处理，与完整编码一致）
        let decode_args = build_decode_args(encoder_config);
        let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
        let mut encode_cmd = Command::new("ffmpeg");
        encode_cmd.arg("-y");
        for arg in &decode_args {
            encode_cmd.arg(arg);
        }
        encode_cmd
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

        // 音频处理（与完整编码保持一致）
        if audio_bitrate > 0 {
            encode_cmd.arg("-c:a").arg("aac").arg("-b:a").arg(format!("{}k", audio_bitrate));
        } else {
            encode_cmd.arg("-c:a").arg("copy");
        }

        encode_cmd.arg(temp_output_path);

        let output = encode_cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
            .context("执行FFmpeg预览编码失败")?;

        drop(concat_list);
        drop(temp_segs);

        if !output.status.success() {
            anyhow::bail!("预览编码失败");
        }
    }

    // 获取编码后的大小
    let output_size = std::fs::metadata(temp_output_path)
        .context("无法读取预览输出文件")?
        .len() as f64;

    drop(temp_output);

    // 返回每秒输出大小，用于按比例推算完整视频大小
    let output_size_per_second = output_size / duration as f64;

    Ok(PreviewResult {
        output_size_per_second,
    })
}

fn full_encode(
    input_path: &Path,
    output_path: &Path,
    crf: u8,
    encoder_config: &EncoderConfig,
    audio_bitrate: u32,
) -> Result<u64> {
    let decode_args = build_decode_args(encoder_config);
    let video_args = build_encode_args(crf, &encoder_config.preset, encoder_config);
    let mut cmd = Command::new("ffmpeg");
    cmd.arg("-y");
    for arg in decode_args {
        cmd.arg(arg);
    }
    cmd.arg("-i").arg(input_path);
    for arg in video_args {
        cmd.arg(arg);
    }

    // 音频处理
    if audio_bitrate > 0 {
        cmd.arg("-c:a")
            .arg("aac")
            .arg("-b:a")
            .arg(format!("{}k", audio_bitrate));
    } else {
        cmd.arg("-c:a").arg("copy");
    }

    cmd.arg(output_path);

    let output = cmd.output().context("执行FFmpeg失败")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg转换失败: {}", stderr);
    }

    Ok(std::fs::metadata(output_path)
        .context("无法读取输出文件")?
        .len())
}

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
    // Check for hevc_nvenc as an actual encoder (V....D hevc_nvenc)
    let nvenc_available = encoders
        .lines()
        .any(|line| line.trim().starts_with('V') && line.contains("hevc_nvenc"));
    Ok(nvenc_available)
}

fn check_ffmpeg(encoder_type: EncoderType) -> Result<()> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("无法找到ffmpeg命令，请确保已安装FFmpeg")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg执行失败");
    }

    let version = String::from_utf8_lossy(&output.stdout);
    let version_lines: Vec<&str> = version.lines().collect();
    if let Some(line) = version_lines.first() {
        println!("检测到: {}", line);
    }

    // 检查是否支持libx265
    if version.contains("libx265") {
        println!("✓ H.265编码器可用");
    } else {
        anyhow::bail!("FFmpeg未编译libx265支持，无法进行H.265编码");
    }

    // 对于GPU模式，检查NVENC
    if matches!(encoder_type, EncoderType::Gpu) {
        if check_nvenc_available()? {
            println!("✓ NVENC编码器可用");
        } else {
            anyhow::bail!("FFmpeg未编译hevc_nvenc支持，无法使用GPU编码");
        }
    }

    Ok(())
}

/// Build hardware decode arguments for FFmpeg (GPU mode only)
fn build_decode_args(config: &EncoderConfig) -> Vec<String> {
    match config.encoder_type {
        EncoderType::Gpu => vec![
            "-hwaccel".to_string(),
            "cuda".to_string(),
        ],
        _ => vec![],
    }
}

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
            "-cq".to_string(),
            (crf + 7).to_string(),  // GPU CQ = CPU CRF + 7 (NVENC CQ较高质量)
            "-rc-lookahead".to_string(),
            "32".to_string(),
            "-b_ref_mode".to_string(),
            "middle".to_string(),
            "-b".to_string(),
            "4".to_string(),
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

/// 流式扫描视频文件（边扫描边处理回调）
fn scan_videos_streaming<F>(dir: &Path, mut callback: F)
where
    F: FnMut(PathBuf),
{
    let video_extensions = [
        "mp4", "mkv", "avi", "mov", "flv", "wmv", "webm", "m4v", "mpg", "mpeg", "3gp",
    ];

    // 使用 min_depth(1) 避免处理根目录本身
    let mut entries: Vec<_> = WalkDir::new(dir)
        .follow_links(true)
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| video_extensions.contains(&ext.to_lowercase().as_str()))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    // 排序保证处理顺序一致
    entries.sort();

    for path in entries {
        callback(path);
    }
}

#[derive(Debug)]
struct VideoInfo {
    codec_name: Option<String>,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
    duration: f64,
    #[allow(dead_code)]
    bitrate: u64,
}

fn get_video_info(path: &Path) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-select_streams")
        .arg("v:0")
        .arg("-show_entries")
        .arg("stream=codec_name,width,height")
        .arg("-show_entries")
        .arg("format=duration,bit_rate")
        .arg("-of")
        .arg("default=nokey=1:noprint_wrappers=1")
        .arg(path)
        .output()
        .context("执行ffprobe失败")?;

    if !output.status.success() {
        anyhow::bail!("ffprobe执行失败");
    }

    let info = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = info.lines().collect();

    let codec_name = lines.get(0).map(|s| s.to_string());
    let width = lines.get(1).and_then(|s| s.parse().ok()).unwrap_or(1920);
    let height = lines.get(2).and_then(|s| s.parse().ok()).unwrap_or(1080);
    let duration = lines.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let bitrate = lines.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    Ok(VideoInfo {
        codec_name,
        width,
        height,
        duration,
        bitrate,
    })
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
