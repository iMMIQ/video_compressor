use std::io::{self, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

/// Runtime log verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LogLevel {
    Quiet = 0,
    Info = 1,
    Debug = 2,
}

impl LogLevel {
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.to_lowercase().as_str() {
            "quiet" | "error" => Ok(Self::Quiet),
            "info" => Ok(Self::Info),
            "debug" | "verbose" => Ok(Self::Debug),
            _ => anyhow::bail!("无效的日志级别: {} (可选: quiet, info, debug)", s),
        }
    }
}

static CURRENT_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Info as u8);
static PRINT_LOCK: Mutex<()> = Mutex::new(());

pub fn set_log_level(level: LogLevel) {
    CURRENT_LEVEL.store(level as u8, Ordering::Relaxed);
}

#[inline]
pub fn current_level() -> LogLevel {
    match CURRENT_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Quiet,
        2 => LogLevel::Debug,
        _ => LogLevel::Info,
    }
}

/// Thread-safe line output to stdout (prevents interleaved lines from concurrent workers).
pub fn log_info(msg: &str) {
    if current_level() >= LogLevel::Info {
        let _guard = PRINT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "{}", msg);
    }
}

/// Thread-safe debug output to stdout.
pub fn log_debug(msg: &str) {
    if current_level() >= LogLevel::Debug {
        let _guard = PRINT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = io::stdout().lock();
        let _ = writeln!(out, "[DEBUG] {}", msg);
    }
}

/// Thread-safe error output to stderr (always printed even in quiet mode).
pub fn log_error(msg: &str) {
    let _guard = PRINT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut err = io::stderr().lock();
    let _ = writeln!(err, "{}", msg);
}

/// Always print summary lines regardless of log level (unless quiet? Still useful for summaries).
pub fn log_always(msg: &str) {
    let _guard = PRINT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = io::stdout().lock();
    let _ = writeln!(out, "{}", msg);
}
