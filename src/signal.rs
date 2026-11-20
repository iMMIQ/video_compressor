use anyhow::{Context, Result};
use std::path::Path;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn handle_signal(_sig: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

/// Install SIGINT and SIGTERM handlers for graceful shutdown.
pub fn install_signal_handlers() {
    #[cfg(unix)]
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            handle_signal as *const () as libc::sighandler_t,
        );
    }
}

/// Check whether a shutdown signal has been received.
#[inline]
pub fn is_shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::SeqCst)
}

/// Request shutdown programmatically (e.g., in tests).
#[allow(dead_code)]
pub fn request_shutdown() {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

/// Real-time output file size watchdog: kills the encoder immediately if
/// the growing temporary output reaches or exceeds `max_bytes`.
#[derive(Debug, Clone, Copy)]
pub struct SizeWatchdog<'a> {
    pub watch_path: &'a Path,
    pub max_bytes: u64,
}

/// Outcome of a watched external command or two-stage pipeline.
#[derive(Debug)]
pub enum WatchedOutcome {
    Finished(Output),
    SizeLimitExceeded { written_bytes: u64 },
}

/// Run an external command while monitoring the global shutdown flag.
/// If a shutdown signal arrives while the child is running, the child process
/// is terminated cleanly so no orphan ffmpeg/gstreamer processes remain.
pub fn run_command_cancellable(cmd: &mut Command, context_msg: &str) -> Result<Output> {
    match run_command_watched(cmd, None, context_msg)? {
        WatchedOutcome::Finished(out) => Ok(out),
        WatchedOutcome::SizeLimitExceeded { .. } => unreachable!(),
    }
}

/// Run an external command with optional real-time output size watchdog.
pub fn run_command_watched(
    cmd: &mut Command,
    watchdog: Option<SizeWatchdog<'_>>,
    context_msg: &str,
) -> Result<WatchedOutcome> {
    if is_shutdown_requested() {
        anyhow::bail!("收到中断信号，取消执行");
    }

    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = cmd.spawn().with_context(|| context_msg.to_string())?;

    // Drain stdout and stderr on helper threads to prevent pipe buffer deadlocks.
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let stdout_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stdout_pipe {
            let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
        }
        buf
    });

    let stderr_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = stderr_pipe {
            let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
        }
        buf
    });

    let mut tick: u32 = 0;
    let status = loop {
        if let Some(status) = child.try_wait().with_context(|| context_msg.to_string())? {
            break status;
        }
        if is_shutdown_requested() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
            anyhow::bail!("收到中断信号，已终止子进程");
        }
        if let Some(wd) = watchdog
            && tick.is_multiple_of(5)
            && let Ok(meta) = std::fs::metadata(wd.watch_path)
            && meta.len() >= wd.max_bytes
        {
            let written_bytes = meta.len();
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_handle.join();
            let _ = stderr_handle.join();
            return Ok(WatchedOutcome::SizeLimitExceeded { written_bytes });
        }
        tick = tick.wrapping_add(1);
        thread::sleep(Duration::from_millis(50));
    };

    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();

    Ok(WatchedOutcome::Finished(Output {
        status,
        stdout,
        stderr,
    }))
}

/// Run a two-stage OS pipe (`producer | consumer`) directly without a shell,
/// safely handling filenames with spaces/quotes/unicode, SIGINT cancellation,
/// and optional real-time output size watchdog.
pub fn run_pipeline_watched(
    producer: &mut Command,
    consumer: &mut Command,
    watchdog: Option<SizeWatchdog<'_>>,
    context_msg: &str,
) -> Result<WatchedOutcome> {
    if is_shutdown_requested() {
        anyhow::bail!("收到中断信号，取消执行");
    }

    producer
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut prod_child = producer
        .spawn()
        .with_context(|| format!("{} (启动上游解码进程失败)", context_msg))?;

    let prod_stdout = prod_child
        .stdout
        .take()
        .context("无法接管上游进程标准输出管道")?;

    consumer
        .stdin(Stdio::from(prod_stdout))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut cons_child = match consumer.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = prod_child.kill();
            let _ = prod_child.wait();
            return Err(anyhow::Error::new(e)
                .context(format!("{} (启动下游编码进程失败)", context_msg)));
        }
    };

    let prod_stderr_pipe = prod_child.stderr.take();
    let cons_stdout_pipe = cons_child.stdout.take();
    let cons_stderr_pipe = cons_child.stderr.take();

    let prod_err_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = prod_stderr_pipe {
            let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
        }
        buf
    });

    let cons_out_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = cons_stdout_pipe {
            let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
        }
        buf
    });

    let cons_err_handle = thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut pipe) = cons_stderr_pipe {
            let _ = std::io::Read::read_to_end(&mut pipe, &mut buf);
        }
        buf
    });

    let mut prod_status = None;
    let mut cons_status = None;
    let mut tick: u32 = 0;

    while prod_status.is_none() || cons_status.is_none() {
        if prod_status.is_none() {
            prod_status = prod_child
                .try_wait()
                .with_context(|| format!("{} (等待上游进程)", context_msg))?;
        }
        if cons_status.is_none() {
            cons_status = cons_child
                .try_wait()
                .with_context(|| format!("{} (等待下游进程)", context_msg))?;
            // If consumer exited early with failure, terminate producer so it doesn't hang.
            if let Some(st) = cons_status
                && !st.success()
                && prod_status.is_none()
            {
                let _ = prod_child.kill();
            }
        }

        if prod_status.is_some() && cons_status.is_some() {
            break;
        }

        if is_shutdown_requested() {
            let _ = prod_child.kill();
            let _ = cons_child.kill();
            let _ = prod_child.wait();
            let _ = cons_child.wait();
            let _ = prod_err_handle.join();
            let _ = cons_out_handle.join();
            let _ = cons_err_handle.join();
            anyhow::bail!("收到中断信号，已终止管线子进程");
        }

        if let Some(wd) = watchdog
            && tick.is_multiple_of(5)
            && let Ok(meta) = std::fs::metadata(wd.watch_path)
            && meta.len() >= wd.max_bytes
        {
            let written_bytes = meta.len();
            let _ = prod_child.kill();
            let _ = cons_child.kill();
            let _ = prod_child.wait();
            let _ = cons_child.wait();
            let _ = prod_err_handle.join();
            let _ = cons_out_handle.join();
            let _ = cons_err_handle.join();
            return Ok(WatchedOutcome::SizeLimitExceeded { written_bytes });
        }

        tick = tick.wrapping_add(1);
        thread::sleep(Duration::from_millis(50));
    }

    let prod_err = prod_err_handle.join().unwrap_or_default();
    let cons_out = cons_out_handle.join().unwrap_or_default();
    let mut combined_err = cons_err_handle.join().unwrap_or_default();
    if !prod_err.is_empty() {
        if !combined_err.is_empty() {
            combined_err.push(b'\n');
        }
        combined_err.extend_from_slice(&prod_err);
    }

    let p_st = prod_status.unwrap();
    let c_st = cons_status.unwrap();
    // Both must succeed; report failing status if either failed.
    let status = if !c_st.success() { c_st } else { p_st };

    Ok(WatchedOutcome::Finished(Output {
        status,
        stdout: cons_out,
        stderr: combined_err,
    }))
}
