use anyhow::{Context, Result};
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

/// Run an external command while monitoring the global shutdown flag.
/// If a shutdown signal arrives while the child is running, the child process
/// is terminated cleanly so no orphan ffmpeg processes remain.
pub fn run_command_cancellable(cmd: &mut Command, context_msg: &str) -> Result<Output> {
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
        thread::sleep(Duration::from_millis(50));
    };

    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();

    Ok(Output {
        status,
        stdout,
        stderr,
    })
}
