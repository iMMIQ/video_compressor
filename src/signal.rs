use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Output, Stdio};
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

enum WaitOutcome {
    Exited(Vec<ExitStatus>),
    SizeLimitExceeded(u64),
}

fn spawn_pipe_reader<R: Read + Send + 'static>(pipe: Option<R>) -> thread::JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut p) = pipe {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    })
}

fn kill_and_reap_all(children: &mut [&mut Child]) {
    for child in children.iter_mut() {
        let _ = child.kill();
    }
    for child in children.iter_mut() {
        let _ = child.wait();
    }
}

/// Shared polling loop for 1-stage commands and 2-stage pipelines with SIGINT
/// cancellation and optional real-time output size watchdog.
fn wait_children_watched(
    children: &mut [&mut Child],
    watchdog: Option<SizeWatchdog<'_>>,
    context_msg: &str,
) -> Result<WaitOutcome> {
    let mut statuses: Vec<Option<ExitStatus>> = vec![None; children.len()];
    let mut tick: u32 = 0;

    loop {
        for (i, child) in children.iter_mut().enumerate() {
            if statuses[i].is_none() {
                statuses[i] = child.try_wait().with_context(|| context_msg.to_string())?;
            }
        }

        // In a 2-stage pipeline, if the downstream consumer exits early with failure,
        // terminate the upstream producer so it does not hang writing to a broken pipe.
        if children.len() == 2
            && let Some(cons_st) = statuses[1]
            && !cons_st.success()
            && statuses[0].is_none()
        {
            let _ = children[0].kill();
        }

        if statuses.iter().all(Option::is_some) {
            return Ok(WaitOutcome::Exited(
                statuses.into_iter().map(Option::unwrap).collect(),
            ));
        }

        if is_shutdown_requested() {
            kill_and_reap_all(children);
            anyhow::bail!("收到中断信号，已终止子进程");
        }

        if let Some(wd) = watchdog
            && tick.is_multiple_of(5)
            && let Ok(meta) = std::fs::metadata(wd.watch_path)
            && meta.len() >= wd.max_bytes
        {
            let written_bytes = meta.len();
            kill_and_reap_all(children);
            return Ok(WaitOutcome::SizeLimitExceeded(written_bytes));
        }

        tick = tick.wrapping_add(1);
        thread::sleep(Duration::from_millis(50));
    }
}

/// Run an external command while monitoring the global shutdown flag.
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
    let stdout_handle = spawn_pipe_reader(child.stdout.take());
    let stderr_handle = spawn_pipe_reader(child.stderr.take());

    let wait_res = wait_children_watched(&mut [&mut child], watchdog, context_msg);
    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();

    match wait_res? {
        WaitOutcome::SizeLimitExceeded(written_bytes) => {
            Ok(WatchedOutcome::SizeLimitExceeded { written_bytes })
        }
        WaitOutcome::Exited(statuses) => Ok(WatchedOutcome::Finished(Output {
            status: statuses[0],
            stdout,
            stderr,
        })),
    }
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
            kill_and_reap_all(&mut [&mut prod_child]);
            return Err(
                anyhow::Error::new(e).context(format!("{} (启动下游编码进程失败)", context_msg))
            );
        }
    };

    let prod_err_handle = spawn_pipe_reader(prod_child.stderr.take());
    let cons_out_handle = spawn_pipe_reader(cons_child.stdout.take());
    let cons_err_handle = spawn_pipe_reader(cons_child.stderr.take());

    let wait_res = wait_children_watched(
        &mut [&mut prod_child, &mut cons_child],
        watchdog,
        context_msg,
    );
    let prod_err = prod_err_handle.join().unwrap_or_default();
    let cons_out = cons_out_handle.join().unwrap_or_default();
    let mut combined_err = cons_err_handle.join().unwrap_or_default();

    match wait_res? {
        WaitOutcome::SizeLimitExceeded(written_bytes) => {
            Ok(WatchedOutcome::SizeLimitExceeded { written_bytes })
        }
        WaitOutcome::Exited(statuses) => {
            if !prod_err.is_empty() {
                if !combined_err.is_empty() {
                    combined_err.push(b'\n');
                }
                combined_err.extend_from_slice(&prod_err);
            }
            let status = if !statuses[1].success() {
                statuses[1]
            } else {
                statuses[0]
            };
            Ok(WatchedOutcome::Finished(Output {
                status,
                stdout: cons_out,
                stderr: combined_err,
            }))
        }
    }
}
