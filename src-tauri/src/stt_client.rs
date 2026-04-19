use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{mpsc, oneshot, watch},
    time::{sleep, Duration},
};

const STT_TIMEOUT: Duration = Duration::from_secs(45);

#[derive(Debug)]
pub enum SttClientError {
    Cancelled,
    Config(String),
    Io(String),
    Protocol(String),
    Timeout,
    WorkerFailed(String),
    WorkerUnavailable,
}

impl SttClientError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    fn should_restart_worker(&self) -> bool {
        matches!(
            self,
            Self::Cancelled
                | Self::Io(_)
                | Self::Protocol(_)
                | Self::Timeout
                | Self::WorkerUnavailable
        )
    }
}

impl fmt::Display for SttClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(formatter, "STT request cancelled"),
            Self::Config(message) => write!(formatter, "STT configuration error: {message}"),
            Self::Io(message) => write!(formatter, "STT I/O error: {message}"),
            Self::Protocol(message) => write!(formatter, "STT protocol error: {message}"),
            Self::Timeout => write!(formatter, "STT worker timed out"),
            Self::WorkerFailed(message) => write!(formatter, "STT worker failed: {message}"),
            Self::WorkerUnavailable => write!(formatter, "STT worker unavailable"),
        }
    }
}

#[derive(Clone)]
pub struct SttClient {
    tx: mpsc::Sender<SttCommand>,
    generation: Arc<AtomicU64>,
    cancel_tx: watch::Sender<u64>,
}

impl SttClient {
    pub fn new(project_root: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel(32);
        let (cancel_tx, cancel_rx) = watch::channel(0);
        let generation = Arc::new(AtomicU64::new(0));

        tauri::async_runtime::spawn(async move {
            SttActor {
                project_root,
                rx,
                cancel_rx,
                worker: None,
            }
            .run()
            .await;
        });

        Self {
            tx,
            generation,
            cancel_tx,
        }
    }

    pub fn cancel_all(&self) {
        let next_generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let _ = self.cancel_tx.send(next_generation);
    }

    pub async fn transcribe(&self, audio_path: &Path) -> Result<String, SttClientError> {
        let generation = self.generation.load(Ordering::SeqCst);
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(SttCommand::Transcribe(SttJob {
                audio_path: audio_path.to_path_buf(),
                generation,
                response_tx,
            }))
            .await
            .map_err(|_| SttClientError::WorkerUnavailable)?;

        response_rx
            .await
            .map_err(|_| SttClientError::WorkerUnavailable)?
    }
}

enum SttCommand {
    Transcribe(SttJob),
}

struct SttJob {
    audio_path: PathBuf,
    generation: u64,
    response_tx: oneshot::Sender<Result<String, SttClientError>>,
}

struct SttActor {
    project_root: PathBuf,
    rx: mpsc::Receiver<SttCommand>,
    cancel_rx: watch::Receiver<u64>,
    worker: Option<WorkerProcess>,
}

impl SttActor {
    async fn run(&mut self) {
        while let Some(command) = self.rx.recv().await {
            match command {
                SttCommand::Transcribe(job) => {
                    let result = self.handle_job(&job).await;
                    let should_restart = result
                        .as_ref()
                        .err()
                        .is_some_and(SttClientError::should_restart_worker);

                    let _ = job.response_tx.send(result);

                    if should_restart {
                        self.stop_worker().await;
                    }
                }
            }
        }

        self.stop_worker().await;
    }

    async fn handle_job(&mut self, job: &SttJob) -> Result<String, SttClientError> {
        let active_generation = *self.cancel_rx.borrow_and_update();
        if job.generation != active_generation {
            return Err(SttClientError::Cancelled);
        }

        if self.worker.is_none() {
            self.worker = Some(start_worker(&self.project_root).await?);
        }

        let worker = self
            .worker
            .as_mut()
            .ok_or(SttClientError::WorkerUnavailable)?;

        let request = WorkerRequest {
            audio_path: job.audio_path.to_string_lossy().to_string(),
        };

        let line = serde_json::to_string(&request)
            .map_err(|error| SttClientError::Protocol(error.to_string()))?;

        worker
            .stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|error| SttClientError::Io(format!("worker stdin write failed: {error}")))?;
        worker
            .stdin
            .write_all(b"\n")
            .await
            .map_err(|error| SttClientError::Io(format!("worker stdin newline failed: {error}")))?;
        worker
            .stdin
            .flush()
            .await
            .map_err(|error| SttClientError::Io(format!("worker stdin flush failed: {error}")))?;

        self.wait_for_response(job).await
    }

    async fn wait_for_response(&mut self, job: &SttJob) -> Result<String, SttClientError> {
        let worker = self
            .worker
            .as_mut()
            .ok_or(SttClientError::WorkerUnavailable)?;
        let WorkerProcess { child, stdout, .. } = worker;
        let requested_path = job.audio_path.to_string_lossy().to_string();

        loop {
            tokio::select! {
                changed = self.cancel_rx.changed() => {
                    if changed.is_ok() {
                        let _ = child.kill().await;
                    }
                    return Err(SttClientError::Cancelled);
                }
                _ = sleep(STT_TIMEOUT) => {
                    let _ = child.kill().await;
                    return Err(SttClientError::Timeout);
                }
                line = read_worker_line(stdout) => {
                    let Some(line) = line? else {
                        return Err(SttClientError::WorkerUnavailable);
                    };

                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let response = match serde_json::from_str::<WorkerResponse>(trimmed) {
                        Ok(response) => response,
                        Err(_) => {
                            eprintln!("STT worker stdout ignored: {trimmed}");
                            continue;
                        }
                    };

                    if response.audio_path != requested_path {
                        continue;
                    }

                    if !response.ok {
                        return Err(SttClientError::WorkerFailed(
                            response.error.unwrap_or_else(|| "unknown worker error".to_string()),
                        ));
                    }

                    return Ok(response.text.unwrap_or_default().trim().to_string());
                }
            }
        }
    }

    async fn stop_worker(&mut self) {
        if let Some(mut worker) = self.worker.take() {
            let _ = worker.child.kill().await;
            let _ = worker.child.wait().await;
        }
    }
}

async fn start_worker(project_root: &PathBuf) -> Result<WorkerProcess, SttClientError> {
    let python = project_root
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    let worker_path = project_root
        .join("python_services")
        .join("stt")
        .join("stt_worker.py");

    if !python.exists() {
        return Err(SttClientError::Config(format!(
            "python.exe not found: {}",
            python.display()
        )));
    }

    if !worker_path.exists() {
        return Err(SttClientError::Config(format!(
            "stt_worker.py not found: {}",
            worker_path.display()
        )));
    }

    let mut child = Command::new(&python)
        .current_dir(project_root)
        .arg("-m")
        .arg("python_services.stt.stt_worker")
        .arg("--server")
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8:replace")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| SttClientError::Io(format!("worker spawn failed: {error}")))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| SttClientError::Io("worker stdin unavailable".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| SttClientError::Io("worker stdout unavailable".to_string()))?;

    Ok(WorkerProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Debug, Serialize)]
struct WorkerRequest {
    audio_path: String,
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    ok: bool,
    audio_path: String,
    text: Option<String>,
    error: Option<String>,
}

async fn read_worker_line(
    stdout: &mut BufReader<ChildStdout>,
) -> Result<Option<String>, SttClientError> {
    let mut bytes = Vec::new();
    let read = stdout
        .read_until(b'\n', &mut bytes)
        .await
        .map_err(|error| SttClientError::Io(format!("worker stdout read failed: {error}")))?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(decode_worker_stdout_line(&bytes)))
}

fn decode_worker_stdout_line(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim_end_matches(['\r', '\n']).to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn invalid_utf8_worker_stdout_is_lossy_decoded() {
        let decoded = super::decode_worker_stdout_line(b"{\"ok\":false}\xff\n");
        assert!(decoded.starts_with("{\"ok\":false}"));
    }
}
