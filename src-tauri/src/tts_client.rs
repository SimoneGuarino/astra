use crate::speech_events::AudioSegmentReadyEvent;
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::PathBuf,
    process::Stdio,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{mpsc, oneshot, watch},
    time::{sleep, Duration},
};
use uuid::Uuid;

const TTS_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug)]
pub enum TtsClientError {
    Cancelled,
    Config(String),
    Io(String),
    Protocol(String),
    Timeout,
    WorkerFailed(String),
    WorkerUnavailable,
}

impl TtsClientError {
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

impl fmt::Display for TtsClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(formatter, "TTS request cancelled"),
            Self::Config(message) => write!(formatter, "TTS configuration error: {message}"),
            Self::Io(message) => write!(formatter, "TTS I/O error: {message}"),
            Self::Protocol(message) => write!(formatter, "TTS protocol error: {message}"),
            Self::Timeout => write!(formatter, "TTS worker timed out"),
            Self::WorkerFailed(message) => write!(formatter, "TTS worker failed: {message}"),
            Self::WorkerUnavailable => write!(formatter, "TTS worker unavailable"),
        }
    }
}

#[derive(Clone)]
pub struct TtsClient {
    tx: mpsc::Sender<TtsCommand>,
    generation: Arc<AtomicU64>,
    cancel_tx: watch::Sender<u64>,
}

impl TtsClient {
    pub fn new(project_root: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel(64);
        let (cancel_tx, cancel_rx) = watch::channel(0);
        let generation = Arc::new(AtomicU64::new(0));

        tauri::async_runtime::spawn(async move {
            TtsActor {
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

    pub async fn synthesize(
        &self,
        request_id: String,
        segment_id: String,
        sequence: u32,
        text: String,
    ) -> Result<AudioSegmentReadyEvent, TtsClientError> {
        let generation = self.generation.load(Ordering::SeqCst);
        let (response_tx, response_rx) = oneshot::channel();

        let job = TtsJob {
            request_id,
            segment_id,
            sequence,
            text,
            generation,
            response_tx,
        };

        self.tx
            .send(TtsCommand::Synthesize(job))
            .await
            .map_err(|_| TtsClientError::WorkerUnavailable)?;

        response_rx
            .await
            .map_err(|_| TtsClientError::WorkerUnavailable)?
    }
}

enum TtsCommand {
    Synthesize(TtsJob),
}

struct TtsJob {
    request_id: String,
    segment_id: String,
    sequence: u32,
    text: String,
    generation: u64,
    response_tx: oneshot::Sender<Result<AudioSegmentReadyEvent, TtsClientError>>,
}

struct TtsActor {
    project_root: PathBuf,
    rx: mpsc::Receiver<TtsCommand>,
    cancel_rx: watch::Receiver<u64>,
    worker: Option<WorkerProcess>,
}

impl TtsActor {
    async fn run(&mut self) {
        while let Some(command) = self.rx.recv().await {
            match command {
                TtsCommand::Synthesize(job) => {
                    let result = self.handle_job(&job).await;
                    let should_restart = result
                        .as_ref()
                        .err()
                        .is_some_and(TtsClientError::should_restart_worker);

                    let _ = job.response_tx.send(result);

                    if should_restart {
                        self.stop_worker().await;
                    }
                }
            }
        }

        self.stop_worker().await;
    }

    async fn handle_job(&mut self, job: &TtsJob) -> Result<AudioSegmentReadyEvent, TtsClientError> {
        let active_generation = *self.cancel_rx.borrow_and_update();
        if job.generation != active_generation {
            return Err(TtsClientError::Cancelled);
        }

        let output_dir = self
            .project_root
            .join("python_services")
            .join("tts")
            .join("generated");

        std::fs::create_dir_all(&output_dir)
            .map_err(|error| TtsClientError::Io(format!("create output dir failed: {error}")))?;

        let output_path = output_dir.join(format!(
            "tts_{}_{}_{}.wav",
            job.request_id,
            job.segment_id,
            Uuid::new_v4()
        ));

        if self.worker.is_none() {
            self.worker = Some(start_worker(&self.project_root).await?);
        }

        let worker = self
            .worker
            .as_mut()
            .ok_or(TtsClientError::WorkerUnavailable)?;
        let worker_request = WorkerRequest {
            request_id: job.request_id.clone(),
            segment_id: job.segment_id.clone(),
            sequence: job.sequence,
            text: job.text.clone(),
            output_path: output_path.to_string_lossy().to_string(),
        };

        let line = serde_json::to_string(&worker_request)
            .map_err(|error| TtsClientError::Protocol(error.to_string()))?;

        worker
            .stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|error| TtsClientError::Io(format!("worker stdin write failed: {error}")))?;
        worker
            .stdin
            .write_all(b"\n")
            .await
            .map_err(|error| TtsClientError::Io(format!("worker stdin newline failed: {error}")))?;
        worker
            .stdin
            .flush()
            .await
            .map_err(|error| TtsClientError::Io(format!("worker stdin flush failed: {error}")))?;

        self.wait_for_response(job, output_path.to_string_lossy().to_string())
            .await
    }

    async fn wait_for_response(
        &mut self,
        job: &TtsJob,
        fallback_output_path: String,
    ) -> Result<AudioSegmentReadyEvent, TtsClientError> {
        let worker = self
            .worker
            .as_mut()
            .ok_or(TtsClientError::WorkerUnavailable)?;
        let WorkerProcess { child, stdout, .. } = worker;

        loop {
            tokio::select! {
                changed = self.cancel_rx.changed() => {
                    if changed.is_ok() {
                        let _ = child.kill().await;
                    }
                    return Err(TtsClientError::Cancelled);
                }
                _ = sleep(TTS_TIMEOUT) => {
                    let _ = child.kill().await;
                    return Err(TtsClientError::Timeout);
                }
                line = stdout.next_line() => {
                    let line = line
                        .map_err(|error| TtsClientError::Io(format!("worker stdout read failed: {error}")))?;

                    let Some(line) = line else {
                        return Err(TtsClientError::WorkerUnavailable);
                    };

                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let parsed = serde_json::from_str::<WorkerResponse>(trimmed);
                    let response = match parsed {
                        Ok(response) => response,
                        Err(_) => {
                            eprintln!("TTS worker stdout ignored: {trimmed}");
                            continue;
                        }
                    };

                    if response.request_id != job.request_id || response.segment_id != job.segment_id {
                        continue;
                    }

                    if !response.ok {
                        return Err(TtsClientError::WorkerFailed(
                            response.error.unwrap_or_else(|| "unknown worker error".to_string()),
                        ));
                    }

                    return Ok(AudioSegmentReadyEvent {
                        request_id: job.request_id.clone(),
                        segment_id: job.segment_id.clone(),
                        sequence: job.sequence,
                        output_path: response.output_path.unwrap_or(fallback_output_path),
                        text: job.text.clone(),
                    });
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

async fn start_worker(project_root: &PathBuf) -> Result<WorkerProcess, TtsClientError> {
    let python = project_root
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    let worker_path = project_root
        .join("python_services")
        .join("tts")
        .join("tts_worker.py");

    if !python.exists() {
        return Err(TtsClientError::Config(format!(
            "python.exe not found: {}",
            python.display()
        )));
    }

    if !worker_path.exists() {
        return Err(TtsClientError::Config(format!(
            "tts_worker.py not found: {}",
            worker_path.display()
        )));
    }

    let mut child = Command::new(&python)
        .arg(&worker_path)
        .arg("--server")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| TtsClientError::Io(format!("worker spawn failed: {error}")))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| TtsClientError::Io("worker stdin unavailable".to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| TtsClientError::Io("worker stdout unavailable".to_string()))?;

    Ok(WorkerProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout).lines(),
    })
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

#[derive(Debug, Serialize)]
struct WorkerRequest {
    request_id: String,
    segment_id: String,
    sequence: u32,
    text: String,
    output_path: String,
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    ok: bool,
    request_id: String,
    segment_id: String,
    output_path: Option<String>,
    error: Option<String>,
}
