use serde::Deserialize;
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
    process::Command,
    time::{timeout, Duration},
};

const STT_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Debug)]
pub enum SttClientError {
    Cancelled,
    Config(String),
    Io(String),
    Protocol(String),
    Timeout,
    WorkerFailed(String),
}

impl SttClientError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
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
        }
    }
}

#[derive(Clone)]
pub struct SttClient {
    project_root: PathBuf,
    generation: Arc<AtomicU64>,
}

impl SttClient {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            project_root,
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn cancel_all(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    pub async fn transcribe(&self, audio_path: &Path) -> Result<String, SttClientError> {
        let generation = self.generation.load(Ordering::SeqCst);
        let python = self
            .project_root
            .join(".venv")
            .join("Scripts")
            .join("python.exe");
        let worker_path = self
            .project_root
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

        if generation != self.generation.load(Ordering::SeqCst) {
            return Err(SttClientError::Cancelled);
        }

        let output = timeout(
            STT_TIMEOUT,
            Command::new(&python)
                .arg(&worker_path)
                .arg(audio_path)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|_| SttClientError::Timeout)?
        .map_err(|error| SttClientError::Io(format!("worker spawn failed: {error}")))?;

        if generation != self.generation.load(Ordering::SeqCst) {
            return Err(SttClientError::Cancelled);
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(SttClientError::WorkerFailed(if stderr.is_empty() {
                format!("worker exited with status {}", output.status)
            } else {
                stderr
            }));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if stdout.is_empty() {
            return Err(SttClientError::Protocol(
                "worker returned empty stdout".to_string(),
            ));
        }

        let response: WorkerResponse = serde_json::from_str(&stdout)
            .map_err(|error| SttClientError::Protocol(error.to_string()))?;

        if let Some(error) = response.error {
            return Err(SttClientError::WorkerFailed(error));
        }

        Ok(response.text.unwrap_or_default().trim().to_string())
    }
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    text: Option<String>,
    error: Option<String>,
}
