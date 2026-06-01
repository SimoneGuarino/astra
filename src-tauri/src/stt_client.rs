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

const DEFAULT_STT_TIMEOUT_SECS: u64 = 45;
const DEFAULT_MEETING_STT_TIMEOUT_SECS: u64 = 120;
const DEFAULT_STT_STARTUP_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MEETING_STT_STARTUP_TIMEOUT_SECS: u64 = 300;
const MIN_STT_TIMEOUT_SECS: u64 = 10;
const MAX_STT_TIMEOUT_SECS: u64 = 600;
const DEFAULT_MEETING_STT_MODEL: &str = "tiny";

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
        Self::with_worker_environment(project_root, SttWorkerEnvironment::default())
    }

    pub fn new_for_meeting(project_root: PathBuf) -> Self {
        Self::with_worker_environment(project_root, SttWorkerEnvironment::meeting_default())
    }

    fn with_worker_environment(
        project_root: PathBuf,
        worker_environment: SttWorkerEnvironment,
    ) -> Self {
        let (tx, rx) = mpsc::channel(32);
        let (cancel_tx, cancel_rx) = watch::channel(0);
        let generation = Arc::new(AtomicU64::new(0));

        tauri::async_runtime::spawn(async move {
            SttActor {
                project_root,
                request_timeout: worker_environment.timeout,
                startup_timeout: worker_environment.startup_timeout,
                worker_environment,
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

    pub fn request_warm_up(&self) -> Result<(), SttClientError> {
        let generation = self.generation.load(Ordering::SeqCst);
        self.tx
            .try_send(SttCommand::WarmUp { generation })
            .map_err(|_| SttClientError::WorkerUnavailable)
    }

    pub async fn warm_up(&self) -> Result<(), SttClientError> {
        let generation = self.generation.load(Ordering::SeqCst);
        let (response_tx, response_rx) = oneshot::channel();

        self.tx
            .send(SttCommand::WarmUpWait {
                generation,
                response_tx,
            })
            .await
            .map_err(|_| SttClientError::WorkerUnavailable)?;

        response_rx
            .await
            .map_err(|_| SttClientError::WorkerUnavailable)?
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

#[derive(Debug, Clone)]
struct SttWorkerEnvironment {
    values: Vec<(String, String)>,
    timeout: Duration,
    startup_timeout: Duration,
}

impl Default for SttWorkerEnvironment {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            timeout: timeout_from_lookup(
                |key| std::env::var(key).ok(),
                "ASTRA_STT_TIMEOUT_SECS",
                DEFAULT_STT_TIMEOUT_SECS,
            ),
            startup_timeout: timeout_from_lookup(
                |key| std::env::var(key).ok(),
                "ASTRA_STT_STARTUP_TIMEOUT_SECS",
                DEFAULT_STT_STARTUP_TIMEOUT_SECS,
            ),
        }
    }
}

impl SttWorkerEnvironment {
    fn meeting_default() -> Self {
        Self::meeting_from_lookup(|key| std::env::var(key).ok())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn meeting_from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Self {
        let gpu_policy = lookup("ASTRA_MEETING_STT_GPU_POLICY")
            .map(|value| value.trim().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| "prefer_cpu_when_llm_gpu".to_string());
        let default_device = match gpu_policy.as_str() {
            "force_cuda" => "cuda",
            "auto" => "auto",
            _ => "cpu",
        };
        let device = lookup("ASTRA_MEETING_STT_DEVICE")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| default_device.to_string());
        let compute_type = lookup("ASTRA_MEETING_STT_COMPUTE_TYPE")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| if device == "cpu" { "int8" } else { "auto" }.to_string());
        let model = lookup("ASTRA_MEETING_STT_MODEL")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_MEETING_STT_MODEL.to_string());
        let timeout = timeout_from_dual_lookup(
            &mut lookup,
            "ASTRA_MEETING_STT_TIMEOUT_SECS",
            "ASTRA_STT_TIMEOUT_SECS",
            DEFAULT_MEETING_STT_TIMEOUT_SECS,
        );
        let startup_timeout = timeout_from_dual_lookup(
            &mut lookup,
            "ASTRA_MEETING_STT_STARTUP_TIMEOUT_SECS",
            "ASTRA_STT_STARTUP_TIMEOUT_SECS",
            DEFAULT_MEETING_STT_STARTUP_TIMEOUT_SECS,
        );
        let mut values = vec![
            ("ASTRA_STT_DEVICE".to_string(), device),
            ("ASTRA_STT_COMPUTE_TYPE".to_string(), compute_type),
            ("ASTRA_STT_MODEL".to_string(), model),
        ];
        if let Some(language) = lookup("ASTRA_MEETING_STT_LANGUAGE")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        {
            values.push(("ASTRA_STT_LANGUAGE".to_string(), language));
        }
        Self {
            values,
            timeout,
            startup_timeout,
        }
    }
}

fn timeout_from_lookup(
    mut lookup: impl FnMut(&str) -> Option<String>,
    key: &str,
    default_secs: u64,
) -> Duration {
    let seconds = lookup(key)
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| (MIN_STT_TIMEOUT_SECS..=MAX_STT_TIMEOUT_SECS).contains(value))
        .unwrap_or(default_secs);
    Duration::from_secs(seconds)
}

fn timeout_from_dual_lookup(
    lookup: &mut impl FnMut(&str) -> Option<String>,
    primary_key: &str,
    fallback_key: &str,
    default_secs: u64,
) -> Duration {
    let seconds = lookup(primary_key)
        .or_else(|| lookup(fallback_key))
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| (MIN_STT_TIMEOUT_SECS..=MAX_STT_TIMEOUT_SECS).contains(value))
        .unwrap_or(default_secs);
    Duration::from_secs(seconds)
}

enum SttCommand {
    WarmUp { generation: u64 },
    WarmUpWait {
        generation: u64,
        response_tx: oneshot::Sender<Result<(), SttClientError>>,
    },
    Transcribe(SttJob),
}

struct SttJob {
    audio_path: PathBuf,
    generation: u64,
    response_tx: oneshot::Sender<Result<String, SttClientError>>,
}

struct SttActor {
    project_root: PathBuf,
    request_timeout: Duration,
    startup_timeout: Duration,
    worker_environment: SttWorkerEnvironment,
    rx: mpsc::Receiver<SttCommand>,
    cancel_rx: watch::Receiver<u64>,
    worker: Option<WorkerProcess>,
}

impl SttActor {
    async fn run(&mut self) {
        while let Some(command) = self.rx.recv().await {
            match command {
                SttCommand::WarmUp { generation } => {
                    let result = self.handle_warm_up(generation).await;
                    if result
                        .as_ref()
                        .err()
                        .is_some_and(SttClientError::should_restart_worker)
                    {
                        self.stop_worker().await;
                    }
                }
                SttCommand::WarmUpWait {
                    generation,
                    response_tx,
                } => {
                    let result = self.handle_warm_up(generation).await;
                    let should_restart = result
                        .as_ref()
                        .err()
                        .is_some_and(SttClientError::should_restart_worker);

                    let _ = response_tx.send(result);

                    if should_restart {
                        self.stop_worker().await;
                    }
                }
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

    async fn handle_warm_up(&mut self, generation: u64) -> Result<(), SttClientError> {
        let active_generation = *self.cancel_rx.borrow_and_update();
        if generation != active_generation {
            return Err(SttClientError::Cancelled);
        }

        self.ensure_worker_ready().await
    }

    async fn handle_job(&mut self, job: &SttJob) -> Result<String, SttClientError> {
        let active_generation = *self.cancel_rx.borrow_and_update();
        if job.generation != active_generation {
            return Err(SttClientError::Cancelled);
        }

        self.ensure_worker_ready().await?;

        let audio_path = job.audio_path.to_string_lossy().to_string();
        let request_id = next_request_id("transcribe");
        let request = WorkerRequest {
            request_id: request_id.clone(),
            kind: WorkerRequestKind::Transcribe,
            audio_path: Some(audio_path.clone()),
        };

        self.send_worker_request(&request).await?;

        self.wait_for_response(WorkerResponseExpectation {
            request_id,
            audio_path: Some(audio_path),
            timeout: self.request_timeout,
        })
        .await
        .map(|response| response.text.unwrap_or_default().trim().to_string())
    }

    async fn ensure_worker_ready(&mut self) -> Result<(), SttClientError> {
        if self.worker.is_none() {
            self.worker = Some(start_worker(&self.project_root, &self.worker_environment).await?);
        }

        if self
            .worker
            .as_ref()
            .ok_or(SttClientError::WorkerUnavailable)?
            .ready
        {
            return Ok(());
        }

        let request_id = next_request_id("warmup");
        let request = WorkerRequest {
            request_id: request_id.clone(),
            kind: WorkerRequestKind::WarmUp,
            audio_path: None,
        };
        self.send_worker_request(&request).await?;
        let _ = self
            .wait_for_response(WorkerResponseExpectation {
                request_id,
                audio_path: None,
                timeout: self.startup_timeout,
            })
            .await?;

        if let Some(worker) = self.worker.as_mut() {
            worker.ready = true;
        }
        Ok(())
    }

    async fn send_worker_request(&mut self, request: &WorkerRequest) -> Result<(), SttClientError> {
        let worker = self
            .worker
            .as_mut()
            .ok_or(SttClientError::WorkerUnavailable)?;
        let line = serde_json::to_string(request)
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
        Ok(())
    }

    async fn wait_for_response(
        &mut self,
        expectation: WorkerResponseExpectation,
    ) -> Result<WorkerResponse, SttClientError> {
        let worker = self
            .worker
            .as_mut()
            .ok_or(SttClientError::WorkerUnavailable)?;
        let WorkerProcess { child, stdout, .. } = worker;
        let mut ignored_json_lines = 0usize;

        loop {
            tokio::select! {
                changed = self.cancel_rx.changed() => {
                    if changed.is_ok() {
                        let _ = child.kill().await;
                    }
                    return Err(SttClientError::Cancelled);
                }
                _ = sleep(expectation.timeout) => {
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
                            ignored_json_lines += 1;
                            eprintln!(
                                "STT worker stdout ignored: non-response JSON/text line count={ignored_json_lines}"
                            );
                            continue;
                        }
                    };

                    if let Some(response_request_id) = response.request_id.as_deref() {
                        if response_request_id != expectation.request_id {
                            return Err(SttClientError::Protocol(
                                "worker response request_id mismatch".to_string(),
                            ));
                        }
                    } else if let Some(expected_audio_path) = expectation.audio_path.as_deref() {
                        if response.audio_path.as_deref() != Some(expected_audio_path) {
                            return Err(SttClientError::Protocol(
                                "worker response audio_path mismatch".to_string(),
                            ));
                        }
                    }

                    if !response.ok {
                        return Err(SttClientError::WorkerFailed(
                            response.error.unwrap_or_else(|| "unknown worker error".to_string()),
                        ));
                    }

                    return Ok(response);
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

async fn start_worker(
    project_root: &PathBuf,
    worker_environment: &SttWorkerEnvironment,
) -> Result<WorkerProcess, SttClientError> {
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

    let mut command = Command::new(&python);
    command
        .current_dir(project_root)
        .arg("-u")
        .arg("-m")
        .arg("python_services.stt.stt_worker")
        .arg("--server")
        .env("PYTHONUTF8", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8:replace");
    for (key, value) in &worker_environment.values {
        command.env(key, value);
    }
    let mut child = command
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
        ready: false,
    })
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    ready: bool,
}

struct WorkerResponseExpectation {
    request_id: String,
    audio_path: Option<String>,
    timeout: Duration,
}

#[derive(Debug, Serialize)]
struct WorkerRequest {
    request_id: String,
    kind: WorkerRequestKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_path: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum WorkerRequestKind {
    WarmUp,
    Transcribe,
}

#[derive(Debug, Deserialize)]
struct WorkerResponse {
    ok: bool,
    request_id: Option<String>,
    kind: Option<String>,
    audio_path: Option<String>,
    text: Option<String>,
    error: Option<String>,
}

fn next_request_id(prefix: &str) -> String {
    static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);
    let next = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{next}")
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
    String::from_utf8_lossy(bytes)
        .trim_end_matches(['\r', '\n'])
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::SttWorkerEnvironment;
    use std::collections::HashMap;

    #[test]
    fn invalid_utf8_worker_stdout_is_lossy_decoded() {
        let decoded = super::decode_worker_stdout_line(b"{\"ok\":false}\xff\n");
        assert!(decoded.starts_with("{\"ok\":false}"));
    }

    #[test]
    fn meeting_stt_environment_defaults_to_cpu_int8() {
        let environment = SttWorkerEnvironment::meeting_from_lookup(|_| None);
        let timeout = environment.timeout;
        let startup_timeout = environment.startup_timeout;
        let values = environment.values.into_iter().collect::<HashMap<_, _>>();

        assert_eq!(
            values.get("ASTRA_STT_DEVICE").map(String::as_str),
            Some("cpu")
        );
        assert_eq!(
            values.get("ASTRA_STT_COMPUTE_TYPE").map(String::as_str),
            Some("int8")
        );
        assert_eq!(
            values.get("ASTRA_STT_MODEL").map(String::as_str),
            Some("tiny")
        );
        assert_eq!(timeout, std::time::Duration::from_secs(120));
        assert_eq!(startup_timeout, std::time::Duration::from_secs(300));
    }

    #[test]
    fn meeting_stt_environment_honors_overrides() {
        let environment = SttWorkerEnvironment::meeting_from_lookup(|key| match key {
            "ASTRA_MEETING_STT_DEVICE" => Some("cuda".to_string()),
            "ASTRA_MEETING_STT_COMPUTE_TYPE" => Some("float16".to_string()),
            "ASTRA_MEETING_STT_MODEL" => Some("base".to_string()),
            "ASTRA_MEETING_STT_TIMEOUT_SECS" => Some("180".to_string()),
            "ASTRA_MEETING_STT_STARTUP_TIMEOUT_SECS" => Some("240".to_string()),
            _ => None,
        });
        let timeout = environment.timeout;
        let startup_timeout = environment.startup_timeout;
        let values = environment.values.into_iter().collect::<HashMap<_, _>>();

        assert_eq!(
            values.get("ASTRA_STT_DEVICE").map(String::as_str),
            Some("cuda")
        );
        assert_eq!(
            values.get("ASTRA_STT_COMPUTE_TYPE").map(String::as_str),
            Some("float16")
        );
        assert_eq!(
            values.get("ASTRA_STT_MODEL").map(String::as_str),
            Some("base")
        );
        assert_eq!(timeout, std::time::Duration::from_secs(180));
        assert_eq!(startup_timeout, std::time::Duration::from_secs(240));
    }
}
