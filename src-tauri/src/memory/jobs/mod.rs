use chrono::{DateTime, Utc};
use futures_util::FutureExt;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    future::Future,
    panic::AssertUnwindSafe,
    pin::Pin,
    sync::{Arc, Mutex},
};
use tauri::async_runtime::JoinHandle;
use tokio::sync::{mpsc, Semaphore};
use uuid::Uuid;

const DEFAULT_MAX_PENDING: usize = 128;
const DEFAULT_MAX_CONCURRENCY: usize = 2;
const DEFAULT_RECENT_EVENTS_LIMIT: usize = 80;

#[derive(Debug, Clone, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MemoryJobKind {
    ConversationConsolidation,
    Reflection,
    EmbeddingMaintenance,
    Reconsolidation,
    SkillExtraction,
    Autopilot,
    Other(String),
}

impl MemoryJobKind {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ConversationConsolidation => "conversation_consolidation",
            Self::Reflection => "reflection",
            Self::EmbeddingMaintenance => "embedding_maintenance",
            Self::Reconsolidation => "reconsolidation",
            Self::SkillExtraction => "skill_extraction",
            Self::Autopilot => "autopilot",
            Self::Other(value) => value.as_str(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryJobRuntimeSnapshot {
    pub job_id: String,
    pub kind: String,
    pub status: String,
    pub dedup_key: Option<String>,
    pub enqueued_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub age_ms: i64,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryJobQueueEvent {
    pub at: DateTime<Utc>,
    pub event: String,
    pub job_id: Option<String>,
    pub kind: Option<String>,
    pub dedup_key: Option<String>,
    pub reason: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Clone)]
struct MemoryJobRuntimeRecord {
    job_id: String,
    kind: MemoryJobKind,
    status: String,
    dedup_key: Option<String>,
    enqueued_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    metadata: Value,
}

impl MemoryJobRuntimeRecord {
    fn snapshot(&self, now: DateTime<Utc>) -> MemoryJobRuntimeSnapshot {
        MemoryJobRuntimeSnapshot {
            job_id: self.job_id.clone(),
            kind: self.kind.as_str().to_string(),
            status: self.status.clone(),
            dedup_key: self.dedup_key.clone(),
            enqueued_at: self.enqueued_at.clone(),
            started_at: self.started_at.clone(),
            age_ms: now.signed_duration_since(self.enqueued_at).num_milliseconds().max(0),
            metadata: self.metadata.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryJobSubmissionReceipt {
    pub accepted: bool,
    pub reason: String,
    pub job_id: Option<String>,
    pub kind: String,
    pub dedup_key: Option<String>,
    pub queued: usize,
    pub running: usize,
    pub max_pending: usize,
    pub max_concurrency: usize,
    pub submitted_at: DateTime<Utc>,
    pub metadata: Value,
}

impl MemoryJobSubmissionReceipt {
    pub fn accepted(
        job_id: String,
        kind: &MemoryJobKind,
        dedup_key: Option<String>,
        snapshot: MemoryJobQueueSnapshot,
        metadata: Value,
    ) -> Self {
        Self {
            accepted: true,
            reason: "memory job accepted by bounded queue".into(),
            job_id: Some(job_id),
            kind: kind.as_str().to_string(),
            dedup_key,
            queued: snapshot.queued,
            running: snapshot.running,
            max_pending: snapshot.max_pending,
            max_concurrency: snapshot.max_concurrency,
            submitted_at: Utc::now(),
            metadata,
        }
    }

    pub fn rejected(
        error: &MemoryJobSubmitError,
        kind: &MemoryJobKind,
        dedup_key: Option<String>,
        snapshot: MemoryJobQueueSnapshot,
        metadata: Value,
    ) -> Self {
        Self {
            accepted: false,
            reason: error.to_string(),
            job_id: None,
            kind: kind.as_str().to_string(),
            dedup_key,
            queued: snapshot.queued,
            running: snapshot.running,
            max_pending: snapshot.max_pending,
            max_concurrency: snapshot.max_concurrency,
            submitted_at: Utc::now(),
            metadata,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryJobQueueSnapshot {
    pub schema_version: u32,
    pub status: String,
    pub max_pending: usize,
    pub max_concurrency: usize,
    pub queued: usize,
    pub running: usize,
    pub pressure_ratio: f32,
    pub concurrency_ratio: f32,
    pub accepted_total: u64,
    pub completed_total: u64,
    pub failed_total: u64,
    pub rejected_full_total: u64,
    pub rejected_duplicate_total: u64,
    pub failed_dispatch_total: u64,
    pub last_event_at: Option<DateTime<Utc>>,
    pub last_rejection_reason: Option<String>,
    #[serde(default)]
    pub active_jobs: Vec<MemoryJobRuntimeSnapshot>,
    #[serde(default)]
    pub recent_events: Vec<MemoryJobQueueEvent>,
}

#[derive(Debug, Default)]
struct MemoryJobQueueStats {
    queued: usize,
    running: usize,
    accepted_total: u64,
    completed_total: u64,
    failed_total: u64,
    rejected_full_total: u64,
    rejected_duplicate_total: u64,
    failed_dispatch_total: u64,
    last_event_at: Option<DateTime<Utc>>,
    last_rejection_reason: Option<String>,
}

#[derive(Clone)]
pub struct MemoryJobQueue {
    inner: Arc<MemoryJobQueueInner>,
}

struct MemoryJobQueueInner {
    sender: mpsc::Sender<MemoryJob>,
    stats: Arc<Mutex<MemoryJobQueueStats>>,
    dedup_keys: Arc<Mutex<HashSet<String>>>,
    active_jobs: Arc<Mutex<HashMap<String, MemoryJobRuntimeRecord>>>,
    recent_events: Arc<Mutex<VecDeque<MemoryJobQueueEvent>>>,
    max_pending: usize,
    max_concurrency: usize,
    _dispatcher: Mutex<Option<JoinHandle<()>>>,
}

type BoxedMemoryJobFuture = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

struct MemoryJob {
    id: String,
    kind: MemoryJobKind,
    dedup_key: Option<String>,
    enqueued_at: DateTime<Utc>,
    metadata: Value,
    task: BoxedMemoryJobFuture,
}

impl MemoryJobQueue {
    pub fn new_from_env() -> Self {
        let max_pending = env_usize("ASTRA_MEMORY_JOB_QUEUE_MAX_PENDING", DEFAULT_MAX_PENDING).max(1);
        let max_concurrency = env_usize("ASTRA_MEMORY_JOB_QUEUE_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY).max(1);
        Self::new(max_pending, max_concurrency)
    }

    pub fn new(max_pending: usize, max_concurrency: usize) -> Self {
        let (sender, mut receiver) = mpsc::channel::<MemoryJob>(max_pending);
        let semaphore = Arc::new(Semaphore::new(max_concurrency));
        let stats = Arc::new(Mutex::new(MemoryJobQueueStats::default()));
        let dedup_keys = Arc::new(Mutex::new(HashSet::<String>::new()));
        let active_jobs = Arc::new(Mutex::new(HashMap::<String, MemoryJobRuntimeRecord>::new()));
        let recent_events = Arc::new(Mutex::new(VecDeque::<MemoryJobQueueEvent>::new()));

        let dispatcher_stats = Arc::clone(&stats);
        let dispatcher_dedup_keys = Arc::clone(&dedup_keys);
        let dispatcher_active_jobs = Arc::clone(&active_jobs);
        let dispatcher_recent_events = Arc::clone(&recent_events);
        let dispatcher_semaphore = Arc::clone(&semaphore);

        let dispatcher = tauri::async_runtime::spawn(async move {
            while let Some(job) = receiver.recv().await {
                {
                    let mut stats = dispatcher_stats
                        .lock()
                        .expect("memory job queue stats mutex poisoned");
                    stats.queued = stats.queued.saturating_sub(1);
                    stats.last_event_at = Some(Utc::now());
                }

                let permit = match dispatcher_semaphore.clone().acquire_owned().await {
                    Ok(permit) => permit,
                    Err(error) => {
                        {
                            let mut active = dispatcher_active_jobs
                                .lock()
                                .expect("memory job queue active jobs mutex poisoned");
                            active.remove(&job.id);
                        }
                        mark_dispatch_failed(
                            &dispatcher_stats,
                            &dispatcher_dedup_keys,
                            job.dedup_key.as_deref(),
                            format!("semaphore_closed:{error}"),
                        );
                        push_event(
                            &dispatcher_recent_events,
                            MemoryJobQueueEvent {
                                at: Utc::now(),
                                event: "dispatch_failed".into(),
                                job_id: Some(job.id.clone()),
                                kind: Some(job.kind.as_str().to_string()),
                                dedup_key: job.dedup_key.clone(),
                                reason: Some(format!("semaphore_closed:{error}")),
                                metadata: json!({"metadata_only": true}),
                            },
                        );
                        continue;
                    }
                };

                let stats = Arc::clone(&dispatcher_stats);
                let dedup_keys = Arc::clone(&dispatcher_dedup_keys);
                let active_jobs = Arc::clone(&dispatcher_active_jobs);
                let recent_events = Arc::clone(&dispatcher_recent_events);
                tauri::async_runtime::spawn(async move {
                    let started_at = Utc::now();
                    {
                        let mut active = active_jobs
                            .lock()
                            .expect("memory job queue active jobs mutex poisoned");
                        if let Some(record) = active.get_mut(&job.id) {
                            record.status = "running".into();
                            record.started_at = Some(started_at);
                        }
                    }
                    {
                        let mut stats = stats
                            .lock()
                            .expect("memory job queue stats mutex poisoned");
                        stats.running = stats.running.saturating_add(1);
                        stats.last_event_at = Some(started_at);
                    }
                    push_event(
                        &recent_events,
                        MemoryJobQueueEvent {
                            at: started_at,
                            event: "started".into(),
                            job_id: Some(job.id.clone()),
                            kind: Some(job.kind.as_str().to_string()),
                            dedup_key: job.dedup_key.clone(),
                            reason: None,
                            metadata: json!({"metadata_only": true, "job_metadata": job.metadata.clone()}),
                        },
                    );

                    eprintln!(
                        "{}",
                        json!({
                            "type": "memory_job_started",
                            "job_id": job.id,
                            "kind": job.kind.as_str(),
                            "enqueued_at": job.enqueued_at.to_rfc3339(),
                            "metadata_only": true,
                        })
                    );

                    let outcome = AssertUnwindSafe(job.task).catch_unwind().await;
                    let success = outcome.is_ok();
                    let finished_at = Utc::now();
                    let failure_reason = if success { None } else { Some("task_panicked".to_string()) };

                    if let Some(dedup_key) = job.dedup_key.as_deref() {
                        let mut keys = dedup_keys
                            .lock()
                            .expect("memory job queue dedup mutex poisoned");
                        keys.remove(dedup_key);
                    }

                    {
                        let mut active = active_jobs
                            .lock()
                            .expect("memory job queue active jobs mutex poisoned");
                        active.remove(&job.id);
                    }

                    {
                        let mut stats = stats
                            .lock()
                            .expect("memory job queue stats mutex poisoned");
                        stats.running = stats.running.saturating_sub(1);
                        if success {
                            stats.completed_total = stats.completed_total.saturating_add(1);
                        } else {
                            stats.failed_total = stats.failed_total.saturating_add(1);
                            stats.last_rejection_reason = failure_reason.clone();
                        }
                        stats.last_event_at = Some(finished_at);
                    }

                    push_event(
                        &recent_events,
                        MemoryJobQueueEvent {
                            at: finished_at,
                            event: if success { "completed".into() } else { "failed".into() },
                            job_id: Some(job.id.clone()),
                            kind: Some(job.kind.as_str().to_string()),
                            dedup_key: job.dedup_key.clone(),
                            reason: failure_reason,
                            metadata: json!({
                                "metadata_only": true,
                                "duration_ms": finished_at.signed_duration_since(started_at).num_milliseconds().max(0),
                                "job_metadata": job.metadata.clone(),
                            }),
                        },
                    );

                    drop(permit);
                });
            }
        });

        Self {
            inner: Arc::new(MemoryJobQueueInner {
                sender,
                stats,
                dedup_keys,
                active_jobs,
                recent_events,
                max_pending,
                max_concurrency,
                _dispatcher: Mutex::new(Some(dispatcher)),
            }),
        }
    }

    pub fn submit<F>(
        &self,
        kind: MemoryJobKind,
        dedup_key: Option<String>,
        task: F,
    ) -> Result<String, MemoryJobSubmitError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        self.submit_with_metadata(kind, dedup_key, Value::Null, task)
    }

    pub fn submit_with_metadata<F>(
        &self,
        kind: MemoryJobKind,
        dedup_key: Option<String>,
        metadata: Value,
        task: F,
    ) -> Result<String, MemoryJobSubmitError>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        if let Some(key) = dedup_key.as_deref() {
            let mut keys = self
                .inner
                .dedup_keys
                .lock()
                .expect("memory job queue dedup mutex poisoned");
            if !keys.insert(key.to_string()) {
                self.record_duplicate_rejection(key);
                push_event(
                    &self.inner.recent_events,
                    MemoryJobQueueEvent {
                        at: Utc::now(),
                        event: "rejected_duplicate".into(),
                        job_id: None,
                        kind: Some(kind.as_str().to_string()),
                        dedup_key: Some(key.to_string()),
                        reason: Some("duplicate_dedup_key".into()),
                        metadata: json!({"metadata_only": true}),
                    },
                );
                return Err(MemoryJobSubmitError::Duplicate { dedup_key: key.to_string() });
            }
        }

        let job_id = Uuid::new_v4().to_string();
        let job = MemoryJob {
            id: job_id.clone(),
            kind: kind.clone(),
            dedup_key: dedup_key.clone(),
            enqueued_at: Utc::now(),
            metadata: metadata.clone(),
            task: Box::pin(task),
        };

        {
            let mut active = self
                .inner
                .active_jobs
                .lock()
                .expect("memory job queue active jobs mutex poisoned");
            active.insert(
                job_id.clone(),
                MemoryJobRuntimeRecord {
                    job_id: job_id.clone(),
                    kind: kind.clone(),
                    status: "queued".into(),
                    dedup_key: dedup_key.clone(),
                    enqueued_at: job.enqueued_at.clone(),
                    started_at: None,
                    metadata: metadata.clone(),
                },
            );
        }

        match self.inner.sender.try_send(job) {
            Ok(()) => {
                let mut stats = self
                    .inner
                    .stats
                    .lock()
                    .expect("memory job queue stats mutex poisoned");
                stats.queued = stats.queued.saturating_add(1);
                stats.accepted_total = stats.accepted_total.saturating_add(1);
                stats.last_event_at = Some(Utc::now());
                drop(stats);
                push_event(
                    &self.inner.recent_events,
                    MemoryJobQueueEvent {
                        at: Utc::now(),
                        event: "accepted".into(),
                        job_id: Some(job_id.clone()),
                        kind: Some(kind.as_str().to_string()),
                        dedup_key: dedup_key.clone(),
                        reason: None,
                        metadata: json!({"metadata_only": true}),
                    },
                );
                Ok(job_id)
            }
            Err(error) => {
                if let Some(key) = dedup_key.as_deref() {
                    let mut keys = self
                        .inner
                        .dedup_keys
                        .lock()
                        .expect("memory job queue dedup mutex poisoned");
                    keys.remove(key);
                }
                {
                    let mut active = self
                        .inner
                        .active_jobs
                        .lock()
                        .expect("memory job queue active jobs mutex poisoned");
                    active.remove(&job_id);
                }
                let mut stats = self
                    .inner
                    .stats
                    .lock()
                    .expect("memory job queue stats mutex poisoned");
                stats.rejected_full_total = stats.rejected_full_total.saturating_add(1);
                stats.last_event_at = Some(Utc::now());
                stats.last_rejection_reason = Some(format!("queue_full_or_closed:{error}"));
                drop(stats);
                push_event(
                    &self.inner.recent_events,
                    MemoryJobQueueEvent {
                        at: Utc::now(),
                        event: "rejected_full_or_closed".into(),
                        job_id: None,
                        kind: Some(kind.as_str().to_string()),
                        dedup_key: dedup_key.clone(),
                        reason: Some(error.to_string()),
                        metadata: json!({"metadata_only": true}),
                    },
                );
                Err(MemoryJobSubmitError::QueueFullOrClosed(error.to_string()))
            }
        }
    }

    pub fn snapshot(&self) -> MemoryJobQueueSnapshot {
        let stats = self
            .inner
            .stats
            .lock()
            .expect("memory job queue stats mutex poisoned");
        let now = Utc::now();
        let active_jobs = self
            .inner
            .active_jobs
            .lock()
            .expect("memory job queue active jobs mutex poisoned")
            .values()
            .map(|record| record.snapshot(now))
            .collect::<Vec<_>>();
        let recent_events = self
            .inner
            .recent_events
            .lock()
            .expect("memory job queue recent events mutex poisoned")
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let pressure_ratio = if self.inner.max_pending == 0 {
            0.0
        } else {
            (stats.queued as f32 / self.inner.max_pending as f32).clamp(0.0, 1.0)
        };
        let concurrency_ratio = if self.inner.max_concurrency == 0 {
            0.0
        } else {
            (stats.running as f32 / self.inner.max_concurrency as f32).clamp(0.0, 1.0)
        };
        let status = if stats.rejected_full_total > 0 && pressure_ratio >= 0.95 {
            "saturated"
        } else if pressure_ratio >= 0.75 || concurrency_ratio >= 1.0 {
            "backpressured"
        } else if stats.failed_total > 0 || stats.failed_dispatch_total > 0 {
            "degraded"
        } else {
            "healthy"
        };
        MemoryJobQueueSnapshot {
            schema_version: 2,
            status: status.into(),
            max_pending: self.inner.max_pending,
            max_concurrency: self.inner.max_concurrency,
            queued: stats.queued,
            running: stats.running,
            pressure_ratio,
            concurrency_ratio,
            accepted_total: stats.accepted_total,
            completed_total: stats.completed_total,
            failed_total: stats.failed_total,
            rejected_full_total: stats.rejected_full_total,
            rejected_duplicate_total: stats.rejected_duplicate_total,
            failed_dispatch_total: stats.failed_dispatch_total,
            last_event_at: stats.last_event_at.clone(),
            last_rejection_reason: stats.last_rejection_reason.clone(),
            active_jobs,
            recent_events,
        }
    }

    fn record_duplicate_rejection(&self, dedup_key: &str) {
        let mut stats = self
            .inner
            .stats
            .lock()
            .expect("memory job queue stats mutex poisoned");
        stats.rejected_duplicate_total = stats.rejected_duplicate_total.saturating_add(1);
        stats.last_event_at = Some(Utc::now());
        stats.last_rejection_reason = Some(format!("duplicate:{dedup_key}"));
    }
}

#[derive(Debug)]
pub enum MemoryJobSubmitError {
    Duplicate { dedup_key: String },
    QueueFullOrClosed(String),
}

impl std::fmt::Display for MemoryJobSubmitError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Duplicate { dedup_key } => write!(formatter, "duplicate memory job: {dedup_key}"),
            Self::QueueFullOrClosed(message) => write!(formatter, "memory job queue unavailable: {message}"),
        }
    }
}

fn mark_dispatch_failed(
    stats: &Arc<Mutex<MemoryJobQueueStats>>,
    dedup_keys: &Arc<Mutex<HashSet<String>>>,
    dedup_key: Option<&str>,
    reason: String,
) {
    if let Some(key) = dedup_key {
        let mut keys = dedup_keys
            .lock()
            .expect("memory job queue dedup mutex poisoned");
        keys.remove(key);
    }
    let mut stats = stats
        .lock()
        .expect("memory job queue stats mutex poisoned");
    stats.failed_dispatch_total = stats.failed_dispatch_total.saturating_add(1);
    stats.last_event_at = Some(Utc::now());
    stats.last_rejection_reason = Some(reason);
}

fn push_event(events: &Arc<Mutex<VecDeque<MemoryJobQueueEvent>>>, event: MemoryJobQueueEvent) {
    let mut events = events
        .lock()
        .expect("memory job queue recent events mutex poisoned");
    events.push_back(event);
    while events.len() > DEFAULT_RECENT_EVENTS_LIMIT {
        events.pop_front();
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(default)
}
