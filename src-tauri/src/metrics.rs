use serde::Serialize;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_METRICS_HISTORY: usize = 50;

#[derive(Debug, Clone, Serialize)]
pub struct RequestMetricsSnapshot {
    pub request_id: String,
    pub selected_model: String,
    pub user_message_length: usize,
    pub request_started_at: u128,
    pub first_llm_chunk_at: Option<u128>,
    pub llm_completed_at: Option<u128>,
    pub first_segment_queued_at: Option<u128>,
    pub first_audio_ready_at: Option<u128>,
    pub first_audio_play_at: Option<u128>,
    pub audio_completed_at: Option<u128>,
    pub total_request_duration_ms: Option<u128>,
    pub time_to_first_llm_chunk_ms: Option<u128>,
    pub time_to_llm_completed_ms: Option<u128>,
    pub time_to_first_segment_queued_ms: Option<u128>,
    pub time_to_first_audio_ready_ms: Option<u128>,
    pub time_to_first_audio_play_ms: Option<u128>,
}

#[derive(Clone)]
pub struct MetricsTracker {
    inner: Arc<Mutex<MetricsState>>,
}

#[derive(Default)]
struct MetricsState {
    active: HashMap<String, RequestMetrics>,
    history: VecDeque<RequestMetricsSnapshot>,
}

struct RequestMetrics {
    request_id: String,
    selected_model: String,
    user_message_length: usize,
    request_started_at: u128,
    first_llm_chunk_at: Option<u128>,
    llm_completed_at: Option<u128>,
    first_segment_queued_at: Option<u128>,
    first_audio_ready_at: Option<u128>,
    first_audio_play_at: Option<u128>,
    audio_completed_at: Option<u128>,
}

impl MetricsTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MetricsState::default())),
        }
    }

    pub fn start_request(
        &self,
        request_id: String,
        selected_model: String,
        user_message_length: usize,
    ) -> RequestMetricsSnapshot {
        let metrics = RequestMetrics {
            request_id: request_id.clone(),
            selected_model,
            user_message_length,
            request_started_at: now_epoch_ms(),
            first_llm_chunk_at: None,
            llm_completed_at: None,
            first_segment_queued_at: None,
            first_audio_ready_at: None,
            first_audio_play_at: None,
            audio_completed_at: None,
        };

        let snapshot = metrics.snapshot();
        let mut state = self.inner.lock().expect("metrics mutex poisoned");
        state.active.insert(request_id, metrics);
        snapshot
    }

    pub fn mark_first_llm_chunk(&self, request_id: &str) -> Option<RequestMetricsSnapshot> {
        self.mark_once(request_id, |metrics, now| {
            set_once(&mut metrics.first_llm_chunk_at, now)
        })
    }

    pub fn mark_llm_completed(&self, request_id: &str) -> Option<RequestMetricsSnapshot> {
        self.mark_once(request_id, |metrics, now| {
            set_once(&mut metrics.llm_completed_at, now)
        })
    }

    pub fn mark_first_segment_queued(&self, request_id: &str) -> Option<RequestMetricsSnapshot> {
        self.mark_once(request_id, |metrics, now| {
            set_once(&mut metrics.first_segment_queued_at, now)
        })
    }

    pub fn mark_first_audio_ready(&self, request_id: &str) -> Option<RequestMetricsSnapshot> {
        self.mark_once(request_id, |metrics, now| {
            set_once(&mut metrics.first_audio_ready_at, now)
        })
    }

    pub fn mark_first_audio_play(&self, request_id: &str) -> Option<RequestMetricsSnapshot> {
        self.mark_once(request_id, |metrics, now| {
            set_once(&mut metrics.first_audio_play_at, now)
        })
    }

    pub fn mark_audio_completed(&self, request_id: &str) -> Option<RequestMetricsSnapshot> {
        self.mark_once(request_id, |metrics, now| {
            set_once(&mut metrics.audio_completed_at, now)
        })
        .inspect(|snapshot| self.archive_snapshot(snapshot.clone()))
    }

    pub fn get_recent(&self) -> Vec<RequestMetricsSnapshot> {
        let state = self.inner.lock().expect("metrics mutex poisoned");
        state.history.iter().cloned().collect()
    }

    fn mark_once<F>(&self, request_id: &str, update: F) -> Option<RequestMetricsSnapshot>
    where
        F: FnOnce(&mut RequestMetrics, u128) -> bool,
    {
        let now = now_epoch_ms();
        let mut state = self.inner.lock().expect("metrics mutex poisoned");
        let metrics = state.active.get_mut(request_id)?;
        let changed = update(metrics, now);
        changed.then(|| metrics.snapshot())
    }

    fn archive_snapshot(&self, snapshot: RequestMetricsSnapshot) {
        let mut state = self.inner.lock().expect("metrics mutex poisoned");
        state.history.push_back(snapshot);
        while state.history.len() > MAX_METRICS_HISTORY {
            state.history.pop_front();
        }
    }
}

impl RequestMetrics {
    fn snapshot(&self) -> RequestMetricsSnapshot {
        RequestMetricsSnapshot {
            request_id: self.request_id.clone(),
            selected_model: self.selected_model.clone(),
            user_message_length: self.user_message_length,
            request_started_at: self.request_started_at,
            first_llm_chunk_at: self.first_llm_chunk_at,
            llm_completed_at: self.llm_completed_at,
            first_segment_queued_at: self.first_segment_queued_at,
            first_audio_ready_at: self.first_audio_ready_at,
            first_audio_play_at: self.first_audio_play_at,
            audio_completed_at: self.audio_completed_at,
            total_request_duration_ms: delta(self.request_started_at, self.audio_completed_at),
            time_to_first_llm_chunk_ms: delta(self.request_started_at, self.first_llm_chunk_at),
            time_to_llm_completed_ms: delta(self.request_started_at, self.llm_completed_at),
            time_to_first_segment_queued_ms: delta(
                self.request_started_at,
                self.first_segment_queued_at,
            ),
            time_to_first_audio_ready_ms: delta(self.request_started_at, self.first_audio_ready_at),
            time_to_first_audio_play_ms: delta(self.request_started_at, self.first_audio_play_at),
        }
    }
}

fn set_once(target: &mut Option<u128>, value: u128) -> bool {
    if target.is_some() {
        return false;
    }

    *target = Some(value);
    true
}

fn delta(start: u128, end: Option<u128>) -> Option<u128> {
    end.and_then(|value| value.checked_sub(start))
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
