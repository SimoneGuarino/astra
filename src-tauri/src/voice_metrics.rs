use serde::Serialize;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_VOICE_METRICS_HISTORY: usize = 80;

#[derive(Debug, Clone, Serialize)]
pub struct VoiceTurnMetricsSnapshot {
    pub session_id: String,
    pub turn_id: String,
    pub vad_backend: String,
    pub utterance_started_at: Option<u128>,
    pub utterance_ended_at: Option<u128>,
    pub stt_started_at: Option<u128>,
    pub stt_completed_at: Option<u128>,
    pub wake_detected_at: Option<u128>,
    pub response_started_at: Option<u128>,
    pub first_llm_chunk_at: Option<u128>,
    pub first_segment_queued_at: Option<u128>,
    pub first_audio_ready_at: Option<u128>,
    pub first_audio_play_at: Option<u128>,
    pub interruption_detected_at: Option<u128>,
    pub interruption_stop_completed_at: Option<u128>,
    pub follow_up_window_opened_at: Option<u128>,
    pub follow_up_window_closed_at: Option<u128>,
    pub action: Option<String>,
    pub reason: Option<String>,
    pub transcript_length: Option<usize>,
    pub request_id: Option<String>,
    pub utterance_duration_ms: Option<u128>,
    pub speech_to_stt_ms: Option<u128>,
    pub user_end_to_stt_ms: Option<u128>,
    pub stt_duration_ms: Option<u128>,
    pub stt_to_response_start_ms: Option<u128>,
    pub user_end_to_response_start_ms: Option<u128>,
    pub response_start_to_first_audio_ms: Option<u128>,
    pub interruption_latency_ms: Option<u128>,
}

#[derive(Clone)]
pub struct VoiceMetricsTracker {
    inner: Arc<Mutex<VoiceMetricsState>>,
}

#[derive(Default)]
struct VoiceMetricsState {
    active: HashMap<String, VoiceTurnMetrics>,
    request_index: HashMap<String, String>,
    history: VecDeque<VoiceTurnMetricsSnapshot>,
}

#[derive(Debug, Clone)]
struct VoiceTurnMetrics {
    session_id: String,
    turn_id: String,
    vad_backend: String,
    utterance_started_at: Option<u128>,
    utterance_ended_at: Option<u128>,
    stt_started_at: Option<u128>,
    stt_completed_at: Option<u128>,
    wake_detected_at: Option<u128>,
    response_started_at: Option<u128>,
    first_llm_chunk_at: Option<u128>,
    first_segment_queued_at: Option<u128>,
    first_audio_ready_at: Option<u128>,
    first_audio_play_at: Option<u128>,
    interruption_detected_at: Option<u128>,
    interruption_stop_completed_at: Option<u128>,
    follow_up_window_opened_at: Option<u128>,
    follow_up_window_closed_at: Option<u128>,
    action: Option<String>,
    reason: Option<String>,
    transcript_length: Option<usize>,
    request_id: Option<String>,
}

impl VoiceMetricsTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VoiceMetricsState::default())),
        }
    }

    pub fn start_utterance(
        &self,
        session_id: &str,
        turn_id: &str,
        vad_backend: &str,
    ) -> VoiceTurnMetricsSnapshot {
        let metrics = VoiceTurnMetrics {
            session_id: session_id.to_string(),
            turn_id: turn_id.to_string(),
            vad_backend: vad_backend.to_string(),
            utterance_started_at: Some(now_epoch_ms()),
            utterance_ended_at: None,
            stt_started_at: None,
            stt_completed_at: None,
            wake_detected_at: None,
            response_started_at: None,
            first_llm_chunk_at: None,
            first_segment_queued_at: None,
            first_audio_ready_at: None,
            first_audio_play_at: None,
            interruption_detected_at: None,
            interruption_stop_completed_at: None,
            follow_up_window_opened_at: None,
            follow_up_window_closed_at: None,
            action: None,
            reason: None,
            transcript_length: None,
            request_id: None,
        };
        let snapshot = metrics.snapshot();
        self.inner
            .lock()
            .expect("voice metrics mutex poisoned")
            .active
            .insert(key(session_id, turn_id), metrics);
        snapshot
    }

    pub fn mark_utterance_ended(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark(session_id, turn_id, |metrics, now| {
            set_once(&mut metrics.utterance_ended_at, now);
        })
    }

    pub fn mark_stt_started(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark(session_id, turn_id, |metrics, now| {
            set_once(&mut metrics.stt_started_at, now);
        })
    }

    pub fn mark_stt_completed(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark(session_id, turn_id, |metrics, now| {
            set_once(&mut metrics.stt_completed_at, now);
        })
    }

    pub fn mark_interruption_detected(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark(session_id, turn_id, |metrics, now| {
            set_once(&mut metrics.interruption_detected_at, now);
        })
    }

    pub fn mark_interruption_stop_completed(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark(session_id, turn_id, |metrics, now| {
            set_once(&mut metrics.interruption_stop_completed_at, now);
        })
    }

    pub fn mark_decision(
        &self,
        session_id: &str,
        turn_id: &str,
        action: &str,
        reason: &str,
        transcript_length: usize,
        wake_detected: bool,
        follow_up_opened: bool,
        follow_up_closed: bool,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark(session_id, turn_id, |metrics, now| {
            metrics.action = Some(action.to_string());
            metrics.reason = Some(reason.to_string());
            metrics.transcript_length = Some(transcript_length);
            if wake_detected {
                set_once(&mut metrics.wake_detected_at, now);
            }
            if follow_up_opened {
                set_once(&mut metrics.follow_up_window_opened_at, now);
            }
            if follow_up_closed {
                set_once(&mut metrics.follow_up_window_closed_at, now);
            }
        })
    }

    pub fn mark_response_started(
        &self,
        session_id: &str,
        turn_id: &str,
        request_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        let mut state = self.inner.lock().expect("voice metrics mutex poisoned");
        let turn_key = key(session_id, turn_id);
        let snapshot = {
            let metrics = state.active.get_mut(&turn_key)?;
            let now = now_epoch_ms();
            set_once(&mut metrics.response_started_at, now);
            metrics.request_id = Some(request_id.to_string());
            metrics.snapshot()
        };
        state.request_index.insert(request_id.to_string(), turn_key);
        Some(snapshot)
    }

    pub fn mark_first_llm_chunk_for_request(
        &self,
        request_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark_by_request(request_id, |metrics, now| {
            set_once(&mut metrics.first_llm_chunk_at, now);
        })
    }

    pub fn mark_first_segment_queued_for_request(
        &self,
        request_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark_by_request(request_id, |metrics, now| {
            set_once(&mut metrics.first_segment_queued_at, now);
        })
    }

    pub fn mark_first_audio_ready_for_request(
        &self,
        request_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark_by_request(request_id, |metrics, now| {
            set_once(&mut metrics.first_audio_ready_at, now);
        })
    }

    pub fn mark_first_audio_play_for_request(
        &self,
        request_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        self.mark_by_request(request_id, |metrics, now| {
            set_once(&mut metrics.first_audio_play_at, now);
        })
    }

    pub fn complete_turn(
        &self,
        session_id: &str,
        turn_id: &str,
    ) -> Option<VoiceTurnMetricsSnapshot> {
        let mut state = self.inner.lock().expect("voice metrics mutex poisoned");
        let snapshot = state.active.remove(&key(session_id, turn_id))?.snapshot();
        if let Some(request_id) = snapshot.request_id.as_ref() {
            state.request_index.remove(request_id);
        }
        archive_snapshot(&mut state.history, snapshot.clone());
        Some(snapshot)
    }

    pub fn get_recent(&self) -> Vec<VoiceTurnMetricsSnapshot> {
        let state = self.inner.lock().expect("voice metrics mutex poisoned");
        let mut snapshots = state.history.iter().cloned().collect::<Vec<_>>();
        snapshots.extend(state.active.values().map(VoiceTurnMetrics::snapshot));
        snapshots
    }

    fn mark<F>(
        &self,
        session_id: &str,
        turn_id: &str,
        update: F,
    ) -> Option<VoiceTurnMetricsSnapshot>
    where
        F: FnOnce(&mut VoiceTurnMetrics, u128),
    {
        let now = now_epoch_ms();
        let mut state = self.inner.lock().expect("voice metrics mutex poisoned");
        let metrics = state.active.get_mut(&key(session_id, turn_id))?;
        update(metrics, now);
        Some(metrics.snapshot())
    }

    fn mark_by_request<F>(&self, request_id: &str, update: F) -> Option<VoiceTurnMetricsSnapshot>
    where
        F: FnOnce(&mut VoiceTurnMetrics, u128),
    {
        let now = now_epoch_ms();
        let mut state = self.inner.lock().expect("voice metrics mutex poisoned");
        let turn_key = state.request_index.get(request_id)?.clone();
        let metrics = state.active.get_mut(&turn_key)?;
        update(metrics, now);
        Some(metrics.snapshot())
    }
}

impl VoiceTurnMetrics {
    fn snapshot(&self) -> VoiceTurnMetricsSnapshot {
        let speech_to_stt_ms = delta(self.utterance_ended_at, self.stt_started_at);
        VoiceTurnMetricsSnapshot {
            session_id: self.session_id.clone(),
            turn_id: self.turn_id.clone(),
            vad_backend: self.vad_backend.clone(),
            utterance_started_at: self.utterance_started_at,
            utterance_ended_at: self.utterance_ended_at,
            stt_started_at: self.stt_started_at,
            stt_completed_at: self.stt_completed_at,
            wake_detected_at: self.wake_detected_at,
            response_started_at: self.response_started_at,
            first_llm_chunk_at: self.first_llm_chunk_at,
            first_segment_queued_at: self.first_segment_queued_at,
            first_audio_ready_at: self.first_audio_ready_at,
            first_audio_play_at: self.first_audio_play_at,
            interruption_detected_at: self.interruption_detected_at,
            interruption_stop_completed_at: self.interruption_stop_completed_at,
            follow_up_window_opened_at: self.follow_up_window_opened_at,
            follow_up_window_closed_at: self.follow_up_window_closed_at,
            action: self.action.clone(),
            reason: self.reason.clone(),
            transcript_length: self.transcript_length,
            request_id: self.request_id.clone(),
            utterance_duration_ms: delta(self.utterance_started_at, self.utterance_ended_at),
            speech_to_stt_ms,
            user_end_to_stt_ms: speech_to_stt_ms,
            stt_duration_ms: delta(self.stt_started_at, self.stt_completed_at),
            stt_to_response_start_ms: delta(self.stt_completed_at, self.response_started_at),
            user_end_to_response_start_ms: delta(self.utterance_ended_at, self.response_started_at),
            response_start_to_first_audio_ms: delta(self.response_started_at, self.first_audio_play_at),
            interruption_latency_ms: delta(
                self.interruption_detected_at,
                self.interruption_stop_completed_at,
            ),
        }
    }
}

fn key(session_id: &str, turn_id: &str) -> String {
    format!("{session_id}:{turn_id}")
}

fn set_once(target: &mut Option<u128>, value: u128) {
    if target.is_none() {
        *target = Some(value);
    }
}

fn archive_snapshot(
    history: &mut VecDeque<VoiceTurnMetricsSnapshot>,
    snapshot: VoiceTurnMetricsSnapshot,
) {
    history.push_back(snapshot);
    while history.len() > MAX_VOICE_METRICS_HISTORY {
        history.pop_front();
    }
}

fn delta(start: Option<u128>, end: Option<u128>) -> Option<u128> {
    end.zip(start)
        .and_then(|(end_value, start_value)| end_value.checked_sub(start_value))
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}
