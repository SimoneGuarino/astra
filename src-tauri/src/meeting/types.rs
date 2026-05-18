//! Meeting types — shared between Rust and TypeScript contracts

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub const CLEAR_MEETING_DATA_CONFIRMATION_PHRASE: &str = "DELETE_MEETING_DATA";
pub const DEFAULT_CAPTURE_SEGMENT_DURATION_MS: u64 = 15_000;
pub const MIN_CAPTURE_SEGMENT_DURATION_MS: u64 = 10_000;
pub const MAX_CAPTURE_SEGMENT_DURATION_MS: u64 = 30_000;
pub const DEFAULT_CAPTURE_MAX_QUEUE_DEPTH: usize = 64;
pub const DEFAULT_CAPTURE_MAX_SEGMENTS_PER_SESSION: u64 = 720;
pub const DEFAULT_CAPTURE_MAX_SEGMENT_BYTES: usize = 8 * 1024 * 1024;
pub const DEFAULT_CAPTURE_MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES: usize = 3;
pub const DEFAULT_CAPTURE_VAD_ENABLED: bool = true;
pub const DEFAULT_CAPTURE_VAD_SILENCE_THRESHOLD_PCM: u16 = 500;
pub const DEFAULT_CAPTURE_VAD_MIN_SPEECH_MS: u64 = 250;
pub const DEFAULT_CAPTURE_VAD_MIN_SILENCE_MS: u64 = 700;
pub const DEFAULT_CAPTURE_VAD_MIN_SPEECH_RATIO_BPS: u16 = 500;

pub fn normalize_meeting_app_name(input: &str) -> String {
    let lowercase = input.trim().to_ascii_lowercase();
    let separated = lowercase
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>();
    let collapsed = separated.split_whitespace().collect::<Vec<_>>().join(" ");

    match collapsed.as_str() {
        "" => String::new(),
        "teams" | "microsoft teams" | "ms teams" | "msteams" => "teams".to_string(),
        "zoom" | "zoom meetings" => "zoom".to_string(),
        "google meet" | "meet" => "google_meet".to_string(),
        "discord" => "discord".to_string(),
        "slack" => "slack".to_string(),
        "webex" | "cisco webex" => "webex".to_string(),
        "edge" | "microsoft edge" | "msedge" => "edge".to_string(),
        "chrome" | "google chrome" => "chrome".to_string(),
        other => other.replace(' ', "_"),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSession {
    pub session_id: String,
    pub platform: String,
    pub status: MeetingStatus,
    pub started_at: DateTime<Utc>,
    pub participants: Vec<ParticipantInfo>,
    pub config: MeetingConfig,
    #[serde(default)]
    pub session_mode: MeetingSessionMode,
    #[serde(default)]
    pub capture_active: bool,
    #[serde(default)]
    pub capture_backend_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MeetingStatus {
    Idle,
    ConsentRequired,
    Detecting,
    Ready,
    Starting,
    Capturing,
    Transcribing,
    Summarizing,
    Paused,
    Stopping,
    Stopped,
    Completed,
    Failed(String),
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantInfo {
    pub name: String,
    #[serde(default)]
    pub speaker_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingConfig {
    pub platform: String,
    pub capture_backend: CaptureBackend,
    pub transcription_model: String,
    pub sample_rate: u32,
    pub diarization_enabled: bool,
    pub privacy_mode: String,
    #[serde(default)]
    pub session_mode: MeetingSessionMode,
    #[serde(default)]
    pub live_transcription_enabled: bool,
    #[serde(default)]
    pub capture_options: MeetingCaptureOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Copy, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingSessionMode {
    #[default]
    Manual,
    RealCapture,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingCaptureOptions {
    #[serde(default = "default_capture_system_audio")]
    pub system_audio: bool,
    #[serde(default)]
    pub microphone: bool,
    #[serde(default)]
    pub segment_transcription: bool,
}

impl Default for MeetingCaptureOptions {
    fn default() -> Self {
        Self {
            system_audio: true,
            microphone: false,
            segment_transcription: false,
        }
    }
}

impl MeetingCaptureOptions {
    pub fn manual() -> Self {
        Self {
            system_audio: false,
            microphone: false,
            segment_transcription: false,
        }
    }

    pub fn any_audio_enabled(self) -> bool {
        self.system_audio || self.microphone
    }
}

fn default_capture_system_audio() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptSource {
    Microphone,
    SystemAudio,
    Manual,
    ImportedFile,
    #[default]
    Unknown,
}

impl TranscriptSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Microphone => "microphone",
            Self::SystemAudio => "system_audio",
            Self::Manual => "manual",
            Self::ImportedFile => "imported_file",
            Self::Unknown => "unknown",
        }
    }
}

pub const LOCAL_USER_SPEAKER_ID: &str = "local_user";
pub const REMOTE_SPEAKER_1_ID: &str = "remote_speaker_1";
pub const MANUAL_SPEAKER_ID: &str = "manual_entry";
pub const IMPORTED_SPEAKER_ID: &str = "imported_file";
pub const UNKNOWN_SPEAKER_ID: &str = "unknown_speaker";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerAttributionMethod {
    SourceDefault,
    UserAssigned,
    HeuristicTurnSplit,
    DiarizationModel,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeakerLabel {
    pub speaker_id: String,
    pub display_name: String,
    pub source: TranscriptSource,
    pub confidence: f32,
    pub attribution_method: SpeakerAttributionMethod,
}

impl SpeakerLabel {
    pub fn source_default(source: TranscriptSource) -> Self {
        let (speaker_id, display_name, confidence, method) = match source {
            TranscriptSource::Microphone => (
                LOCAL_USER_SPEAKER_ID,
                "You",
                1.0,
                SpeakerAttributionMethod::SourceDefault,
            ),
            TranscriptSource::SystemAudio => (
                REMOTE_SPEAKER_1_ID,
                "Speaker 1",
                0.65,
                SpeakerAttributionMethod::SourceDefault,
            ),
            TranscriptSource::Manual => (
                MANUAL_SPEAKER_ID,
                "Manual",
                0.9,
                SpeakerAttributionMethod::SourceDefault,
            ),
            TranscriptSource::ImportedFile => (
                IMPORTED_SPEAKER_ID,
                "Imported",
                0.8,
                SpeakerAttributionMethod::SourceDefault,
            ),
            TranscriptSource::Unknown => (
                UNKNOWN_SPEAKER_ID,
                "Unknown",
                0.0,
                SpeakerAttributionMethod::Unknown,
            ),
        };

        Self {
            speaker_id: speaker_id.to_string(),
            display_name: display_name.to_string(),
            source,
            confidence,
            attribution_method: method,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Copy)]
#[serde(rename_all = "snake_case")]
pub enum CaptureBackend {
    PipeWire,
    CoreAudio,
    Wasapi,
    Default,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CaptureControllerState {
    #[default]
    Idle,
    Unsupported,
    Starting,
    Capturing,
    Paused,
    Stopping,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CaptureHealthStatus {
    #[default]
    Idle,
    Healthy,
    Unsupported,
    Backpressure,
    ConsentRevoked,
    StopTimedOut,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CaptureOverflowPolicy {
    #[default]
    RejectNewest,
    DropOldestAndReport,
    StopCapture,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapturePipelineConfig {
    pub max_queued_chunks: usize,
    pub chunk_duration_ms: u64,
    pub max_memory_bytes: usize,
    pub overflow_policy: CaptureOverflowPolicy,
    pub max_retries: u8,
    pub max_segments_per_session: u64,
    #[serde(default = "default_max_consecutive_transcription_failures")]
    pub max_consecutive_transcription_failures: usize,
    #[serde(default = "default_vad_enabled")]
    pub vad_enabled: bool,
    #[serde(default = "default_vad_silence_threshold_pcm")]
    pub vad_silence_threshold_pcm: u16,
    #[serde(default = "default_vad_min_speech_ms")]
    pub vad_min_speech_ms: u64,
    #[serde(default = "default_vad_min_silence_ms")]
    pub vad_min_silence_ms: u64,
    #[serde(default = "default_vad_min_speech_ratio_bps")]
    pub vad_min_speech_ratio_bps: u16,
}

impl Default for CapturePipelineConfig {
    fn default() -> Self {
        Self {
            max_queued_chunks: DEFAULT_CAPTURE_MAX_QUEUE_DEPTH,
            chunk_duration_ms: DEFAULT_CAPTURE_SEGMENT_DURATION_MS,
            max_memory_bytes: DEFAULT_CAPTURE_MAX_SEGMENT_BYTES,
            overflow_policy: CaptureOverflowPolicy::RejectNewest,
            max_retries: 2,
            max_segments_per_session: DEFAULT_CAPTURE_MAX_SEGMENTS_PER_SESSION,
            max_consecutive_transcription_failures:
                DEFAULT_CAPTURE_MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES,
            vad_enabled: DEFAULT_CAPTURE_VAD_ENABLED,
            vad_silence_threshold_pcm: DEFAULT_CAPTURE_VAD_SILENCE_THRESHOLD_PCM,
            vad_min_speech_ms: DEFAULT_CAPTURE_VAD_MIN_SPEECH_MS,
            vad_min_silence_ms: DEFAULT_CAPTURE_VAD_MIN_SILENCE_MS,
            vad_min_speech_ratio_bps: DEFAULT_CAPTURE_VAD_MIN_SPEECH_RATIO_BPS,
        }
    }
}

fn default_max_consecutive_transcription_failures() -> usize {
    DEFAULT_CAPTURE_MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES
}

fn default_vad_enabled() -> bool {
    DEFAULT_CAPTURE_VAD_ENABLED
}

fn default_vad_silence_threshold_pcm() -> u16 {
    DEFAULT_CAPTURE_VAD_SILENCE_THRESHOLD_PCM
}

fn default_vad_min_speech_ms() -> u64 {
    DEFAULT_CAPTURE_VAD_MIN_SPEECH_MS
}

fn default_vad_min_silence_ms() -> u64 {
    DEFAULT_CAPTURE_VAD_MIN_SILENCE_MS
}

fn default_vad_min_speech_ratio_bps() -> u16 {
    DEFAULT_CAPTURE_VAD_MIN_SPEECH_RATIO_BPS
}

impl CapturePipelineConfig {
    pub fn effective(&self) -> EffectiveCapturePipelineConfig {
        let effective_segment_duration_ms = self.chunk_duration_ms.clamp(
            MIN_CAPTURE_SEGMENT_DURATION_MS,
            MAX_CAPTURE_SEGMENT_DURATION_MS,
        );
        let effective_max_queue_depth = self.max_queued_chunks.max(1);
        let effective_max_segments_per_session = self.max_segments_per_session.max(1);
        let max_segment_bytes = self
            .max_memory_bytes
            .clamp(44, DEFAULT_CAPTURE_MAX_SEGMENT_BYTES);
        let estimated_max_session_duration_ms =
            effective_segment_duration_ms.saturating_mul(effective_max_segments_per_session);

        EffectiveCapturePipelineConfig {
            requested_segment_duration_ms: self.chunk_duration_ms,
            effective_segment_duration_ms,
            min_segment_duration_ms: MIN_CAPTURE_SEGMENT_DURATION_MS,
            max_segment_duration_ms: MAX_CAPTURE_SEGMENT_DURATION_MS,
            requested_max_queue_depth: self.max_queued_chunks,
            effective_max_queue_depth,
            requested_max_segments_per_session: self.max_segments_per_session,
            effective_max_segments_per_session,
            max_segment_bytes,
            estimated_max_session_duration_ms,
            duration_clamped: self.chunk_duration_ms != effective_segment_duration_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffectiveCapturePipelineConfig {
    pub requested_segment_duration_ms: u64,
    pub effective_segment_duration_ms: u64,
    pub min_segment_duration_ms: u64,
    pub max_segment_duration_ms: u64,
    pub requested_max_queue_depth: usize,
    pub effective_max_queue_depth: usize,
    pub requested_max_segments_per_session: u64,
    pub effective_max_segments_per_session: u64,
    pub max_segment_bytes: usize,
    pub estimated_max_session_duration_ms: u64,
    pub duration_clamped: bool,
}

impl Default for EffectiveCapturePipelineConfig {
    fn default() -> Self {
        CapturePipelineConfig::default().effective()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioSampleFormat {
    #[default]
    F32Pcm,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingAudioChunk {
    pub sample_rate: u32,
    pub channels: u16,
    pub format: AudioSampleFormat,
    pub monotonic_timestamp_ms: u64,
    pub sequence_number: u64,
    pub duration_ms: u64,
    pub source_backend: CaptureBackend,
    #[serde(default)]
    pub transcript_source: TranscriptSource,
    pub byte_length: usize,
    pub frame_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingAudioSegment {
    #[serde(default)]
    pub session_id: Option<String>,
    pub chunks: Vec<MeetingAudioChunk>,
    pub total_duration_ms: u64,
    pub source_backend: CaptureBackend,
    #[serde(default)]
    pub transcript_source: TranscriptSource,
    pub contains_raw_audio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CaptureMetrics {
    pub chunks_received: u64,
    pub chunks_dropped: u64,
    pub chunks_transcribed: u64,
    #[serde(default)]
    pub wasapi_endpoint_acquired: bool,
    #[serde(default)]
    pub wasapi_mix_format_detected: bool,
    #[serde(default)]
    pub wasapi_sample_rate: Option<u32>,
    #[serde(default)]
    pub wasapi_channel_count: Option<u16>,
    #[serde(default)]
    pub wasapi_sample_format: Option<String>,
    #[serde(default)]
    pub wasapi_buffer_frame_count: Option<u32>,
    #[serde(default)]
    pub wasapi_stream_initialized: bool,
    #[serde(default)]
    pub wasapi_stream_started: bool,
    #[serde(default)]
    pub wasapi_packets_read: u64,
    #[serde(default)]
    pub frames_captured: u64,
    #[serde(default)]
    pub frames_converted: u64,
    #[serde(default)]
    pub silence_frames_skipped: u64,
    pub segments_written: u64,
    #[serde(default)]
    pub segments_queued: u64,
    #[serde(default)]
    pub segments_queued_total: u64,
    #[serde(default)]
    pub current_queue_depth: usize,
    #[serde(default)]
    pub segments_dequeued_total: u64,
    #[serde(default)]
    pub segments_in_flight: u64,
    pub segments_transcribed: u64,
    #[serde(default)]
    pub segments_failed: u64,
    #[serde(default)]
    pub segment_transcription_timeouts: u64,
    pub segments_dropped: u64,
    #[serde(default)]
    pub dropped_silence_segments: u64,
    pub segment_write_failures: u64,
    pub segment_transcription_failures: u64,
    #[serde(default)]
    pub segment_transcription_failures_total: u64,
    #[serde(default)]
    pub segment_transcription_failures_consecutive: u64,
    pub queue_full_events: u64,
    pub bytes_queued: u64,
    pub max_queue_depth_seen: usize,
    pub backpressure_active: bool,
    #[serde(default)]
    pub last_segment_status: Option<String>,
    #[serde(default)]
    pub last_overflow_policy_applied: Option<CaptureOverflowPolicy>,
    #[serde(default)]
    pub last_segment_transcription_error_kind: Option<String>,
    #[serde(default)]
    pub last_segment_transcription_failure_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub last_transcription_started_segment_id: Option<String>,
    #[serde(default)]
    pub last_transcription_completed_segment_id: Option<String>,
    #[serde(default)]
    pub last_transcription_failed_segment_id: Option<String>,
    #[serde(default)]
    pub drain_started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub drain_completed_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub drain_timeout: bool,
    #[serde(default)]
    pub segment_transcription_drain_status: Option<String>,
    #[serde(default)]
    pub vad_speech_frames: u64,
    #[serde(default)]
    pub vad_silence_frames: u64,
    #[serde(default)]
    pub last_speech_ratio_bps: u16,
    #[serde(default)]
    pub last_silence_ratio_bps: u16,
    #[serde(default)]
    pub audio_clipped_sample_count: u64,
    #[serde(default)]
    pub audio_peak_abs: u16,
    #[serde(default)]
    pub audio_rms_bps: u16,
    #[serde(default)]
    pub audio_normalization_gain_bps: u16,
    #[serde(default)]
    pub last_backend_error_kind: Option<String>,
    #[serde(default)]
    pub last_backend_error_message: Option<String>,
    #[serde(default)]
    pub last_successful_segment_at: Option<DateTime<Utc>>,
    pub restarts_attempted: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CaptureHealth {
    pub state: CaptureControllerState,
    pub status: CaptureHealthStatus,
    #[serde(default)]
    pub backend: Option<CaptureBackend>,
    pub active_handle_present: bool,
    pub backpressure_active: bool,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub last_segment_status: Option<String>,
    #[serde(default)]
    pub last_overflow_policy_applied: Option<CaptureOverflowPolicy>,
    pub pipeline: CapturePipelineConfig,
    #[serde(default)]
    pub effective_pipeline: EffectiveCapturePipelineConfig,
    pub metrics: CaptureMetrics,
}

impl Default for CaptureHealth {
    fn default() -> Self {
        Self {
            state: CaptureControllerState::Idle,
            status: CaptureHealthStatus::Idle,
            backend: None,
            active_handle_present: false,
            backpressure_active: false,
            last_error: None,
            last_segment_status: None,
            last_overflow_policy_applied: None,
            pipeline: CapturePipelineConfig::default(),
            effective_pipeline: EffectiveCapturePipelineConfig::default(),
            metrics: CaptureMetrics::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingSttCompletenessStatus {
    Complete,
    CompleteNoSpeech,
    IncompleteDrainTimeout,
    IncompletePendingQueue,
    IncompleteInFlight,
    IncompleteFailedSegments,
    IncompleteTimeouts,
    Unavailable,
    #[default]
    Unknown,
}

impl MeetingSttCompletenessStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::CompleteNoSpeech => "complete_no_speech",
            Self::IncompleteDrainTimeout => "incomplete_drain_timeout",
            Self::IncompletePendingQueue => "incomplete_pending_queue",
            Self::IncompleteInFlight => "incomplete_in_flight",
            Self::IncompleteFailedSegments => "incomplete_failed_segments",
            Self::IncompleteTimeouts => "incomplete_timeouts",
            Self::Unavailable => "unavailable",
            Self::Unknown => "unknown",
        }
    }

    pub fn is_incomplete(self) -> bool {
        matches!(
            self,
            Self::IncompleteDrainTimeout
                | Self::IncompletePendingQueue
                | Self::IncompleteInFlight
                | Self::IncompleteFailedSegments
                | Self::IncompleteTimeouts
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingSttCompletenessSource {
    pub status: MeetingSttCompletenessStatus,
    pub segments_written: u64,
    pub segments_transcribed: u64,
    pub current_queue_depth: usize,
    pub segments_in_flight: u64,
    pub segments_failed: u64,
    pub timeouts: u64,
    pub dropped_silence_segments: u64,
    #[serde(default)]
    pub drain_status: Option<String>,
    pub drain_timeout: bool,
    #[serde(default)]
    pub last_error_kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingSttCompletenessReport {
    pub overall: MeetingSttCompletenessStatus,
    pub system_audio: MeetingSttCompletenessSource,
    pub microphone: MeetingSttCompletenessSource,
    pub segments_written: u64,
    pub segments_transcribed: u64,
    pub current_queue_depth: usize,
    pub segments_in_flight: u64,
    pub segments_failed: u64,
    pub timeouts: u64,
}

pub fn derive_meeting_stt_completeness(
    system_health: &CaptureHealth,
    microphone_health: &CaptureHealth,
) -> MeetingSttCompletenessReport {
    let system_audio = derive_meeting_stt_source_completeness(&system_health.metrics);
    let microphone = derive_meeting_stt_source_completeness(&microphone_health.metrics);
    let overall = aggregate_meeting_stt_completeness(system_audio.status, microphone.status);

    MeetingSttCompletenessReport {
        overall,
        segments_written: system_audio
            .segments_written
            .saturating_add(microphone.segments_written),
        segments_transcribed: system_audio
            .segments_transcribed
            .saturating_add(microphone.segments_transcribed),
        current_queue_depth: system_audio
            .current_queue_depth
            .saturating_add(microphone.current_queue_depth),
        segments_in_flight: system_audio
            .segments_in_flight
            .saturating_add(microphone.segments_in_flight),
        segments_failed: system_audio
            .segments_failed
            .saturating_add(microphone.segments_failed),
        timeouts: system_audio.timeouts.saturating_add(microphone.timeouts),
        system_audio,
        microphone,
    }
}

pub fn derive_meeting_stt_source_completeness(
    metrics: &CaptureMetrics,
) -> MeetingSttCompletenessSource {
    let status = if metrics.drain_timeout
        || matches!(
            metrics.segment_transcription_drain_status.as_deref(),
            Some("timed_out")
        ) {
        MeetingSttCompletenessStatus::IncompleteDrainTimeout
    } else if metrics.current_queue_depth > 0 {
        MeetingSttCompletenessStatus::IncompletePendingQueue
    } else if metrics.segments_in_flight > 0 {
        MeetingSttCompletenessStatus::IncompleteInFlight
    } else if metrics.segments_failed > 0
        || metrics.segment_transcription_failures_total > 0
        || metrics.last_segment_transcription_error_kind.is_some()
    {
        MeetingSttCompletenessStatus::IncompleteFailedSegments
    } else if metrics.segment_transcription_timeouts > 0 {
        MeetingSttCompletenessStatus::IncompleteTimeouts
    } else if metrics.segments_written > 0
        && metrics.segments_transcribed < metrics.segments_written
    {
        MeetingSttCompletenessStatus::IncompleteFailedSegments
    } else if metrics.segments_written == 0 && metrics.dropped_silence_segments > 0 {
        MeetingSttCompletenessStatus::CompleteNoSpeech
    } else if metrics.segments_written > 0
        && metrics.segments_transcribed == metrics.segments_written
    {
        MeetingSttCompletenessStatus::Complete
    } else if matches!(
        metrics.segment_transcription_drain_status.as_deref(),
        Some("completed" | "closed")
    ) {
        MeetingSttCompletenessStatus::CompleteNoSpeech
    } else {
        MeetingSttCompletenessStatus::Unknown
    };

    MeetingSttCompletenessSource {
        status,
        segments_written: metrics.segments_written,
        segments_transcribed: metrics.segments_transcribed,
        current_queue_depth: metrics.current_queue_depth,
        segments_in_flight: metrics.segments_in_flight,
        segments_failed: metrics.segments_failed,
        timeouts: metrics.segment_transcription_timeouts,
        dropped_silence_segments: metrics.dropped_silence_segments,
        drain_status: metrics.segment_transcription_drain_status.clone(),
        drain_timeout: metrics.drain_timeout,
        last_error_kind: metrics.last_segment_transcription_error_kind.clone(),
    }
}

fn aggregate_meeting_stt_completeness(
    system_audio: MeetingSttCompletenessStatus,
    microphone: MeetingSttCompletenessStatus,
) -> MeetingSttCompletenessStatus {
    let statuses = [system_audio, microphone];
    for candidate in [
        MeetingSttCompletenessStatus::IncompleteDrainTimeout,
        MeetingSttCompletenessStatus::IncompletePendingQueue,
        MeetingSttCompletenessStatus::IncompleteInFlight,
        MeetingSttCompletenessStatus::IncompleteFailedSegments,
        MeetingSttCompletenessStatus::IncompleteTimeouts,
    ] {
        if statuses.contains(&candidate) {
            return candidate;
        }
    }
    if statuses.contains(&MeetingSttCompletenessStatus::Complete) {
        MeetingSttCompletenessStatus::Complete
    } else if statuses.contains(&MeetingSttCompletenessStatus::CompleteNoSpeech) {
        MeetingSttCompletenessStatus::CompleteNoSpeech
    } else if statuses.contains(&MeetingSttCompletenessStatus::Unavailable) {
        MeetingSttCompletenessStatus::Unavailable
    } else {
        MeetingSttCompletenessStatus::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeetingCapabilityState {
    Ready,
    Disabled,
    Unavailable,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingCapabilityReadiness {
    pub capability: String,
    pub available: bool,
    pub state: MeetingCapabilityState,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingSttAdapterStatus {
    pub state: MeetingCapabilityState,
    pub existing_boundary: String,
    pub file_transcription: MeetingCapabilityReadiness,
    pub live_transcription: MeetingCapabilityReadiness,
    pub chunk_streaming: MeetingCapabilityReadiness,
    pub chunk_streaming_supported: bool,
    pub emits_placeholder_transcripts: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingLiveCapabilitySnapshot {
    pub manual_session: MeetingCapabilityReadiness,
    pub audio_capture: MeetingCapabilityReadiness,
    pub microphone_capture: MeetingCapabilityReadiness,
    pub system_audio_capture: MeetingCapabilityReadiness,
    pub windows_wasapi_capture: MeetingCapabilityReadiness,
    pub system_capture_health: CaptureHealth,
    pub microphone_capture_health: CaptureHealth,
    pub live_transcription: MeetingCapabilityReadiness,
    pub live_segment_transcription: MeetingCapabilityReadiness,
    pub live_streaming_stt: MeetingCapabilityReadiness,
    pub chunk_streaming: MeetingCapabilityReadiness,
    pub diarization: MeetingCapabilityReadiness,
    pub live_summarization: MeetingCapabilityReadiness,
    pub follow_up: MeetingCapabilityReadiness,
    pub capture_health: CaptureHealth,
    pub stt_adapter: MeetingSttAdapterStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionState {
    pub session: MeetingSession,
    pub transcript: Vec<TranscriptEntry>,
    pub summary: Vec<SummaryEntry>,
    pub action_items: Vec<ActionItem>,
    pub decisions: Vec<DecisionLogEntry>,
    pub notes: Vec<NoteEntry>,
    #[serde(default)]
    pub screen_contexts: Vec<MeetingScreenContext>,
    #[serde(default)]
    pub intelligence: Option<MeetingIntelligenceResult>,
    #[serde(default)]
    pub speakers: Vec<SpeakerLabel>,
    #[serde(default)]
    pub speaker_rename_count: u64,
    pub status: MeetingStatus,
    #[serde(default)]
    pub paused_from: Option<MeetingStatus>,
    #[serde(default)]
    pub diagnostics: Vec<MeetingDiagnostic>,
    pub started_at: DateTime<Utc>,
    pub last_updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScreenContextSource {
    #[default]
    ManualCapture,
    UserRequested,
    SessionMarker,
    Imported,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScreenContextAttachmentMode {
    #[default]
    CurrentMoment,
    NearestTranscriptWindow,
    ManualSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScreenContextRedaction {
    MetadataOnly,
    ScreenshotStored,
    #[default]
    ScreenshotNotStored,
    Redacted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingTimeWindow {
    pub start_at: DateTime<Utc>,
    pub end_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScreenArtifactRef {
    pub storage_uri: String,
    pub media_type: String,
    pub bytes: u64,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScreenStructuredObservation {
    pub provider: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub semantic_frame: Option<serde_json::Value>,
    #[serde(default)]
    pub visible_app: Option<String>,
    #[serde(default)]
    pub page_kind: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeetingScreenContext {
    #[serde(default = "new_meeting_artifact_id")]
    pub context_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default = "utc_now")]
    pub captured_at: DateTime<Utc>,
    #[serde(default)]
    pub source: ScreenContextSource,
    #[serde(default)]
    pub attachment_mode: ScreenContextAttachmentMode,
    #[serde(default)]
    pub linked_transcript_segment_ids: Vec<String>,
    #[serde(default)]
    pub linked_time_window: Option<MeetingTimeWindow>,
    pub summary: String,
    #[serde(default)]
    pub structured_observation: Option<ScreenStructuredObservation>,
    #[serde(default)]
    pub screenshot_ref: Option<ScreenArtifactRef>,
    #[serde(default)]
    pub redaction: ScreenContextRedaction,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub diagnostics: Vec<MeetingDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ArtifactGenerator {
    RuleBased,
    LocalLlm { provider: String, model: String },
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingIntelligenceStatus {
    #[default]
    Idle,
    Generating,
    Generated,
    Degraded,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RiskSeverity {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FollowUpTone {
    #[default]
    Professional,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingLanguage {
    Italian,
    English,
    Mixed,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingLanguageSource {
    TranscriptHeuristic,
    UserSourceWeighted,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingSessionType {
    TechnicalDebugging,
    WorkMeeting,
    Planning,
    DecisionReview,
    SupportCall,
    #[default]
    General,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingSessionTypeSource {
    TranscriptHeuristic,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingIntelligenceGenerationOptions {
    #[serde(default)]
    pub use_local_llm: bool,
    #[serde(default = "default_intelligence_max_segments")]
    pub max_transcript_segments: usize,
}

impl Default for MeetingIntelligenceGenerationOptions {
    fn default() -> Self {
        Self {
            use_local_llm: false,
            max_transcript_segments: default_intelligence_max_segments(),
        }
    }
}

fn default_intelligence_max_segments() -> usize {
    120
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSummary {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub text: String,
    #[serde(default)]
    pub bullets: Vec<String>,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub generator: ArtifactGenerator,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingDecision {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub decision: String,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub made_by_speaker_id: Option<String>,
    #[serde(default)]
    pub made_by_display_name: Option<String>,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    pub confidence: f32,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub generator: ArtifactGenerator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingActionItem {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub task: String,
    #[serde(default)]
    pub assignee_speaker_id: Option<String>,
    #[serde(default)]
    pub assignee_display_name: Option<String>,
    #[serde(default)]
    pub due_date: Option<String>,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    pub confidence: f32,
    pub status: ActionItemStatus,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub generator: ArtifactGenerator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingOpenQuestion {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub question: String,
    #[serde(default)]
    pub asked_by_speaker_id: Option<String>,
    #[serde(default)]
    pub asked_by_display_name: Option<String>,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    pub confidence: f32,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub generator: ArtifactGenerator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingRisk {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub risk: String,
    pub severity: RiskSeverity,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    pub confidence: f32,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub generator: ArtifactGenerator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingTechnicalRecap {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub bullets: Vec<String>,
    #[serde(default)]
    pub mentioned_files: Vec<String>,
    #[serde(default)]
    pub mentioned_commands: Vec<String>,
    #[serde(default)]
    pub mentioned_errors: Vec<String>,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    pub confidence: f32,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub generator: ArtifactGenerator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingFollowUpDraft {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub subject: String,
    pub body: String,
    pub tone: FollowUpTone,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    pub confidence: f32,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
    pub generator: ArtifactGenerator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingTimelineItem {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    pub timestamp_ms: Option<u64>,
    #[serde(default)]
    pub speaker_id: Option<String>,
    #[serde(default)]
    pub speaker_display_name: Option<String>,
    pub title: String,
    pub detail: String,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingIntelligenceDiagnostics {
    pub status: MeetingIntelligenceStatus,
    pub generator: ArtifactGenerator,
    #[serde(default)]
    pub model_provider: Option<String>,
    #[serde(default)]
    pub model_name: Option<String>,
    #[serde(default)]
    pub llm_endpoint: Option<String>,
    #[serde(default)]
    pub degraded_reason: Option<String>,
    #[serde(default)]
    pub model_unavailable_reason: Option<String>,
    #[serde(default)]
    pub llm_used: bool,
    pub json_parse_failed: bool,
    pub invalid_evidence_ids: usize,
    pub rejected_artifact_count: usize,
    pub fallback_used: bool,
    #[serde(default)]
    pub input_segment_count: usize,
    #[serde(default)]
    pub input_truncated: bool,
    #[serde(default)]
    pub input_char_count: usize,
    #[serde(default)]
    pub max_segments: usize,
    #[serde(default)]
    pub max_chars_total: usize,
    #[serde(default)]
    pub max_chars_per_segment: usize,
    #[serde(default)]
    pub detected_language: MeetingLanguage,
    #[serde(default)]
    pub language_confidence: f32,
    #[serde(default)]
    pub language_source: MeetingLanguageSource,
    #[serde(default)]
    pub output_language: MeetingLanguage,
    #[serde(default)]
    pub output_language_mismatch: bool,
    #[serde(default)]
    pub language_retry_attempted: bool,
    #[serde(default)]
    pub language_retry_succeeded: bool,
    #[serde(default)]
    pub session_type: MeetingSessionType,
    #[serde(default)]
    pub session_type_confidence: f32,
    #[serde(default)]
    pub session_type_source: MeetingSessionTypeSource,
    #[serde(default)]
    pub llm_generation_duration_ms: Option<u64>,
    #[serde(default)]
    pub total_generation_duration_ms: Option<u64>,
    #[serde(default)]
    pub transcript_changed_during_generation: bool,
    #[serde(default)]
    pub snapshot_transcript_segment_count: usize,
    pub transcript_text_logged: bool,
    pub audit_redacted: bool,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingIntelligenceResult {
    pub session_id: String,
    pub status: MeetingIntelligenceStatus,
    #[serde(default)]
    pub summary: Option<MeetingSummary>,
    #[serde(default)]
    pub decisions: Vec<MeetingDecision>,
    #[serde(default)]
    pub action_items: Vec<MeetingActionItem>,
    #[serde(default)]
    pub open_questions: Vec<MeetingOpenQuestion>,
    #[serde(default)]
    pub risks: Vec<MeetingRisk>,
    #[serde(default)]
    pub technical_recap: Option<MeetingTechnicalRecap>,
    #[serde(default)]
    pub follow_up_draft: Option<MeetingFollowUpDraft>,
    #[serde(default)]
    pub timeline: Vec<MeetingTimelineItem>,
    pub diagnostics: MeetingIntelligenceDiagnostics,
    #[serde(default)]
    pub source_transcript_segment_count: usize,
    #[serde(default = "utc_now")]
    pub generated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    #[serde(default = "new_meeting_artifact_id")]
    pub segment_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub source: TranscriptSource,
    pub timestamp: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    pub speaker: String,
    #[serde(default)]
    pub speaker_id: Option<String>,
    #[serde(default)]
    pub speaker_label: Option<String>,
    #[serde(default)]
    pub speaker_confidence: Option<f32>,
    #[serde(default)]
    pub speaker_attribution_method: SpeakerAttributionMethod,
    pub text: String,
    pub confidence: f32,
    #[serde(default)]
    pub start_ms: Option<u64>,
    #[serde(default)]
    pub end_ms: Option<u64>,
    #[serde(default)]
    pub stt_model: Option<String>,
    #[serde(default)]
    pub audio_backend: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryEntry {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    pub summary: String,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy)]
#[serde(rename_all = "snake_case")]
pub enum ActionItemStatus {
    Open,
    InProgress,
    Closed,
    Deferred,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub title: String,
    pub description: String,
    #[serde(default)]
    pub assignee: Option<ParticipantInfo>,
    #[serde(default)]
    pub deadline: Option<DateTime<Utc>>,
    pub status: ActionItemStatus,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLogEntry {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    pub decision: String,
    pub rationale: String,
    #[serde(default)]
    pub made_by: Option<ParticipantInfo>,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteEntry {
    #[serde(default = "new_meeting_artifact_id")]
    pub id: String,
    #[serde(default)]
    pub session_id: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default = "utc_now")]
    pub created_at: DateTime<Utc>,
    pub content: String,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MeetingDiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingDiagnostic {
    pub code: String,
    pub severity: MeetingDiagnosticSeverity,
    pub message: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMeeting {
    pub session_id: String,
    pub platform: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub participants: Vec<ParticipantInfo>,
    pub transcript: Vec<TranscriptEntry>,
    pub summary: Vec<SummaryEntry>,
    pub action_items: Vec<ActionItem>,
    pub decisions: Vec<DecisionLogEntry>,
    pub notes: Vec<NoteEntry>,
    #[serde(default)]
    pub intelligence: Option<MeetingIntelligenceResult>,
    #[serde(default)]
    pub screen_contexts: Vec<MeetingScreenContext>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionArchiveDocument {
    pub schema_version: u32,
    pub session_id: String,
    pub archived_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub state: MeetingSessionState,
    pub exported: ExportedMeeting,
    #[serde(default)]
    pub screen_contexts: Vec<MeetingScreenContext>,
    pub capture_health: CaptureHealth,
    pub system_capture_health: CaptureHealth,
    pub microphone_capture_health: CaptureHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionListRequest {
    #[serde(default = "default_session_memory_limit")]
    pub limit: usize,
    #[serde(default)]
    pub cursor: Option<String>,
    #[serde(default)]
    pub date_from: Option<DateTime<Utc>>,
    #[serde(default)]
    pub date_to: Option<DateTime<Utc>>,
    #[serde(default)]
    pub has_intelligence: Option<bool>,
    #[serde(default)]
    pub query: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionListResponse {
    pub sessions: Vec<MeetingSessionListItem>,
    #[serde(default)]
    pub next_cursor: Option<String>,
    #[serde(default)]
    pub diagnostics: Vec<MeetingDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeetingSessionListItem {
    pub session_id: String,
    pub title: String,
    pub platform: String,
    pub session_mode: MeetingSessionMode,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub transcript_count: usize,
    pub intelligence_present: bool,
    pub summary_preview: String,
    pub action_item_count: usize,
    pub decision_count: usize,
    pub open_question_count: usize,
    pub risk_count: usize,
    pub technical_recap_present: bool,
    pub speakers_preview: Vec<String>,
    pub capture_sources: Vec<String>,
    pub stt_completeness_status: String,
    #[serde(default)]
    pub stt_completeness_detail: String,
    #[serde(default)]
    pub screen_context_count: usize,
    pub drain_status: String,
    pub last_updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionReadRequest {
    pub session_id: String,
    #[serde(default = "default_true")]
    pub include_transcript: bool,
    #[serde(default = "default_true")]
    pub include_intelligence: bool,
    #[serde(default = "default_true")]
    pub include_diagnostics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionReadResponse {
    pub archive: MeetingSessionArchiveDocument,
    #[serde(default)]
    pub diagnostics: Vec<MeetingDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionSearchRequest {
    pub query: String,
    #[serde(default = "default_session_memory_limit")]
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionSearchResponse {
    pub results: Vec<MeetingSessionSearchResult>,
    pub searched_session_count: usize,
    pub matched_session_count: usize,
    pub truncated: bool,
    pub corrupt_archive_count: usize,
    #[serde(default)]
    pub diagnostics: Vec<MeetingDiagnostic>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeetingSessionSearchResult {
    pub session_id: String,
    pub session_title: String,
    pub matched_kind: String,
    pub title: String,
    pub snippet: String,
    pub score: f32,
    #[serde(default)]
    pub evidence_segment_ids: Vec<String>,
    #[serde(default)]
    pub speaker_display_name: Option<String>,
    #[serde(default)]
    pub timestamp_ms: Option<u64>,
    #[serde(default)]
    pub screen_context_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingScreenContextAttachRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub store_screenshot: bool,
    #[serde(default = "default_true")]
    pub capture_fresh: bool,
    #[serde(default)]
    pub attachment_mode: ScreenContextAttachmentMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingScreenContextAttachResponse {
    pub context: MeetingScreenContext,
    #[serde(default)]
    pub diagnostics: Vec<MeetingDiagnostic>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingSessionExportFormat {
    #[default]
    Markdown,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionExportRequest {
    pub session_id: String,
    #[serde(default)]
    pub format: MeetingSessionExportFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingSessionExportResponse {
    pub session_id: String,
    pub format: MeetingSessionExportFormat,
    pub filename: String,
    pub content: String,
    pub content_length: usize,
    #[serde(default)]
    pub diagnostics: Vec<MeetingDiagnostic>,
}

fn default_session_memory_limit() -> usize {
    20
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CallInfo {
    pub platform: String,
    pub window_title: String,
    pub process_name: String,
    #[serde(rename = "is_active_call")]
    pub is_active_call: bool,
    #[serde(default)]
    pub detection_state: CallDetectionState,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CallDetectionState {
    #[default]
    Idle,
    Detected,
    Likely,
    Confirmed,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MeetingRuntimeError {
    ConsentRequired {
        platform: String,
    },
    PermissionDenied {
        permission: String,
    },
    UnsupportedCapability {
        capability: String,
        reason: String,
    },
    InvalidLifecycleTransition {
        from: MeetingStatus,
        to: MeetingStatus,
    },
    ActiveSessionExists {
        session_id: String,
    },
    NoActiveSession,
    SessionPaused {
        previous_status: Option<MeetingStatus>,
    },
    SessionCompleted,
    InvalidConfig {
        message: String,
    },
    ConfirmationRequired {
        action: String,
        required_phrase: String,
    },
    CaptureUnavailable {
        backend: CaptureBackend,
        reason: String,
    },
    CaptureStartFailed {
        backend: CaptureBackend,
        reason: String,
    },
    CaptureStartupTimeout {
        backend: CaptureBackend,
        timeout_ms: u64,
    },
    CaptureStartupChannelClosed {
        backend: CaptureBackend,
    },
    CaptureStopTimedOut {
        backend: CaptureBackend,
        timeout_ms: u64,
    },
    CaptureDeviceUnavailable {
        backend: CaptureBackend,
        reason: String,
    },
    CaptureStreamError {
        backend: CaptureBackend,
        reason: String,
    },
    ClearAbortedCaptureStopFailed {
        operation: String,
        error_kind: String,
    },
    ConsentRevoked {
        platform: String,
    },
    SegmentWriteFailed {
        reason: String,
    },
    SegmentTooLarge {
        max_bytes: u64,
        actual_bytes: u64,
    },
    TranscriptionUnavailable {
        reason: String,
    },
    SttUnavailable {
        reason: String,
    },
    AudioCaptureUnavailable {
        backend: CaptureBackend,
        reason: String,
    },
    NoAudioFramesReceived {
        source: TranscriptSource,
    },
    TranscriptionInactive,
    TranscriptionFailedWithCleanupWarning {
        reason: String,
        cleanup_requested: bool,
        cleanup_performed: bool,
        cleanup_error: Option<String>,
        managed_path_redacted: bool,
    },
    StorageError {
        message: String,
    },
    SerializationError {
        message: String,
    },
    MutexPoisoned {
        component: String,
    },
}

impl std::fmt::Display for MeetingRuntimeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConsentRequired { platform } => write!(
                formatter,
                "Explicit meeting recording/transcription consent is required for {platform}"
            ),
            Self::PermissionDenied { permission } => {
                write!(formatter, "Meeting permission denied: {permission}")
            }
            Self::UnsupportedCapability { capability, reason } => {
                write!(formatter, "{capability} unsupported: {reason}")
            }
            Self::InvalidLifecycleTransition { from, to } => {
                write!(
                    formatter,
                    "Invalid meeting lifecycle transition from {from:?} to {to:?}"
                )
            }
            Self::ActiveSessionExists { session_id } => {
                write!(
                    formatter,
                    "Active meeting session already exists: {session_id}"
                )
            }
            Self::NoActiveSession => write!(formatter, "No active meeting session"),
            Self::SessionPaused { previous_status } => write!(
                formatter,
                "Meeting session is paused from {:?}",
                previous_status
            ),
            Self::SessionCompleted => write!(formatter, "Meeting session is completed"),
            Self::InvalidConfig { message } => {
                write!(formatter, "Invalid meeting config: {message}")
            }
            Self::ConfirmationRequired {
                action,
                required_phrase,
            } => write!(
                formatter,
                "{action} requires typing the confirmation phrase {required_phrase}"
            ),
            Self::CaptureUnavailable { backend, reason } => {
                write!(
                    formatter,
                    "Capture backend {backend:?} unavailable: {reason}"
                )
            }
            Self::CaptureStartFailed { backend, reason } => {
                write!(
                    formatter,
                    "Capture backend {backend:?} failed to start: {reason}"
                )
            }
            Self::CaptureStartupTimeout {
                backend,
                timeout_ms,
            } => write!(
                formatter,
                "Capture backend {backend:?} did not report startup within {timeout_ms} ms"
            ),
            Self::CaptureStartupChannelClosed { backend } => write!(
                formatter,
                "Capture backend {backend:?} startup channel closed before reporting a result"
            ),
            Self::CaptureStopTimedOut {
                backend,
                timeout_ms,
            } => write!(
                formatter,
                "Capture backend {backend:?} did not stop within {timeout_ms} ms"
            ),
            Self::CaptureDeviceUnavailable { backend, reason } => {
                write!(
                    formatter,
                    "Capture backend {backend:?} device unavailable: {reason}"
                )
            }
            Self::CaptureStreamError { backend, reason } => {
                write!(
                    formatter,
                    "Capture backend {backend:?} stream failed: {reason}"
                )
            }
            Self::ClearAbortedCaptureStopFailed {
                operation,
                error_kind,
            } => write!(
                formatter,
                "{operation} aborted because active capture could not stop safely: {error_kind}"
            ),
            Self::ConsentRevoked { platform } => {
                write!(
                    formatter,
                    "Meeting consent was revoked before segment transcription for {platform}"
                )
            }
            Self::SegmentWriteFailed { reason } => {
                write!(formatter, "Meeting segment write failed: {reason}")
            }
            Self::SegmentTooLarge {
                max_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "Meeting segment is too large: {actual_bytes} bytes exceeds {max_bytes} bytes"
            ),
            Self::TranscriptionUnavailable { reason } => {
                write!(formatter, "Meeting transcription unavailable: {reason}")
            }
            Self::SttUnavailable { reason } => {
                write!(formatter, "Meeting STT unavailable: {reason}")
            }
            Self::AudioCaptureUnavailable { backend, reason } => {
                write!(
                    formatter,
                    "Audio capture backend {backend:?} unavailable: {reason}"
                )
            }
            Self::NoAudioFramesReceived { source } => {
                write!(
                    formatter,
                    "No audio frames received for transcript source {}",
                    source.as_str()
                )
            }
            Self::TranscriptionInactive => {
                write!(formatter, "Meeting transcription is inactive")
            }
            Self::TranscriptionFailedWithCleanupWarning {
                reason,
                cleanup_requested,
                cleanup_performed,
                cleanup_error,
                managed_path_redacted,
            } => write!(
                formatter,
                "Meeting transcription failed: {reason}; cleanup_requested: {cleanup_requested}; cleanup_performed: {cleanup_performed}; cleanup_error: {}; managed_path_redacted: {managed_path_redacted}",
                cleanup_error.as_deref().unwrap_or("none")
            ),
            Self::StorageError { message } => write!(formatter, "{message}"),
            Self::SerializationError { message } => write!(formatter, "{message}"),
            Self::MutexPoisoned { component } => {
                write!(formatter, "Meeting runtime mutex poisoned: {component}")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingFile {
    pub path: String,
    pub filename: String,
    pub size: u64,
    pub checksum: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    pub timestamp: DateTime<Utc>,
    pub samples: Vec<f32>,
    pub device_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerFrame {
    pub timestamp: DateTime<Utc>,
    pub voice_features: Vec<f32>,
    pub speaker_id: Option<String>,
    pub confidence: f32,
    pub audio_frame: AudioFrame,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub id: String,
    pub voice_features: Vec<f32>,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrivacyMode {
    Default,
    Redact,
    Pause,
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyControl {
    pub global_enabled: bool,
    pub mode: String,
    pub consent_given: bool,
    pub per_app_consent: HashMap<String, bool>,
    pub data_retention: DataRetentionPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DataRetentionPolicy {
    pub raw_audio_days: u32,
    pub transcript_days: u32,
    pub summary_days: u32,
    pub action_items_days: u32,
    pub decisions_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingStartRequest {
    pub platform: String,
    pub config: MeetingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingAudioFileTranscriptionRequest {
    #[serde(default)]
    pub session_id: Option<String>,
    pub audio_path: String,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub cleanup_after_transcription: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenameSpeakerRequest {
    pub speaker_id: String,
    pub display_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenameSpeakerResult {
    pub speaker: SpeakerLabel,
    pub renamed_entries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingAudioFileTranscriptionResult {
    pub transcript_added: bool,
    pub transcript_index: usize,
    pub text_length: usize,
    pub audio_file_extension: String,
    pub file_size_bytes: u64,
    pub stt_boundary: String,
    pub transcript_source: TranscriptSource,
    pub segment_id: String,
    #[serde(default)]
    pub start_ms: Option<u64>,
    #[serde(default)]
    pub end_ms: Option<u64>,
    pub source_audio_path_redacted: bool,
    pub managed_audio_path_redacted: bool,
    pub cleanup_requested: bool,
    pub cleanup_performed: bool,
    #[serde(default)]
    pub cleanup_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentState {
    pub given: bool,
    pub per_app: HashMap<String, bool>,
    pub global_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MeetingClearScope {
    #[default]
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearMeetingDataRequest {
    #[serde(default)]
    pub scope: MeetingClearScope,
    pub confirmation_phrase: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingDataClearPreview {
    pub scope: MeetingClearScope,
    pub runtime_state_present: bool,
    pub persisted_entries: usize,
    pub storage_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeetingDataClearResult {
    pub runtime_state_cleared: bool,
    pub persisted_entries_removed: usize,
    pub storage_path: String,
    #[serde(default)]
    pub capture_stop_attempted: bool,
    #[serde(default)]
    pub capture_stop_succeeded: bool,
    #[serde(default)]
    pub capture_stop_error_kind: Option<String>,
    #[serde(default)]
    pub clear_aborted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub session_id: String,
    pub event_type: String,
    pub timestamp: DateTime<Utc>,
    pub details: serde_json::Value,
}

impl TranscriptEntry {
    pub fn sourced(
        session_id: impl Into<String>,
        source: TranscriptSource,
        speaker: impl Into<String>,
        text: impl Into<String>,
        confidence: f32,
    ) -> Self {
        let now = Utc::now();
        Self {
            segment_id: new_meeting_artifact_id(),
            session_id: session_id.into(),
            source,
            timestamp: now,
            created_at: now,
            speaker: speaker.into(),
            speaker_id: None,
            speaker_label: None,
            speaker_confidence: None,
            speaker_attribution_method: SpeakerAttributionMethod::Unknown,
            text: text.into(),
            confidence,
            start_ms: None,
            end_ms: None,
            stt_model: None,
            audio_backend: None,
        }
    }

    pub fn speaker_display_name(&self) -> &str {
        self.speaker_label
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| self.speaker.trim())
    }
}

impl std::fmt::Display for ActionItemStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionItemStatus::Open => write!(f, "Open"),
            ActionItemStatus::InProgress => write!(f, "InProgress"),
            ActionItemStatus::Closed => write!(f, "Closed"),
            ActionItemStatus::Deferred => write!(f, "Deferred"),
        }
    }
}

pub fn new_meeting_artifact_id() -> String {
    Uuid::new_v4().to_string()
}

pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
}
