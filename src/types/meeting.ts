export type CaptureBackend = "pipe_wire" | "core_audio" | "wasapi" | "default";

export type MeetingStatus =
    | "idle"
    | "consent_required"
    | "detecting"
    | "ready"
    | "starting"
    | "capturing"
    | "transcribing"
    | "summarizing"
    | "paused"
    | "stopping"
    | "stopped"
    | "completed"
    | { failed: string }
    | { error: string };

export type MeetingSessionMode = "manual" | "real_capture";
export type TranscriptSource = "microphone" | "system_audio" | "manual" | "imported_file" | "unknown";
export type SpeakerAttributionMethod =
    | "source_default"
    | "user_assigned"
    | "heuristic_turn_split"
    | "diarization_model"
    | "unknown";

export type MeetingCaptureOptions = {
    system_audio: boolean;
    microphone: boolean;
    segment_transcription: boolean;
};

export type ParticipantInfo = {
    name: string;
    speaker_id?: string | null;
};

export type SpeakerLabel = {
    speaker_id: string;
    display_name: string;
    source: TranscriptSource;
    confidence: number;
    attribution_method: SpeakerAttributionMethod;
};

export type MeetingConfig = {
    platform: string;
    capture_backend: CaptureBackend;
    transcription_model: string;
    sample_rate: number;
    diarization_enabled: boolean;
    privacy_mode: string;
    session_mode: MeetingSessionMode;
    live_transcription_enabled: boolean;
    capture_options: MeetingCaptureOptions;
};

export type MeetingSession = {
    session_id: string;
    platform: string;
    status: MeetingStatus;
    started_at: string;
    participants: ParticipantInfo[];
    config: MeetingConfig;
    session_mode: MeetingSessionMode;
    capture_active: boolean;
    capture_backend_status?: string | null;
};

export type TranscriptEntry = {
    segment_id: string;
    session_id: string;
    source: TranscriptSource;
    timestamp: string;
    created_at: string;
    speaker: string;
    speaker_id?: string | null;
    speaker_label?: string | null;
    speaker_confidence?: number | null;
    speaker_attribution_method?: SpeakerAttributionMethod;
    text: string;
    confidence: number;
    start_ms?: number | null;
    end_ms?: number | null;
    stt_model?: string | null;
    audio_backend?: string | null;
};

export type RenameSpeakerRequest = {
    speaker_id: string;
    display_name: string;
};

export type RenameSpeakerResult = {
    speaker: SpeakerLabel;
    renamed_entries: number;
};

export type MeetingAudioFileTranscriptionRequest = {
    session_id?: string | null;
    audio_path: string;
    speaker?: string | null;
    cleanup_after_transcription: boolean;
};

export type MeetingAudioFileTranscriptionResult = {
    transcript_added: boolean;
    transcript_index: number;
    text_length: number;
    audio_file_extension: string;
    file_size_bytes: number;
    stt_boundary: string;
    transcript_source: TranscriptSource;
    segment_id: string;
    start_ms?: number | null;
    end_ms?: number | null;
    source_audio_path_redacted: boolean;
    managed_audio_path_redacted: boolean;
    cleanup_requested: boolean;
    cleanup_performed: boolean;
    cleanup_error?: string | null;
};

export type SummaryEntry = {
    id: string;
    session_id: string;
    timestamp: string;
    created_at: string;
    summary: string;
    evidence_segment_ids: string[];
};

export type ActionItemStatus = "open" | "in_progress" | "closed" | "deferred";

export type ActionItem = {
    id?: string;
    session_id?: string;
    timestamp: string;
    created_at?: string;
    title?: string;
    description: string;
    assignee?: ParticipantInfo | null;
    deadline?: string | null;
    status: ActionItemStatus;
    evidence_segment_ids?: string[];
};

export type DecisionLogEntry = {
    id?: string;
    session_id?: string;
    timestamp: string;
    created_at?: string;
    decision: string;
    rationale: string;
    made_by?: ParticipantInfo | null;
    evidence_segment_ids?: string[];
};

export type NoteEntry = {
    id: string;
    session_id: string;
    timestamp: string;
    created_at: string;
    content: string;
    evidence_segment_ids: string[];
};

export type MeetingDiagnosticSeverity = "info" | "warning" | "error";

export type MeetingDiagnostic = {
    code: string;
    severity: MeetingDiagnosticSeverity;
    message: string;
    created_at: string;
};

export type MeetingSessionState = {
    session: MeetingSession;
    transcript: TranscriptEntry[];
    summary: SummaryEntry[];
    action_items: ActionItem[];
    decisions: DecisionLogEntry[];
    notes: NoteEntry[];
    intelligence?: MeetingIntelligenceResult | null;
    speakers: SpeakerLabel[];
    speaker_rename_count: number;
    status: MeetingStatus;
    paused_from?: MeetingStatus | null;
    diagnostics: MeetingDiagnostic[];
    started_at: string;
    last_updated_at: string;
};

export type ArtifactGenerator =
    | { type: "rule_based" }
    | { type: "local_llm"; provider: string; model: string }
    | { type: "hybrid" };

export type MeetingIntelligenceStatus =
    | "idle"
    | "generating"
    | "generated"
    | "degraded"
    | "failed";

export type RiskSeverity = "low" | "medium" | "high";
export type FollowUpTone = "professional";

export type MeetingIntelligenceGenerationOptions = {
    use_local_llm: boolean;
    max_transcript_segments: number;
};

export type MeetingSummary = {
    id: string;
    session_id: string;
    text: string;
    bullets: string[];
    evidence_segment_ids: string[];
    generated_at: string;
    generator: ArtifactGenerator;
    confidence: number;
};

export type MeetingDecision = {
    id: string;
    session_id: string;
    decision: string;
    rationale?: string | null;
    made_by_speaker_id?: string | null;
    made_by_display_name?: string | null;
    evidence_segment_ids: string[];
    confidence: number;
    generated_at: string;
    generator: ArtifactGenerator;
};

export type MeetingActionItem = {
    id: string;
    session_id: string;
    task: string;
    assignee_speaker_id?: string | null;
    assignee_display_name?: string | null;
    due_date?: string | null;
    evidence_segment_ids: string[];
    confidence: number;
    status: ActionItemStatus;
    generated_at: string;
    generator: ArtifactGenerator;
};

export type MeetingOpenQuestion = {
    id: string;
    session_id: string;
    question: string;
    asked_by_speaker_id?: string | null;
    asked_by_display_name?: string | null;
    evidence_segment_ids: string[];
    confidence: number;
    generated_at: string;
    generator: ArtifactGenerator;
};

export type MeetingRisk = {
    id: string;
    session_id: string;
    risk: string;
    severity: RiskSeverity;
    evidence_segment_ids: string[];
    confidence: number;
    generated_at: string;
    generator: ArtifactGenerator;
};

export type MeetingTechnicalRecap = {
    id: string;
    session_id: string;
    bullets: string[];
    mentioned_files: string[];
    mentioned_commands: string[];
    mentioned_errors: string[];
    evidence_segment_ids: string[];
    confidence: number;
    generated_at: string;
    generator: ArtifactGenerator;
};

export type MeetingFollowUpDraft = {
    id: string;
    session_id: string;
    subject: string;
    body: string;
    tone: FollowUpTone;
    evidence_segment_ids: string[];
    confidence: number;
    generated_at: string;
    generator: ArtifactGenerator;
};

export type MeetingTimelineItem = {
    id: string;
    timestamp_ms?: number | null;
    speaker_id?: string | null;
    speaker_display_name?: string | null;
    title: string;
    detail: string;
    evidence_segment_ids: string[];
};

export type MeetingIntelligenceDiagnostics = {
    status: MeetingIntelligenceStatus;
    generator: ArtifactGenerator;
    model_provider?: string | null;
    model_name?: string | null;
    model_unavailable_reason?: string | null;
    json_parse_failed: boolean;
    invalid_evidence_ids: number;
    rejected_artifact_count: number;
    fallback_used: boolean;
    transcript_text_logged: boolean;
    audit_redacted: boolean;
    warnings: string[];
    generated_at: string;
};

export type MeetingIntelligenceResult = {
    session_id: string;
    status: MeetingIntelligenceStatus;
    summary?: MeetingSummary | null;
    decisions: MeetingDecision[];
    action_items: MeetingActionItem[];
    open_questions: MeetingOpenQuestion[];
    risks: MeetingRisk[];
    technical_recap?: MeetingTechnicalRecap | null;
    follow_up_draft?: MeetingFollowUpDraft | null;
    timeline: MeetingTimelineItem[];
    diagnostics: MeetingIntelligenceDiagnostics;
    source_transcript_segment_count: number;
    generated_at: string;
};

export type ExportedMeeting = {
    session_id: string;
    platform: string;
    started_at: string;
    ended_at: string;
    participants: ParticipantInfo[];
    transcript: TranscriptEntry[];
    summary: SummaryEntry[];
    action_items: ActionItem[];
    decisions: DecisionLogEntry[];
    notes: NoteEntry[];
    intelligence?: MeetingIntelligenceResult | null;
    metadata: unknown;
};

export type CallDetectionState = "idle" | "detected" | "likely" | "confirmed";

export type CallInfo = {
    platform: string;
    window_title: string;
    process_name: string;
    is_active_call: boolean;
    detection_state: CallDetectionState;
    confidence: number;
};

export type ConsentState = {
    given: boolean;
    per_app: Record<string, boolean>;
    global_enabled: boolean;
};

export const CLEAR_MEETING_DATA_CONFIRMATION_PHRASE = "DELETE_MEETING_DATA";

export type MeetingClearScope = "all";

export type ClearMeetingDataRequest = {
    scope: MeetingClearScope;
    confirmation_phrase: string;
};

export type MeetingDataClearPreview = {
    scope: MeetingClearScope;
    runtime_state_present: boolean;
    persisted_entries: number;
    storage_path: string;
};

export type MeetingDataClearResult = {
    runtime_state_cleared: boolean;
    persisted_entries_removed: number;
    storage_path: string;
    capture_stop_attempted: boolean;
    capture_stop_succeeded: boolean;
    capture_stop_error_kind?: string | null;
    clear_aborted: boolean;
};

export type CaptureControllerState =
    | "idle"
    | "unsupported"
    | "starting"
    | "capturing"
    | "paused"
    | "stopping"
    | "failed";

export type CaptureHealthStatus =
    | "idle"
    | "healthy"
    | "unsupported"
    | "backpressure"
    | "consent_revoked"
    | "stop_timed_out"
    | "failed";

export type CaptureOverflowPolicy =
    | "reject_newest"
    | "drop_oldest_and_report"
    | "stop_capture";

export type CapturePipelineConfig = {
    max_queued_chunks: number;
    chunk_duration_ms: number;
    max_memory_bytes: number;
    overflow_policy: CaptureOverflowPolicy;
    max_retries: number;
    max_segments_per_session: number;
    max_consecutive_transcription_failures: number;
    vad_enabled: boolean;
    vad_silence_threshold_pcm: number;
    vad_min_speech_ms: number;
    vad_min_silence_ms: number;
    vad_min_speech_ratio_bps: number;
};

export type EffectiveCapturePipelineConfig = {
    requested_segment_duration_ms: number;
    effective_segment_duration_ms: number;
    min_segment_duration_ms: number;
    max_segment_duration_ms: number;
    requested_max_queue_depth: number;
    effective_max_queue_depth: number;
    requested_max_segments_per_session: number;
    effective_max_segments_per_session: number;
    max_segment_bytes: number;
    estimated_max_session_duration_ms: number;
    duration_clamped: boolean;
};

export type CaptureMetrics = {
    chunks_received: number;
    chunks_dropped: number;
    chunks_transcribed: number;
    wasapi_endpoint_acquired: boolean;
    wasapi_mix_format_detected: boolean;
    wasapi_sample_rate?: number | null;
    wasapi_channel_count?: number | null;
    wasapi_sample_format?: string | null;
    wasapi_buffer_frame_count?: number | null;
    wasapi_stream_initialized: boolean;
    wasapi_stream_started: boolean;
    wasapi_packets_read: number;
    frames_captured: number;
    frames_converted: number;
    silence_frames_skipped: number;
    segments_written: number;
    segments_queued: number;
    segments_transcribed: number;
    segments_dropped: number;
    dropped_silence_segments: number;
    segment_write_failures: number;
    segment_transcription_failures: number;
    segment_transcription_failures_total: number;
    segment_transcription_failures_consecutive: number;
    queue_full_events: number;
    bytes_queued: number;
    max_queue_depth_seen: number;
    backpressure_active: boolean;
    last_segment_status?: string | null;
    last_overflow_policy_applied?: CaptureOverflowPolicy | null;
    last_segment_transcription_error_kind?: string | null;
    last_segment_transcription_failure_at?: string | null;
    vad_speech_frames: number;
    vad_silence_frames: number;
    last_speech_ratio_bps: number;
    last_silence_ratio_bps: number;
    audio_clipped_sample_count: number;
    audio_peak_abs: number;
    audio_rms_bps: number;
    audio_normalization_gain_bps: number;
    last_backend_error_kind?: string | null;
    last_backend_error_message?: string | null;
    last_successful_segment_at?: string | null;
    restarts_attempted: number;
};

export type CaptureHealth = {
    state: CaptureControllerState;
    status: CaptureHealthStatus;
    backend?: CaptureBackend | null;
    active_handle_present: boolean;
    backpressure_active: boolean;
    last_error?: string | null;
    last_segment_status?: string | null;
    last_overflow_policy_applied?: CaptureOverflowPolicy | null;
    pipeline: CapturePipelineConfig;
    effective_pipeline: EffectiveCapturePipelineConfig;
    metrics: CaptureMetrics;
};

export type MeetingCapabilityState = "ready" | "disabled" | "unavailable" | "error";

export type MeetingCapabilityReadiness = {
    capability: string;
    available: boolean;
    state: MeetingCapabilityState;
    reason?: string | null;
};

export type MeetingSttAdapterStatus = {
    state: MeetingCapabilityState;
    existing_boundary: string;
    file_transcription: MeetingCapabilityReadiness;
    live_transcription: MeetingCapabilityReadiness;
    chunk_streaming: MeetingCapabilityReadiness;
    chunk_streaming_supported: boolean;
    emits_placeholder_transcripts: boolean;
    reason?: string | null;
};

export type MeetingLiveCapabilitySnapshot = {
    manual_session: MeetingCapabilityReadiness;
    audio_capture: MeetingCapabilityReadiness;
    microphone_capture: MeetingCapabilityReadiness;
    system_audio_capture: MeetingCapabilityReadiness;
    windows_wasapi_capture: MeetingCapabilityReadiness;
    system_capture_health: CaptureHealth;
    microphone_capture_health: CaptureHealth;
    live_transcription: MeetingCapabilityReadiness;
    live_segment_transcription: MeetingCapabilityReadiness;
    live_streaming_stt: MeetingCapabilityReadiness;
    chunk_streaming: MeetingCapabilityReadiness;
    diarization: MeetingCapabilityReadiness;
    live_summarization: MeetingCapabilityReadiness;
    follow_up: MeetingCapabilityReadiness;
    capture_health: CaptureHealth;
    stt_adapter: MeetingSttAdapterStatus;
};
