export type AssistantStatus =
    | "idle"
    | "passive"
    | "armed"
    | "thinking"
    | "listening"
    | "speaking"
    | "settling";

export type ChatMessage = {
    id: string;
    role: "user" | "assistant";
    content: string;
};

export type StartChatResponse = {
    request_id: string;
    model: string;
};

export type AssistantRequestStartedEvent = {
    request_id: string;
    model: string;
    source: "typed" | "voice_session" | string;
    user_message: string | null;
};

export type AssistantRequestFinishedEvent = {
    request_id: string;
    full_text: string;
};

export type AssistantRequestSettledEvent = {
    request_id: string;
    had_tts_failures: boolean;
};

export type AssistantErrorEvent = {
    request_id: string;
    stage: string;
    message: string;
};

export type StreamChunkEvent = {
    request_id: string;
    chunk: string;
};

export type SpeechSegmentQueuedEvent = {
    request_id: string;
    segment_id: string;
    sequence: number;
    text: string;
};

export type RequestMetricsSnapshot = {
    request_id: string;
    selected_model: string;
    user_message_length: number;
    request_started_at: number;
    first_llm_chunk_at: number | null;
    llm_completed_at: number | null;
    first_segment_queued_at: number | null;
    first_audio_ready_at: number | null;
    first_audio_play_at: number | null;
    audio_completed_at: number | null;
    total_request_duration_ms: number | null;
    time_to_first_llm_chunk_ms: number | null;
    time_to_llm_completed_ms: number | null;
    time_to_first_segment_queued_ms: number | null;
    time_to_first_audio_ready_ms: number | null;
    time_to_first_audio_play_ms: number | null;
};

export type VoiceTranscriptionResponse = {
    request_id: string;
    text: string;
    auto_submit: boolean;
};

export type VoiceSessionStartResponse = {
    session_id: string;
};

export type VoiceSessionState =
    | "disabled"
    | "passive"
    | "armed"
    | "listening"
    | "processing"
    | "speaking"
    | "interrupted"
    | "cooldown";

export type VoiceSessionMode = "passive" | "conversation";

export type VoiceSessionStateEvent = {
    session_id: string | null;
    turn_id: string | null;
    state: VoiceSessionState;
    mode: VoiceSessionMode;
    reason: string;
    conversation_expires_in_ms: number | null;
    vad: VadFrameSnapshot;
};

export type VadFrameSnapshot = {
    backend: string;
    rms: number;
    smoothed_rms: number;
    noise_floor: number;
    start_threshold: number;
    end_threshold: number;
    start_gate_ms: number;
    speech_ms: number;
    silence_ms: number;
    utterance_ms: number;
    in_speech: boolean;
};

export type VoiceTurnMetricsSnapshot = {
    session_id: string;
    turn_id: string;
    vad_backend: string;
    utterance_started_at: number | null;
    utterance_ended_at: number | null;
    stt_started_at: number | null;
    stt_completed_at: number | null;
    wake_detected_at: number | null;
    response_started_at: number | null;
    first_llm_chunk_at: number | null;
    first_segment_queued_at: number | null;
    first_audio_ready_at: number | null;
    first_audio_play_at: number | null;
    interruption_detected_at: number | null;
    interruption_stop_completed_at: number | null;
    follow_up_window_opened_at: number | null;
    follow_up_window_closed_at: number | null;
    action: string | null;
    reason: string | null;
    transcript_length: number | null;
    request_id: string | null;
    utterance_duration_ms: number | null;
    speech_to_stt_ms: number | null;
    user_end_to_stt_ms: number | null;
    stt_duration_ms: number | null;
    stt_to_response_start_ms: number | null;
    user_end_to_response_start_ms: number | null;
    response_start_to_first_audio_ms: number | null;
    interruption_latency_ms: number | null;
};

export type VoiceSessionTranscriptEvent = {
    session_id: string;
    turn_id: string;
    text: string;
    accepted: boolean;
    reason: string;
    action: "ignored" | "armed" | "responding" | string;
    response_text: string | null;
};

export type AssistantInterruptedEvent = {
    request_id: string | null;
    reason: string;
};
