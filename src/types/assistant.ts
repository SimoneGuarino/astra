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
    audio_response_enabled: boolean;
};

export type AssistantRequestStartedEvent = {
    request_id: string;
    model: string;
    source: "typed" | "voice_session" | string;
    user_message: string | null;
    audio_response_enabled: boolean;
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
    tts_enabled: boolean;
    tts_skipped_reason: string | null;
    tts_segments_queued: number;
    tts_segments_synthesized: number;
    tts_segments_failed: number;
    tts_segments_skipped_budget: number;
    tts_chars_requested: number;
    tts_chars_queued: number;
};

export type AssistantRouterDiagnostic = {
    request_id: string | null;
    router_called: boolean;
    model: string | null;
    endpoint_label: string | null;
    route: string | null;
    tool: string | null;
    target_kind: string | null;
    confidence: number | null;
    reason_code: string | null;
    failure_reason: string | null;
    used_json_mode: boolean;
    duration_ms: number | null;
    fallback_kind: string | null;
    repair_attempted: boolean;
    repair_succeeded: boolean;
    prompt_char_count: number | null;
    full_router_invoked_reason: string | null;
    pending_governed_action_present: boolean;
    pending_governed_action_tool: string | null;
    pending_governed_action_status: string | null;
    pending_governed_action_expired: boolean | null;
    pending_governed_action_policy_action: string | null;
    pending_governed_action_retry_attempted: boolean | null;
    pending_continuation_decision: string | null;
    pending_continuation_reason: string | null;
    pending_continuation_model_called: boolean | null;
    pending_continuation_model_failure: string | null;
    pending_continuation_safe_to_ignore: boolean | null;
    metadata_only: boolean;
    raw_message_included: boolean;
    raw_router_prompt_included: boolean;
    raw_model_output_included: boolean;
    transcript_text_included: boolean;
    answer_text_included: boolean;
    screen_summary_included: boolean;
};

export type AssistantOrchestratorDiagnostic = {
    request_id: string | null;
    stage: string;
    planner_stage: string;
    working_context_present: boolean;
    last_tool_result_present: boolean;
    selected_route: string | null;
    context_ref: string | null;
    planner_model: string | null;
    planner_duration_ms: number | null;
    planner_failure_reason: string | null;
    planner_confidence: number | null;
    policy_action: string | null;
    fallback_policy: string | null;
    fallback_reason: string | null;
    planner_empty: boolean;
    used_context_boundary_fallback: boolean;
    normal_chat_context_injected: boolean;
    normal_chat_bypassed_tool_router: boolean | null;
    normal_chat_policy_action: string | null;
    normal_chat_policy_reason: string | null;
    normal_chat_direct_confidence_threshold: number | null;
    normal_chat_accepted_directly: boolean | null;
    planner_intent_kind: string | null;
    planner_capability_family: string | null;
    planner_requires_tool_arbitration: boolean | null;
    planner_requires_memory_lookup: boolean | null;
    planner_requires_governed_action: boolean | null;
    planner_requires_context_boundary: boolean | null;
    planner_safe_to_bypass_tools: boolean | null;
    tool_router_invoked_reason: string | null;
    needs_tool_policy_action: string | null;
    needs_tool_policy_reason: string | null;
    needs_tool_confidence_threshold: number | null;
    needs_tool_accepted: boolean | null;
    tool_affinity_source: string | null;
    accepted_decision: boolean;
    prompt_char_count: number;
    prompt_budget_exceeded: boolean;
    used_full_router: boolean;
    tool_affinity_risk: boolean | null;
    context_salience_score: number | null;
    context_turn_age: number | null;
    context_stale: boolean | null;
    context_decay_action: string | null;
    context_continuation_policy_action: string | null;
    context_continuation_policy_reason: string | null;
    context_answer_first_attempted: boolean | null;
    context_answer_fallback_used: boolean | null;
    context_answer_empty_model_content: boolean | null;
    expected_language: string | null;
    output_language: string | null;
    language_mismatch: boolean | null;
    language_retry_attempted: boolean | null;
    language_retry_succeeded: boolean | null;
    budget_compaction_applied: boolean | null;
    user_facing_context_label: string | null;
    sanitized_internal_context_refs: boolean;
    tool_manifest_count: number;
    metadata_only: boolean;
};

export type AssistantToolSynthesisDiagnostic = {
    request_id: string | null;
    model: string | null;
    endpoint_label: string | null;
    source_kind: string;
    evidence_count: number;
    evidence_chars: number;
    used_json_mode: boolean;
    duration_ms: number | null;
    status: string | null;
    failure_reason: string | null;
    fallback_used: boolean;
    repair_attempted: boolean;
    repair_succeeded: boolean;
    metadata_only: boolean;
    raw_message_included: boolean;
    raw_prompt_included: boolean;
    raw_model_output_included: boolean;
    transcript_text_included: boolean;
    answer_text_included: boolean;
    screen_summary_included: boolean;
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
