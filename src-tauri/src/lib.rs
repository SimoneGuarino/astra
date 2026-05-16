mod accessibility_layer;
mod action_policy;
mod action_resolution;
mod assistant_context;
mod assistant_memory;
mod assistant_response;
mod audio_files;
mod audit_log;
mod browser_agent;
mod capability_manifest;
mod contextual_learning;
mod conversation_history;
mod conversation_router;
mod desktop_agent;
mod desktop_agent_types;
mod filesystem_service;
pub mod meeting;
mod metrics;
mod model_assisted_planner;
mod model_routing;
mod pending_approvals_store;
mod permissions;
mod planner_evaluation;
mod screen_capture;
mod screen_vision;
mod screen_workflow;
mod semantic_frame;
mod semantic_intent;
mod speech_events;
mod structured_vision;
mod stt_client;
mod terminal_runner;
mod text_segmentation;
mod tools_registry;
mod tts_client;
mod ui_control;
mod ui_target_grounding;
mod vad;
mod voice_metrics;
mod voice_session;
mod workflow_continuation;

use assistant_context::build_capability_context;
use assistant_memory::RecentArtifactMemory;
use assistant_response::{
    fallback_display_for_empty_response, present_display_text, render_action_response,
    speech_safe_text, RenderedAssistantResponse, StreamPresentationState,
};
use audio_files::AudioFileRegistry;
use conversation_history::ConversationHistoryManager;
use conversation_router::{route_message, ConversationRoute};
use desktop_agent::DesktopAgentRuntime;
use desktop_agent_types::{
    ApprovalDecisionRequest, CapabilityManifest, ConversationRouteDiagnostic, DesktopActionRequest,
    DesktopActionResponse, DesktopAuditEvent, DesktopPolicySnapshot, GoalLoopRun, PendingApproval,
    ScreenAnalysisRequest, ScreenAnalysisResult, ScreenCaptureResult, ScreenObservationStatus,
    ToolDescriptor,
};
use futures_util::StreamExt;
use meeting::{
    runtime::MeetingRuntime,
    types::{
        ActionItem, CallInfo, CaptureBackend, ClearMeetingDataRequest, ConsentState,
        DecisionLogEntry, ExportedMeeting, MeetingAudioFileTranscriptionRequest,
        MeetingAudioFileTranscriptionResult, MeetingConfig, MeetingDataClearPreview,
        MeetingDataClearResult, MeetingDiagnostic, MeetingFollowUpDraft,
        MeetingIntelligenceGenerationOptions, MeetingIntelligenceResult,
        MeetingLiveCapabilitySnapshot, MeetingSession, MeetingSessionMode, MeetingSessionState,
        NoteEntry, RenameSpeakerRequest, RenameSpeakerResult, SummaryEntry, TranscriptEntry,
    },
};
use metrics::{MetricsTracker, RequestMetricsSnapshot};
use model_routing::resolve_ollama_request;
use reqwest::Client;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use speech_events::{
    AssistantErrorEvent, AssistantInterruptedEvent, AssistantRequestFinishedEvent,
    AssistantRequestSettledEvent, AssistantRequestStartedEvent, AudioPlaybackEvent,
    AudioSegmentFailedEvent, AudioSessionCompletedRequest, ChatStartRequest,
    SpeechSegmentQueuedEvent, StartChatResponse, StreamChunkEvent, VoiceSessionAudioChunk,
    VoiceSessionStartResponse, VoiceSessionStateEvent, VoiceSessionTranscriptEvent,
    VoiceTranscriptionFinishedEvent, VoiceTranscriptionRequest, VoiceTranscriptionResponse,
    VoiceTranscriptionStartedEvent,
};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use stt_client::SttClient;
use tauri::{Emitter, Manager, State, WebviewWindow};
use text_segmentation::{SentenceSegmenter, SpeechSegment};
use tts_client::TtsClient;
use uuid::Uuid;
use voice_metrics::{VoiceMetricsTracker, VoiceTurnMetricsSnapshot};
use voice_session::{
    TranscriptDecision, VoiceSessionAction, VoiceSessionManager, VoiceSessionSnapshot,
};

#[derive(Debug, Serialize, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaStreamChunk {
    message: Option<OllamaMessage>,
    done: Option<bool>,
}

#[derive(Clone)]
struct AssistantRuntime {
    active_request_id: Arc<Mutex<Option<String>>>,
    active_voice_request_id: Arc<Mutex<Option<String>>>,
    audio_files: AudioFileRegistry,
    metrics: MetricsTracker,
    stt_client: SttClient,
    tts_client: TtsClient,
    voice_metrics: VoiceMetricsTracker,
    voice_session: VoiceSessionManager,
    conversation_history: ConversationHistoryManager,
    desktop_agent: DesktopAgentRuntime,
    recent_artifacts: RecentArtifactMemory,
    tts_segment_fingerprints: Arc<Mutex<HashMap<String, HashSet<String>>>>,
    meeting_runtime: MeetingRuntime,
}

impl AssistantRuntime {
    fn new(project_root: PathBuf) -> Self {
        let audio_files = AudioFileRegistry::new(project_root.clone());
        if let Err(error) = audio_files.ensure_generated_dir() {
            eprintln!(
                "{}",
                serde_json::json!({
                    "type": "audio_file_cleanup",
                    "event": "generated_dir_setup_failed",
                    "error": error,
                })
            );
        }
        audio_files.cleanup_stale_files();
        let stt_client = SttClient::new(project_root.clone());

        Self {
            active_request_id: Arc::new(Mutex::new(None)),
            active_voice_request_id: Arc::new(Mutex::new(None)),
            audio_files,
            metrics: MetricsTracker::new(),
            stt_client: stt_client.clone(),
            tts_client: TtsClient::new(project_root.clone()),
            voice_metrics: VoiceMetricsTracker::new(),
            voice_session: VoiceSessionManager::new(project_root.clone()),
            conversation_history: ConversationHistoryManager::new(),
            desktop_agent: DesktopAgentRuntime::new(project_root.clone()),
            recent_artifacts: RecentArtifactMemory::default(),
            tts_segment_fingerprints: Arc::new(Mutex::new(HashMap::new())),
            meeting_runtime: MeetingRuntime::with_stt_client(project_root, stt_client),
        }
    }

    fn begin_request(&self, request_id: String) {
        let mut active_request_id = self
            .active_request_id
            .lock()
            .expect("active_request_id mutex poisoned");
        if let Some(previous_request_id) = active_request_id.take() {
            self.audio_files.cleanup_request(&previous_request_id);
            self.conversation_history.discard_turn(&previous_request_id);
            self.clear_tts_fingerprints(&previous_request_id);
        }
        *active_request_id = Some(request_id);
        self.tts_client.cancel_all();
    }

    fn cancel_active_request(&self) {
        self.tts_client.cancel_all();
        let mut active_request_id = self
            .active_request_id
            .lock()
            .expect("active_request_id mutex poisoned");
        if let Some(previous_request_id) = active_request_id.take() {
            self.audio_files.cleanup_request(&previous_request_id);
            self.conversation_history.discard_turn(&previous_request_id);
            self.clear_tts_fingerprints(&previous_request_id);
        }
    }

    fn interrupt_active_for_replacement(&self) -> Option<String> {
        self.tts_client.cancel_all();
        let mut active_request_id = self
            .active_request_id
            .lock()
            .expect("active_request_id mutex poisoned");
        let previous_request_id = active_request_id.take()?;
        self.audio_files.cleanup_request(&previous_request_id);
        self.conversation_history.discard_turn(&previous_request_id);
        self.clear_tts_fingerprints(&previous_request_id);
        Some(previous_request_id)
    }

    fn finish_request(&self, request_id: &str) {
        let mut active_request_id = self
            .active_request_id
            .lock()
            .expect("active_request_id mutex poisoned");
        if active_request_id.as_deref() == Some(request_id) {
            *active_request_id = None;
        }
        self.clear_tts_fingerprints(request_id);
    }

    fn begin_voice_request(&self, request_id: String) {
        let mut active_voice_request_id = self
            .active_voice_request_id
            .lock()
            .expect("active_voice_request_id mutex poisoned");
        *active_voice_request_id = Some(request_id);
        self.stt_client.cancel_all();
    }

    fn cancel_voice_request(&self) {
        self.stt_client.cancel_all();
        let mut active_voice_request_id = self
            .active_voice_request_id
            .lock()
            .expect("active_voice_request_id mutex poisoned");
        *active_voice_request_id = None;
    }

    fn finish_voice_request(&self, request_id: &str) {
        let mut active_voice_request_id = self
            .active_voice_request_id
            .lock()
            .expect("active_voice_request_id mutex poisoned");
        if active_voice_request_id.as_deref() == Some(request_id) {
            *active_voice_request_id = None;
        }
    }

    fn is_active(&self, request_id: &str) -> bool {
        let active_request_id = self
            .active_request_id
            .lock()
            .expect("active_request_id mutex poisoned");
        active_request_id.as_deref() == Some(request_id)
    }

    fn is_voice_active(&self, request_id: &str) -> bool {
        let active_voice_request_id = self
            .active_voice_request_id
            .lock()
            .expect("active_voice_request_id mutex poisoned");
        active_voice_request_id.as_deref() == Some(request_id)
    }

    fn should_synthesize_segment(&self, request_id: &str, text: &str) -> bool {
        let fingerprint = tts_segment_fingerprint(text);
        if fingerprint.is_empty() || fingerprint == "ho completato la richiesta" {
            return false;
        }

        let mut fingerprints = self
            .tts_segment_fingerprints
            .lock()
            .expect("tts_segment_fingerprints mutex poisoned");
        fingerprints
            .entry(request_id.to_string())
            .or_default()
            .insert(fingerprint)
    }

    fn clear_tts_fingerprints(&self, request_id: &str) {
        let mut fingerprints = self
            .tts_segment_fingerprints
            .lock()
            .expect("tts_segment_fingerprints mutex poisoned");
        fingerprints.remove(request_id);
    }
}

fn project_root() -> Result<PathBuf, String> {
    let tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    tauri_dir
        .parent()
        .map(|path| path.to_path_buf())
        .ok_or_else(|| "Unable to resolve project root".to_string())
}

#[tauri::command]
async fn start_chat_message_stream(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    payload: ChatStartRequest,
) -> Result<StartChatResponse, String> {
    let message = payload.message.trim().to_string();
    if message.is_empty() {
        return Err("Message is empty".to_string());
    }

    start_assistant_response(
        window,
        state.inner().clone(),
        message.clone(),
        Some(message),
        "typed",
    )
    .await
}

async fn start_assistant_response(
    window: WebviewWindow,
    runtime: AssistantRuntime,
    message: String,
    display_user_message: Option<String>,
    source: &str,
) -> Result<StartChatResponse, String> {
    if let Some(previous_request_id) = runtime.interrupt_active_for_replacement() {
        let _ = window.emit(
            "assistant-interrupted",
            AssistantInterruptedEvent {
                request_id: Some(previous_request_id),
                reason: "replaced_by_new_request".into(),
            },
        );
    }

    let history = runtime.conversation_history.recent_messages(10);
    let manifest = runtime.desktop_agent.capability_manifest().await;

    let route_result = route_message(&runtime.desktop_agent, &manifest, &message).await?;
    emit_route_diagnostic(&window, &route_result.diagnostic);

    match route_result.route {
        ConversationRoute::DirectResponse(response_text) => {
            return start_grounded_response(
                window,
                runtime,
                message,
                display_user_message,
                source,
                RenderedAssistantResponse::from_display(response_text),
                "capability-router",
            )
            .await;
        }
        ConversationRoute::ActionResponse(action_response) => {
            runtime
                .recent_artifacts
                .remember_action_response(&action_response);
            let rendered = render_action_response(&action_response, &message);
            return start_grounded_response(
                window,
                runtime,
                message,
                display_user_message,
                source,
                rendered,
                "desktop-agent",
            )
            .await;
        }
        ConversationRoute::ScreenAnalysis(result) => {
            if let Some(analysis) = result.analysis.as_ref() {
                runtime.recent_artifacts.remember_screen_analysis(analysis);
            }
            return start_grounded_response(
                window,
                runtime,
                message,
                display_user_message,
                source,
                RenderedAssistantResponse::from_display(result.response_text),
                "screen-vision",
            )
            .await;
        }
        ConversationRoute::Continue => {}
    }

    if let Some(memory_response) = runtime.recent_artifacts.answer_followup(&message) {
        emit_route_diagnostic(
            &window,
            &recent_artifact_diagnostic(&message, "recent_artifact_memory"),
        );
        return start_grounded_response(
            window,
            runtime,
            message,
            display_user_message,
            source,
            RenderedAssistantResponse::from_display(memory_response),
            "artifact-memory",
        )
        .await;
    }

    let request_id = Uuid::new_v4().to_string();
    let assistant_context = build_capability_context(&manifest);
    let resolved =
        resolve_ollama_request(&message, source, &history, Some(&assistant_context)).await?;
    let model = resolved.model.clone();

    runtime.begin_request(request_id.clone());
    let history_user_message = display_user_message
        .clone()
        .unwrap_or_else(|| message.clone());
    runtime
        .conversation_history
        .begin_turn(request_id.clone(), &history_user_message);

    let metrics_snapshot =
        runtime
            .metrics
            .start_request(request_id.clone(), model.clone(), message.chars().count());

    emit_request_started(
        &window,
        &request_id,
        &model,
        source,
        display_user_message.clone(),
    )?;
    emit_metrics_update(&window, &metrics_snapshot);
    window
        .emit("assistant-status", "thinking")
        .map_err(|error| format!("assistant-status emit failed: {error}"))?;

    let task_window = window.clone();
    let task_runtime = runtime.clone();
    let task_request_id = request_id.clone();
    tauri::async_runtime::spawn(async move {
        let result = run_ollama_stream(
            task_window.clone(),
            task_runtime.clone(),
            task_request_id.clone(),
            message.clone(),
            resolved,
        )
        .await;
        if let Err(message) = result {
            task_runtime
                .conversation_history
                .discard_turn(&task_request_id);
            if task_runtime.is_active(&task_request_id) {
                emit_error(&task_window, &task_request_id, "ollama", message);
                let _ = task_window.emit("assistant-status", "idle");
            }
        }
    });

    Ok(StartChatResponse { request_id, model })
}

async fn start_grounded_response(
    window: WebviewWindow,
    runtime: AssistantRuntime,
    original_message: String,
    display_user_message: Option<String>,
    source: &str,
    rendered: RenderedAssistantResponse,
    model_label: &str,
) -> Result<StartChatResponse, String> {
    let display_text = rendered.display_text;
    let speech_text = rendered.speech_text;
    let request_id = Uuid::new_v4().to_string();
    runtime.begin_request(request_id.clone());
    let history_user_message = display_user_message
        .clone()
        .unwrap_or_else(|| original_message.clone());
    runtime
        .conversation_history
        .begin_turn(request_id.clone(), &history_user_message);
    let metrics_snapshot = runtime.metrics.start_request(
        request_id.clone(),
        model_label.to_string(),
        original_message.chars().count(),
    );
    emit_request_started(
        &window,
        &request_id,
        model_label,
        source,
        display_user_message,
    )?;
    emit_metrics_update(&window, &metrics_snapshot);
    window
        .emit("assistant-status", "thinking")
        .map_err(|error| format!("assistant-status emit failed: {error}"))?;
    runtime
        .conversation_history
        .commit_turn(&request_id, &display_text);
    window
        .emit(
            "assistant-stream-chunk",
            StreamChunkEvent {
                request_id: request_id.clone(),
                chunk: display_text.clone(),
            },
        )
        .map_err(|error| format!("assistant-stream-chunk emit failed: {error}"))?;
    if let Some(snapshot) = runtime.metrics.mark_first_llm_chunk(&request_id) {
        emit_metrics_update(&window, &snapshot);
    }
    if let Some(snapshot) = runtime.metrics.mark_llm_completed(&request_id) {
        emit_metrics_update(&window, &snapshot);
    }
    let mut segmenter = SentenceSegmenter::new();
    for segment in segmenter.push(&speech_text) {
        spawn_tts_segment(window.clone(), runtime.clone(), request_id.clone(), segment);
    }
    for segment in segmenter.flush() {
        spawn_tts_segment(window.clone(), runtime.clone(), request_id.clone(), segment);
    }
    window
        .emit(
            "assistant-request-finished",
            AssistantRequestFinishedEvent {
                request_id: request_id.clone(),
                full_text: display_text,
            },
        )
        .map_err(|error| format!("assistant-request-finished emit failed: {error}"))?;
    window
        .emit("assistant-status", "settling")
        .map_err(|error| format!("assistant-status settling emit failed: {error}"))?;
    Ok(StartChatResponse {
        request_id,
        model: model_label.to_string(),
    })
}

fn emit_route_diagnostic(window: &WebviewWindow, diagnostic: &ConversationRouteDiagnostic) {
    let _ = window.emit("assistant-route-diagnostic", diagnostic.clone());
}

fn recent_artifact_diagnostic(message: &str, routed_to: &str) -> ConversationRouteDiagnostic {
    ConversationRouteDiagnostic {
        message_excerpt: message.chars().take(160).collect(),
        classifier_source: "recent_artifact_memory".into(),
        intent: "artifact_followup".into(),
        target: Some("recent_artifact".into()),
        action: Some("answer_from_memory".into()),
        tool_name: None,
        extracted_params: None,
        confidence: Some(0.70),
        routed_to: routed_to.into(),
        grounded: true,
        fallback_used: false,
        submit_action_called: false,
        action_id: None,
        action_status: None,
        approval_created: false,
        audit_expected: false,
        rationale: Some(
            "Resolved an unambiguous follow-up against session-scoped recent artifact memory"
                .into(),
        ),
        error: None,
    }
}

async fn run_ollama_stream(
    window: WebviewWindow,
    runtime: AssistantRuntime,
    request_id: String,
    original_message: String,
    resolved: model_routing::ResolvedOllamaRequest,
) -> Result<(), String> {
    let client = Client::new();
    let response = client
        .post("http://127.0.0.1:11434/api/chat")
        .json(&serde_json::json!({
            "model": resolved.model,
            "stream": true,
            "messages": resolved.messages,
            "options": resolved.options,
            "keep_alive": "30m"
        }))
        .send()
        .await
        .map_err(|error| format!("Ollama request failed: {error}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .map_err(|error| format!("Ollama error body read failed: {error}"))?;
        return Err(format!("Ollama HTTP error {status}: {body}"));
    }

    let mut stream = response.bytes_stream();
    let mut stream_buffer = String::new();
    let mut full_text = String::new();
    let mut presentation = StreamPresentationState::new();
    let mut emitted_display_text = false;

    while let Some(item) = stream.next().await {
        if !runtime.is_active(&request_id) {
            println!("Ollama stream cancelled for request_id={request_id}");
            runtime.conversation_history.discard_turn(&request_id);
            return Ok(());
        }

        let chunk = item.map_err(|error| format!("Ollama stream read failed: {error}"))?;
        let text = String::from_utf8_lossy(&chunk);
        stream_buffer.push_str(&text);

        while let Some(newline_index) = stream_buffer.find('\n') {
            let line = stream_buffer[..newline_index].trim().to_string();
            stream_buffer = stream_buffer[newline_index + 1..].to_string();

            if line.is_empty() {
                continue;
            }

            emitted_display_text |= process_ollama_line(
                &window,
                &runtime,
                &request_id,
                &line,
                &mut full_text,
                &mut presentation,
            )?;
        }
    }

    let trailing = stream_buffer.trim().to_string();
    if !trailing.is_empty() && runtime.is_active(&request_id) {
        emitted_display_text |= process_ollama_line(
            &window,
            &runtime,
            &request_id,
            &trailing,
            &mut full_text,
            &mut presentation,
        )?;
    }

    if runtime.is_active(&request_id) {
        let trailing_display = presentation.finish();
        if !trailing_display.trim().is_empty() {
            if !emitted_display_text {
                if let Some(snapshot) = runtime.metrics.mark_first_llm_chunk(&request_id) {
                    emit_metrics_update(&window, &snapshot);
                }
                if let Some(snapshot) = runtime
                    .voice_metrics
                    .mark_first_llm_chunk_for_request(&request_id)
                {
                    emit_voice_metrics_update(&window, &snapshot);
                }
            }
            emitted_display_text = true;
            window
                .emit(
                    "assistant-stream-chunk",
                    StreamChunkEvent {
                        request_id: request_id.clone(),
                        chunk: trailing_display.clone(),
                    },
                )
                .map_err(|error| format!("assistant-stream-chunk emit failed: {error}"))?;
        }

        let mut final_text = present_display_text(&full_text);
        if final_text.trim().is_empty() {
            final_text = fallback_display_for_empty_response(&original_message);
        }

        if !emitted_display_text {
            if let Some(snapshot) = runtime.metrics.mark_first_llm_chunk(&request_id) {
                emit_metrics_update(&window, &snapshot);
            }
            if let Some(snapshot) = runtime
                .voice_metrics
                .mark_first_llm_chunk_for_request(&request_id)
            {
                emit_voice_metrics_update(&window, &snapshot);
            }
            window
                .emit(
                    "assistant-stream-chunk",
                    StreamChunkEvent {
                        request_id: request_id.clone(),
                        chunk: final_text.clone(),
                    },
                )
                .map_err(|error| format!("assistant-stream-chunk emit failed: {error}"))?;
        }

        runtime
            .conversation_history
            .commit_turn(&request_id, &final_text);

        if let Some(snapshot) = runtime.metrics.mark_llm_completed(&request_id) {
            emit_metrics_update(&window, &snapshot);
        }

        let mut segmenter = SentenceSegmenter::new();
        let speech_text = speech_safe_text(&final_text);
        if !speech_text.trim().is_empty() {
            for segment in segmenter.push(&speech_text) {
                spawn_tts_segment(window.clone(), runtime.clone(), request_id.clone(), segment);
            }
        }
        for segment in segmenter.flush() {
            spawn_tts_segment(window.clone(), runtime.clone(), request_id.clone(), segment);
        }

        window
            .emit(
                "assistant-request-finished",
                AssistantRequestFinishedEvent {
                    request_id: request_id.clone(),
                    full_text: final_text,
                },
            )
            .map_err(|error| format!("assistant-request-finished emit failed: {error}"))?;

        window
            .emit("assistant-status", "settling")
            .map_err(|error| format!("assistant-status settling emit failed: {error}"))?;
    }

    if !runtime.is_active(&request_id) {
        runtime.conversation_history.discard_turn(&request_id);
    }

    Ok(())
}

fn process_ollama_line(
    window: &WebviewWindow,
    runtime: &AssistantRuntime,
    request_id: &str,
    line: &str,
    full_text: &mut String,
    presentation: &mut StreamPresentationState,
) -> Result<bool, String> {
    let parsed: OllamaStreamChunk = match serde_json::from_str(line) {
        Ok(parsed) => parsed,
        Err(error) => {
            println!("Ollama stream parse ignored: {error} | line={line:?}");
            return Ok(false);
        }
    };

    if !runtime.is_active(request_id) {
        return Ok(false);
    }

    let Some(message) = parsed.message else {
        return Ok(false);
    };

    if message.content.is_empty() {
        return Ok(false);
    }

    full_text.push_str(&message.content);
    let display_chunk = presentation.display_chunk(&message.content);
    if display_chunk.trim().is_empty() {
        return Ok(false);
    }

    if let Some(snapshot) = runtime.metrics.mark_first_llm_chunk(request_id) {
        emit_metrics_update(window, &snapshot);
    }
    if let Some(snapshot) = runtime
        .voice_metrics
        .mark_first_llm_chunk_for_request(request_id)
    {
        emit_voice_metrics_update(window, &snapshot);
    }

    window
        .emit(
            "assistant-stream-chunk",
            StreamChunkEvent {
                request_id: request_id.to_string(),
                chunk: display_chunk.clone(),
            },
        )
        .map_err(|error| format!("assistant-stream-chunk emit failed: {error}"))?;

    if !runtime.is_active(&request_id) {
        runtime.conversation_history.discard_turn(&request_id);
    }

    Ok(true)
}

fn spawn_tts_segment(
    window: WebviewWindow,
    runtime: AssistantRuntime,
    request_id: String,
    segment: SpeechSegment,
) {
    if !runtime.is_active(&request_id) {
        return;
    }
    if !runtime.should_synthesize_segment(&request_id, &segment.text) {
        return;
    }

    if let Some(snapshot) = runtime.metrics.mark_first_segment_queued(&request_id) {
        emit_metrics_update(&window, &snapshot);
    }
    if let Some(snapshot) = runtime
        .voice_metrics
        .mark_first_segment_queued_for_request(&request_id)
    {
        emit_voice_metrics_update(&window, &snapshot);
    }

    let queued_event = SpeechSegmentQueuedEvent {
        request_id: request_id.clone(),
        segment_id: segment.segment_id.clone(),
        sequence: segment.sequence,
        text: segment.text.clone(),
    };

    if let Err(error) = window.emit("assistant-speech-segment-queued", queued_event) {
        println!("assistant-speech-segment-queued emit failed: {error}");
    }

    tauri::async_runtime::spawn(async move {
        if !runtime.is_active(&request_id) {
            return;
        }

        let result = runtime
            .tts_client
            .synthesize(
                request_id.clone(),
                segment.segment_id.clone(),
                segment.sequence,
                segment.text.clone(),
            )
            .await;

        match result {
            Ok(event) => {
                if runtime.is_active(&request_id) {
                    runtime
                        .audio_files
                        .register(&request_id, PathBuf::from(&event.output_path));
                    if let Some(snapshot) = runtime.metrics.mark_first_audio_ready(&request_id) {
                        emit_metrics_update(&window, &snapshot);
                    }
                    if let Some(snapshot) = runtime
                        .voice_metrics
                        .mark_first_audio_ready_for_request(&request_id)
                    {
                        emit_voice_metrics_update(&window, &snapshot);
                    }

                    if let Err(error) = window.emit("assistant-audio-segment-ready", event) {
                        println!("assistant-audio-segment-ready emit failed: {error}");
                    }
                } else {
                    runtime
                        .audio_files
                        .cleanup_played_file(&request_id, PathBuf::from(event.output_path));
                }
            }
            Err(error) if error.is_cancelled() => {}
            Err(error) => {
                if runtime.is_active(&request_id) {
                    let failed = AudioSegmentFailedEvent {
                        request_id: request_id.clone(),
                        segment_id: segment.segment_id,
                        sequence: segment.sequence,
                        message: error.to_string(),
                    };

                    if let Err(emit_error) = window.emit("assistant-audio-segment-failed", failed) {
                        println!("assistant-audio-segment-failed emit failed: {emit_error}");
                    }

                    emit_error(&window, &request_id, "tts", error.to_string());
                }
            }
        }
    });
}

fn tts_segment_fingerprint(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|ch: char| matches!(ch, '.' | ',' | ';' | ':' | '!' | '?'))
        .to_lowercase()
}

fn emit_request_started(
    window: &WebviewWindow,
    request_id: &str,
    model: &str,
    source: &str,
    user_message: Option<String>,
) -> Result<(), String> {
    window
        .emit(
            "assistant-request-started",
            AssistantRequestStartedEvent {
                request_id: request_id.to_string(),
                model: model.to_string(),
                source: source.to_string(),
                user_message,
            },
        )
        .map_err(|error| format!("assistant-request-started emit failed: {error}"))?;

    window
        .emit("assistant-model", model)
        .map_err(|error| format!("assistant-model emit failed: {error}"))?;

    Ok(())
}

fn emit_error(window: &WebviewWindow, request_id: &str, stage: &str, message: String) {
    let event = AssistantErrorEvent {
        request_id: request_id.to_string(),
        stage: stage.to_string(),
        message,
    };

    if let Err(error) = window.emit("assistant-error", event) {
        println!("assistant-error emit failed: {error}");
    }
}

fn emit_metrics_update(window: &WebviewWindow, snapshot: &RequestMetricsSnapshot) {
    if let Err(error) = window.emit("assistant-metrics-updated", snapshot) {
        println!("assistant-metrics-updated emit failed: {error}");
    }
}

fn emit_voice_session_state(window: &WebviewWindow, snapshot: &VoiceSessionSnapshot) {
    let event = VoiceSessionStateEvent {
        session_id: snapshot.session_id.clone(),
        turn_id: snapshot.turn_id.clone(),
        state: snapshot.state.as_str().to_string(),
        mode: snapshot.mode.as_str().to_string(),
        reason: snapshot.reason.clone(),
        conversation_expires_in_ms: snapshot.conversation_expires_in_ms,
        vad: snapshot.vad,
    };

    if let Err(error) = window.emit("voice-session-state-changed", event) {
        println!("voice-session-state-changed emit failed: {error}");
    }
}

fn emit_voice_metrics_update(window: &WebviewWindow, snapshot: &VoiceTurnMetricsSnapshot) {
    if let Err(error) = window.emit("voice-turn-metrics-updated", snapshot) {
        println!("voice-turn-metrics-updated emit failed: {error}");
    }
}

fn emit_voice_session_transcript(
    window: &WebviewWindow,
    session_id: String,
    turn_id: String,
    text: String,
    accepted: bool,
    reason: String,
    action: &str,
    response_text: Option<String>,
) {
    let event = VoiceSessionTranscriptEvent {
        session_id,
        turn_id,
        text,
        accepted,
        reason,
        action: action.to_string(),
        response_text,
    };

    if let Err(error) = window.emit("voice-session-transcript", event) {
        println!("voice-session-transcript emit failed: {error}");
    }
}

fn log_metrics_completed(snapshot: &RequestMetricsSnapshot) {
    println!(
        "{}",
        serde_json::json!({
            "type": "assistant_request_metrics",
            "event": "completed",
            "metrics": snapshot,
        })
    );
}

#[tauri::command]
fn cancel_active_response(state: State<'_, AssistantRuntime>) -> Result<(), String> {
    state.cancel_active_request();
    Ok(())
}

#[tauri::command]
fn notify_audio_playback_started(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    payload: AudioPlaybackEvent,
) -> Result<(), String> {
    if !state.is_active(&payload.request_id) {
        return Ok(());
    }

    if let Some(snapshot) = state.metrics.mark_first_audio_play(&payload.request_id) {
        emit_metrics_update(&window, &snapshot);
    }
    if let Some(snapshot) = state
        .voice_metrics
        .mark_first_audio_play_for_request(&payload.request_id)
    {
        emit_voice_metrics_update(&window, &snapshot);
    }
    let voice_snapshot = state.voice_session.mark_speaking();
    emit_voice_session_state(&window, &voice_snapshot);

    Ok(())
}

#[tauri::command]
fn notify_audio_playback_completed(
    state: State<'_, AssistantRuntime>,
    payload: AudioPlaybackEvent,
) -> Result<(), String> {
    if !state.is_active(&payload.request_id) {
        return Ok(());
    }

    state
        .audio_files
        .cleanup_played_file(&payload.request_id, PathBuf::from(payload.output_path));
    Ok(())
}

#[tauri::command]
fn notify_audio_session_completed(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    payload: AudioSessionCompletedRequest,
) -> Result<(), String> {
    if !state.is_active(&payload.request_id) {
        return Ok(());
    }

    if !payload.had_failures {
        if let Some(snapshot) = state.metrics.mark_audio_completed(&payload.request_id) {
            emit_metrics_update(&window, &snapshot);
            log_metrics_completed(&snapshot);
        }
    } else {
        println!(
            "{}",
            serde_json::json!({
                "type": "assistant_request_metrics",
                "event": "audio_session_finished_with_tts_failures",
                "request_id": payload.request_id,
            })
        );
    }

    state.audio_files.cleanup_request(&payload.request_id);
    state.finish_request(&payload.request_id);
    window
        .emit(
            "assistant-request-settled",
            AssistantRequestSettledEvent {
                request_id: payload.request_id.clone(),
                had_tts_failures: payload.had_failures,
            },
        )
        .map_err(|error| format!("assistant-request-settled emit failed: {error}"))?;
    window
        .emit("assistant-status", "idle")
        .map_err(|error| format!("assistant-status idle emit failed: {error}"))?;
    let voice_snapshot = state.voice_session.mark_assistant_idle();
    emit_voice_session_state(&window, &voice_snapshot);
    Ok(())
}

#[tauri::command]
fn get_recent_request_metrics(
    state: State<'_, AssistantRuntime>,
) -> Result<Vec<RequestMetricsSnapshot>, String> {
    Ok(state.metrics.get_recent())
}

#[tauri::command]
fn get_recent_voice_turn_metrics(
    state: State<'_, AssistantRuntime>,
) -> Result<Vec<VoiceTurnMetricsSnapshot>, String> {
    Ok(state.voice_metrics.get_recent())
}

#[tauri::command]
fn start_voice_session(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<VoiceSessionStartResponse, String> {
    let snapshot = state.voice_session.start();
    emit_voice_session_state(&window, &snapshot);

    let Some(session_id) = snapshot.session_id else {
        return Err("voice session did not produce a session id".to_string());
    };

    Ok(VoiceSessionStartResponse { session_id })
}

#[tauri::command]
fn stop_voice_session(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<(), String> {
    let snapshot = state.voice_session.stop();
    state.cancel_voice_request();
    emit_voice_session_state(&window, &snapshot);
    window
        .emit("assistant-status", "idle")
        .map_err(|error| format!("assistant-status idle emit failed: {error}"))
}

#[tauri::command]
fn report_voice_session_error(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    message: String,
) -> Result<(), String> {
    let snapshot = state.voice_session.stop();
    emit_voice_session_state(&window, &snapshot);
    emit_error(&window, "", "voice_session", message);
    Ok(())
}

#[tauri::command]
fn voice_session_audio_chunk(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    payload: VoiceSessionAudioChunk,
) -> Result<(), String> {
    let runtime = state.inner().clone();
    match runtime.voice_session.process_audio_chunk(
        &payload.session_id,
        payload.sample_rate,
        &payload.samples,
    ) {
        VoiceSessionAction::None => {}
        VoiceSessionAction::StateChanged(snapshot) => {
            emit_voice_session_state(&window, &snapshot);
            if snapshot.reason == "speech_started" {
                if let (Some(session_id), Some(turn_id)) =
                    (snapshot.session_id.as_deref(), snapshot.turn_id.as_deref())
                {
                    let metrics = runtime.voice_metrics.start_utterance(
                        session_id,
                        turn_id,
                        snapshot.vad.backend,
                    );
                    emit_voice_metrics_update(&window, &metrics);
                }
            }
            if snapshot.state.as_str() == "listening" {
                let _ = window.emit("assistant-status", "listening");
            }
        }
        VoiceSessionAction::BargeIn(snapshot) => {
            if let (Some(session_id), Some(turn_id)) =
                (snapshot.session_id.as_deref(), snapshot.turn_id.as_deref())
            {
                let metrics = runtime.voice_metrics.start_utterance(
                    session_id,
                    turn_id,
                    snapshot.vad.backend,
                );
                emit_voice_metrics_update(&window, &metrics);
                if let Some(metrics) = runtime
                    .voice_metrics
                    .mark_interruption_detected(session_id, turn_id)
                {
                    emit_voice_metrics_update(&window, &metrics);
                }
            }
            let request_id = runtime
                .active_request_id
                .lock()
                .expect("active_request_id mutex poisoned")
                .clone();
            runtime.cancel_active_request();
            runtime.cancel_voice_request();
            if let (Some(session_id), Some(turn_id)) =
                (snapshot.session_id.as_deref(), snapshot.turn_id.as_deref())
            {
                if let Some(metrics) = runtime
                    .voice_metrics
                    .mark_interruption_stop_completed(session_id, turn_id)
                {
                    emit_voice_metrics_update(&window, &metrics);
                }
            }
            emit_voice_session_state(&window, &snapshot);
            let _ = window.emit(
                "assistant-interrupted",
                AssistantInterruptedEvent {
                    request_id,
                    reason: "user_barge_in".to_string(),
                },
            );
            let _ = window.emit("assistant-status", "listening");
        }
        VoiceSessionAction::UtteranceReady(utterance) => {
            emit_voice_session_state(&window, &utterance.snapshot);
            if let Some(metrics) = runtime
                .voice_metrics
                .mark_utterance_ended(&utterance.session_id, &utterance.turn_id)
            {
                emit_voice_metrics_update(&window, &metrics);
            }
            let task_window = window.clone();
            tauri::async_runtime::spawn(async move {
                if let Some(metrics) = runtime
                    .voice_metrics
                    .mark_stt_started(&utterance.session_id, &utterance.turn_id)
                {
                    emit_voice_metrics_update(&task_window, &metrics);
                }
                let transcription = runtime.stt_client.transcribe(&utterance.path).await;
                cleanup_temp_recording(&utterance.path);

                let transcript = match transcription {
                    Ok(text) => {
                        if let Some(metrics) = runtime
                            .voice_metrics
                            .mark_stt_completed(&utterance.session_id, &utterance.turn_id)
                        {
                            emit_voice_metrics_update(&task_window, &metrics);
                        }
                        text
                    }
                    Err(error) if error.is_cancelled() => return,
                    Err(error) => {
                        if let Some(metrics) = runtime
                            .voice_metrics
                            .mark_stt_completed(&utterance.session_id, &utterance.turn_id)
                        {
                            emit_voice_metrics_update(&task_window, &metrics);
                        }
                        emit_error(
                            &task_window,
                            &utterance.session_id,
                            "stt",
                            error.to_string(),
                        );
                        let snapshot = runtime.voice_session.mark_assistant_idle();
                        emit_voice_session_state(&task_window, &snapshot);
                        return;
                    }
                };

                match runtime.voice_session.decide_transcript(
                    &utterance.session_id,
                    &utterance.turn_id,
                    &transcript,
                ) {
                    TranscriptDecision::Ignore {
                        session_id,
                        turn_id,
                        text,
                        reason,
                        snapshot,
                    } => {
                        let metrics_session_id = session_id.clone();
                        let metrics_turn_id = turn_id.clone();
                        if let Some(metrics) = runtime.voice_metrics.mark_decision(
                            &session_id,
                            &turn_id,
                            "ignored",
                            &reason,
                            text.chars().count(),
                            false,
                            false,
                            reason == "wake_word_required",
                        ) {
                            emit_voice_metrics_update(&task_window, &metrics);
                        }
                        emit_voice_session_transcript(
                            &task_window,
                            session_id,
                            turn_id,
                            text,
                            false,
                            reason,
                            "ignored",
                            None,
                        );
                        emit_voice_session_state(&task_window, &snapshot);
                        let _ = task_window.emit("assistant-status", "idle");
                        if let Some(metrics) = runtime
                            .voice_metrics
                            .complete_turn(&metrics_session_id, &metrics_turn_id)
                        {
                            emit_voice_metrics_update(&task_window, &metrics);
                        }
                    }
                    TranscriptDecision::Arm {
                        session_id,
                        turn_id,
                        text,
                        reason,
                        snapshot,
                    } => {
                        let metrics_session_id = session_id.clone();
                        let metrics_turn_id = turn_id.clone();
                        if let Some(metrics) = runtime.voice_metrics.mark_decision(
                            &session_id,
                            &turn_id,
                            "armed",
                            &reason,
                            text.chars().count(),
                            true,
                            true,
                            false,
                        ) {
                            emit_voice_metrics_update(&task_window, &metrics);
                        }
                        emit_voice_session_transcript(
                            &task_window,
                            session_id,
                            turn_id,
                            text,
                            true,
                            reason,
                            "armed",
                            None,
                        );
                        emit_voice_session_state(&task_window, &snapshot);
                        let _ = task_window.emit("assistant-status", "listening");
                        if let Some(metrics) = runtime
                            .voice_metrics
                            .complete_turn(&metrics_session_id, &metrics_turn_id)
                        {
                            emit_voice_metrics_update(&task_window, &metrics);
                        }
                    }
                    TranscriptDecision::Respond {
                        session_id,
                        turn_id,
                        text,
                        response_text,
                        reason,
                        snapshot,
                    } => {
                        let wake_detected = reason == "wake_word_detected";
                        if let Some(metrics) = runtime.voice_metrics.mark_decision(
                            &session_id,
                            &turn_id,
                            "responding",
                            &reason,
                            text.chars().count(),
                            wake_detected,
                            true,
                            false,
                        ) {
                            emit_voice_metrics_update(&task_window, &metrics);
                        }
                        emit_voice_session_transcript(
                            &task_window,
                            session_id,
                            turn_id,
                            text.clone(),
                            true,
                            reason,
                            "responding",
                            Some(response_text.clone()),
                        );
                        emit_voice_session_state(&task_window, &snapshot);
                        match start_assistant_response(
                            task_window.clone(),
                            runtime.clone(),
                            response_text,
                            Some(text),
                            "voice_session",
                        )
                        .await
                        {
                            Ok(started) => {
                                if let Some(metrics) = runtime.voice_metrics.mark_response_started(
                                    &utterance.session_id,
                                    &utterance.turn_id,
                                    &started.request_id,
                                ) {
                                    emit_voice_metrics_update(&task_window, &metrics);
                                }
                                if let Some(metrics) = runtime
                                    .voice_metrics
                                    .complete_turn(&utterance.session_id, &utterance.turn_id)
                                {
                                    emit_voice_metrics_update(&task_window, &metrics);
                                }
                            }
                            Err(error) => {
                                emit_error(&task_window, &utterance.session_id, "ollama", error);
                            }
                        }
                    }
                }
            });
        }
    }
    Ok(())
}

#[tauri::command]
async fn transcribe_voice_input(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    payload: VoiceTranscriptionRequest,
) -> Result<VoiceTranscriptionResponse, String> {
    if payload.audio_bytes.is_empty() {
        return Err("Voice recording is empty".to_string());
    }

    let request_id = Uuid::new_v4().to_string();
    let runtime = state.inner().clone();
    runtime.begin_voice_request(request_id.clone());

    window
        .emit("assistant-status", "listening")
        .map_err(|error| format!("assistant-status listening emit failed: {error}"))?;
    window
        .emit(
            "voice-transcription-started",
            VoiceTranscriptionStartedEvent {
                request_id: request_id.clone(),
            },
        )
        .map_err(|error| format!("voice-transcription-started emit failed: {error}"))?;

    let audio_path = write_voice_recording(&request_id, &payload)?;
    let transcription = runtime.stt_client.transcribe(&audio_path).await;
    cleanup_temp_recording(&audio_path);

    match transcription {
        Ok(text) if runtime.is_voice_active(&request_id) => {
            runtime.finish_voice_request(&request_id);

            let event = VoiceTranscriptionFinishedEvent {
                request_id: request_id.clone(),
                text: text.clone(),
                auto_submit: payload.auto_submit,
            };
            window
                .emit("voice-transcription-finished", &event)
                .map_err(|error| format!("voice-transcription-finished emit failed: {error}"))?;
            window
                .emit("assistant-status", "idle")
                .map_err(|error| format!("assistant-status idle emit failed: {error}"))?;

            Ok(VoiceTranscriptionResponse {
                request_id,
                text,
                auto_submit: payload.auto_submit,
            })
        }
        Ok(_) => Err("Voice transcription was cancelled".to_string()),
        Err(error) if error.is_cancelled() => Err(error.to_string()),
        Err(error) => {
            runtime.finish_voice_request(&request_id);
            emit_error(&window, &request_id, "stt", error.to_string());
            let _ = window.emit("assistant-status", "idle");
            Err(error.to_string())
        }
    }
}

#[tauri::command]
fn cancel_voice_input(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<(), String> {
    state.cancel_voice_request();
    window
        .emit("assistant-status", "idle")
        .map_err(|error| format!("assistant-status idle emit failed: {error}"))
}

fn write_voice_recording(
    request_id: &str,
    payload: &VoiceTranscriptionRequest,
) -> Result<PathBuf, String> {
    let root = project_root()?;
    let recordings_dir = root.join("python_services").join("stt").join("recordings");
    fs::create_dir_all(&recordings_dir)
        .map_err(|error| format!("create voice recording dir failed: {error}"))?;

    let extension = audio_extension_for_mime_type(&payload.mime_type);
    let audio_path = recordings_dir.join(format!("stt_{request_id}.{extension}"));
    fs::write(&audio_path, &payload.audio_bytes)
        .map_err(|error| format!("write voice recording failed: {error}"))?;

    Ok(audio_path)
}

fn cleanup_temp_recording(path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => eprintln!(
            "{}",
            serde_json::json!({
                "type": "stt_recording_cleanup",
                "event": "remove_failed",
                "path": path.display().to_string(),
                "error": error.to_string(),
            })
        ),
    }
}

fn audio_extension_for_mime_type(mime_type: &str) -> &'static str {
    let normalized = mime_type.to_ascii_lowercase();
    if normalized.contains("wav") {
        "wav"
    } else if normalized.contains("mp4")
        || normalized.contains("mpeg")
        || normalized.contains("aac")
    {
        "m4a"
    } else if normalized.contains("ogg") {
        "ogg"
    } else {
        "webm"
    }
}

#[tauri::command]
fn list_desktop_tools(state: State<'_, AssistantRuntime>) -> Result<Vec<ToolDescriptor>, String> {
    Ok(state.desktop_agent.list_tools())
}

#[tauri::command]
fn get_desktop_policy_snapshot(
    state: State<'_, AssistantRuntime>,
) -> Result<DesktopPolicySnapshot, String> {
    Ok(state.desktop_agent.policy_snapshot())
}

#[tauri::command]
fn get_pending_desktop_approvals(
    state: State<'_, AssistantRuntime>,
) -> Result<Vec<PendingApproval>, String> {
    Ok(state.desktop_agent.pending_approvals())
}

#[tauri::command]
fn get_recent_desktop_audit_events(
    state: State<'_, AssistantRuntime>,
    limit: Option<usize>,
) -> Result<Vec<DesktopAuditEvent>, String> {
    Ok(state.desktop_agent.recent_audit_events(limit.unwrap_or(50)))
}

#[tauri::command]
fn execute_desktop_action(
    state: State<'_, AssistantRuntime>,
    payload: DesktopActionRequest,
) -> Result<DesktopActionResponse, String> {
    let request_id = Uuid::new_v4().to_string();
    state.desktop_agent.submit_action(request_id, payload)
}

#[tauri::command]
fn approve_desktop_action(
    state: State<'_, AssistantRuntime>,
    payload: ApprovalDecisionRequest,
) -> Result<DesktopActionResponse, String> {
    state
        .desktop_agent
        .approve_pending(&payload.action_id, payload.note)
}

#[tauri::command]
fn reject_desktop_action(
    state: State<'_, AssistantRuntime>,
    payload: ApprovalDecisionRequest,
) -> Result<(), String> {
    state
        .desktop_agent
        .reject_pending(&payload.action_id, payload.note)
}

#[tauri::command]
async fn get_capability_manifest(
    state: State<'_, AssistantRuntime>,
) -> Result<CapabilityManifest, String> {
    Ok(state.desktop_agent.capability_manifest().await)
}

#[tauri::command]
fn get_screen_observation_status(
    state: State<'_, AssistantRuntime>,
) -> Result<ScreenObservationStatus, String> {
    Ok(state.desktop_agent.screen_status())
}

#[tauri::command]
fn set_screen_observation_enabled(
    state: State<'_, AssistantRuntime>,
    enabled: bool,
) -> Result<ScreenObservationStatus, String> {
    Ok(state.desktop_agent.set_screen_observation_enabled(enabled))
}

#[tauri::command]
fn capture_screen_snapshot(
    state: State<'_, AssistantRuntime>,
) -> Result<ScreenCaptureResult, String> {
    state.desktop_agent.capture_screen_snapshot()
}

#[tauri::command]
async fn analyze_screen_context(
    state: State<'_, AssistantRuntime>,
    payload: ScreenAnalysisRequest,
) -> Result<ScreenAnalysisResult, String> {
    state.desktop_agent.analyze_screen(payload).await
}

#[tauri::command]
fn get_recent_goal_loop(state: State<'_, AssistantRuntime>) -> Result<Option<GoalLoopRun>, String> {
    Ok(state.desktop_agent.recent_goal_loop())
}

#[tauri::command]
fn minimize_window(window: WebviewWindow) -> Result<(), String> {
    window.minimize().map_err(|error| error.to_string())
}

#[tauri::command]
fn toggle_always_on_top(window: WebviewWindow) -> Result<bool, String> {
    let is_always_on_top = window
        .is_always_on_top()
        .map_err(|error| error.to_string())?;
    let next_value = !is_always_on_top;
    window
        .set_always_on_top(next_value)
        .map_err(|error| error.to_string())?;
    Ok(next_value)
}

#[tauri::command]
fn close_window(window: WebviewWindow, state: State<'_, AssistantRuntime>) -> Result<(), String> {
    state.cancel_active_request();
    state.cancel_voice_request();
    window.close().map_err(|error| error.to_string())
}

#[tauri::command]
fn start_window_drag(window: WebviewWindow) -> Result<(), String> {
    window.start_dragging().map_err(|error| error.to_string())
}

#[tauri::command]
fn set_compact_mode(window: WebviewWindow) -> Result<(), String> {
    window
        .set_size(tauri::Size::Logical(tauri::LogicalSize {
            width: 320.0,
            height: 110.0,
        }))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_expanded_mode(window: WebviewWindow) -> Result<(), String> {
    window
        .set_size(tauri::Size::Logical(tauri::LogicalSize {
            width: 420.0,
            height: 720.0,
        }))
        .map_err(|error| error.to_string())
}

// ========================
// Meeting Engine Commands
// ========================

fn governed_meeting_command<F>(
    runtime: &AssistantRuntime,
    tool_name: &str,
    params: serde_json::Value,
    operation: F,
) -> Result<serde_json::Value, String>
where
    F: FnOnce() -> Result<serde_json::Value, String>,
{
    runtime.desktop_agent.execute_governed_direct_action(
        Uuid::new_v4().to_string(),
        tool_name,
        params,
        false,
        operation,
    )
}

fn meeting_capture_preflight_params(platform: &str, config: &MeetingConfig) -> serde_json::Value {
    serde_json::json!({
        "platform": platform,
        "session_mode": config.session_mode,
        "capture_backend": config.capture_backend,
        "capture_options": config.capture_options,
        "metadata_only": true,
        "raw_audio_included": false,
        "transcript_text_included": false,
    })
}

fn meeting_segment_transcription_preflight_params(
    platform: &str,
    config: &MeetingConfig,
) -> serde_json::Value {
    serde_json::json!({
        "platform": platform,
        "session_mode": config.session_mode,
        "capture_backend": config.capture_backend,
        "capture_options": config.capture_options,
        "transcription_model": config.transcription_model.clone(),
        "metadata_only": true,
        "raw_audio_included": false,
        "transcript_text_included": false,
    })
}

fn meeting_capture_start_confirmation_details(
    platform: &str,
    config: &MeetingConfig,
    tool_name: &str,
) -> serde_json::Value {
    serde_json::json!({
        "method": "meeting_control_center_explicit_start",
        "user_initiated": true,
        "operation": "start_capture",
        "tool_name": tool_name,
        "platform": platform,
        "capture_backend": config.capture_backend,
        "capture_options": config.capture_options,
        "raw_audio": "not_included",
        "transcript_text": "not_included",
        "metadata_only": true,
    })
}

fn confirmed_meeting_capability_permission_check(
    desktop_agent: &DesktopAgentRuntime,
    tool_name: &str,
    params: serde_json::Value,
    confirmation_details: serde_json::Value,
) -> Result<(), String> {
    desktop_agent
        .execute_confirmed_governed_direct_action(
            Uuid::new_v4().to_string(),
            tool_name,
            params,
            confirmation_details,
            || {
                Ok(serde_json::json!({
                    "permission_checked": true,
                    "capability_available": true,
                    "direct_confirmation": true,
                    "metadata_only": true,
                }))
            },
        )
        .map(|_| ())
}

fn meeting_capture_preflight_tool_names(config: &MeetingConfig) -> Vec<&'static str> {
    let mut tool_names = vec!["meeting.audio.capture"];
    if config.capture_options.system_audio {
        tool_names.push("meeting.audio.capture.system");
    }
    if config.capture_options.microphone {
        tool_names.push("meeting.audio.capture.microphone");
    }
    tool_names
}

fn meeting_segment_transcription_requested(config: &MeetingConfig) -> bool {
    config.live_transcription_enabled || config.capture_options.segment_transcription
}

fn confirmed_meeting_start_preflight_checks(
    desktop_agent: &DesktopAgentRuntime,
    platform: &str,
    config: &MeetingConfig,
) -> Result<(), String> {
    for tool_name in meeting_capture_preflight_tool_names(config) {
        confirmed_meeting_capability_permission_check(
            desktop_agent,
            tool_name,
            meeting_capture_preflight_params(platform, config),
            meeting_capture_start_confirmation_details(platform, config, tool_name),
        )?;
    }

    if meeting_segment_transcription_requested(config) {
        confirmed_meeting_capability_permission_check(
            desktop_agent,
            "meeting.transcription.segment",
            meeting_segment_transcription_preflight_params(platform, config),
            meeting_capture_start_confirmation_details(
                platform,
                config,
                "meeting.transcription.segment",
            ),
        )?;
    }

    Ok(())
}

fn confirmed_governed_meeting_command<F>(
    runtime: &AssistantRuntime,
    tool_name: &str,
    params: serde_json::Value,
    confirmation_details: serde_json::Value,
    operation: F,
) -> Result<serde_json::Value, String>
where
    F: FnOnce() -> Result<serde_json::Value, String>,
{
    runtime
        .desktop_agent
        .execute_confirmed_governed_direct_action(
            Uuid::new_v4().to_string(),
            tool_name,
            params,
            confirmation_details,
            operation,
        )
}

fn safe_audio_extension_hint(audio_path: &str) -> String {
    let segment = audio_path
        .trim()
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        .trim();
    let Some((_, extension)) = segment.rsplit_once('.') else {
        return "unknown".to_string();
    };
    let extension = extension.trim().to_ascii_lowercase();
    if extension.is_empty()
        || extension.len() > 16
        || !extension
            .chars()
            .all(|character| character.is_ascii_alphanumeric())
    {
        return "unknown".to_string();
    }
    extension
}

fn meeting_file_transcription_preflight_params(
    request: &MeetingAudioFileTranscriptionRequest,
) -> serde_json::Value {
    serde_json::json!({
        "session_id_provided": request
            .session_id
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty()),
        "path_provided": !request.audio_path.trim().is_empty(),
        "extension_hint": safe_audio_extension_hint(&request.audio_path),
        "audio_path_redacted": true,
        "speaker_provided": request
            .speaker
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty()),
        "cleanup_after_transcription": request.cleanup_after_transcription,
        "metadata_only": true,
    })
}

fn meeting_value<T: Serialize>(value: T) -> Result<serde_json::Value, String> {
    serde_json::to_value(value)
        .map_err(|error| format!("meeting result serialization failed: {error}"))
}

fn meeting_from_value<T: DeserializeOwned>(value: serde_json::Value) -> Result<T, String> {
    serde_json::from_value(value)
        .map_err(|error| format!("meeting result deserialization failed: {error}"))
}

fn emit_meeting_update_events(window: &WebviewWindow, events: &[&str]) {
    let payload = serde_json::json!({
        "source": "meeting_runtime",
        "metadata_only": true,
    });
    if let Some(main_window) = window.app_handle().get_webview_window("main") {
        for event in events {
            let _ = main_window.emit(event, payload.clone());
        }
    } else {
        for event in events {
            let _ = window.emit(event, payload.clone());
        }
    }
}

#[tauri::command]
fn get_meeting_consent_state(state: State<'_, AssistantRuntime>) -> Result<ConsentState, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.consent.read",
        serde_json::json!({}),
        move || meeting_value(meeting.consent_state().map_err(|error| error.to_string())?),
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn grant_meeting_consent(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    app_name: String,
) -> Result<ConsentState, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.consent.grant",
        serde_json::json!({ "app_name": app_name.clone() }),
        move || {
            meeting_value(
                meeting
                    .grant_consent(&app_name)
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    let consent = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &["meeting-session-updated", "meeting-diagnostics-updated"],
    );
    Ok(consent)
}

#[tauri::command]
fn revoke_meeting_consent(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    app_name: String,
) -> Result<ConsentState, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.consent.revoke",
        serde_json::json!({ "app_name": app_name.clone() }),
        move || {
            meeting_value(
                meeting
                    .revoke_consent(&app_name)
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    let consent = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &["meeting-session-updated", "meeting-diagnostics-updated"],
    );
    Ok(consent)
}

#[tauri::command]
fn start_meeting_session(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    platform: String,
    config: MeetingConfig,
) -> Result<MeetingSession, String> {
    let meeting = state.meeting_runtime.clone();
    let desktop_agent = state.desktop_agent.clone();
    let params = serde_json::json!({
        "platform": platform.clone(),
        "capture_backend": config.capture_backend,
        "transcription_model": config.transcription_model.clone(),
        "session_mode": config.session_mode,
        "segment_transcription_enabled": config.capture_options.segment_transcription || config.live_transcription_enabled,
        "streaming_stt": "unsupported",
        "capture_options": config.capture_options,
    });
    let value =
        governed_meeting_command(state.inner(), "meeting.session.start", params, move || {
            if config.session_mode == MeetingSessionMode::RealCapture {
                confirmed_meeting_start_preflight_checks(&desktop_agent, &platform, &config)?;
            }
            meeting_value(
                meeting
                    .start_session(platform, config)
                    .map_err(|error| error.to_string())?,
            )
        })?;
    let session = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-session-updated",
            "meeting-diagnostics-updated",
            "meeting-artifacts-updated",
        ],
    );
    Ok(session)
}

#[tauri::command]
fn get_active_meeting_session(
    state: State<'_, AssistantRuntime>,
) -> Result<Option<MeetingSession>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.session.read",
        serde_json::json!({ "read": "active_session" }),
        move || {
            meeting_value(
                meeting
                    .get_active_session()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn get_active_meeting_state(
    state: State<'_, AssistantRuntime>,
) -> Result<MeetingSessionState, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.session.read",
        serde_json::json!({ "read": "active_state", "data_category": "meeting_state" }),
        move || {
            meeting_value(
                meeting
                    .get_active_state()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn get_last_completed_meeting_state(
    state: State<'_, AssistantRuntime>,
) -> Result<Option<MeetingSessionState>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.session.read",
        serde_json::json!({ "read": "last_completed_state", "data_category": "meeting_state" }),
        move || {
            meeting_value(
                meeting
                    .get_last_completed_state()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn list_meeting_transcript(
    state: State<'_, AssistantRuntime>,
) -> Result<Vec<TranscriptEntry>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.transcript.list",
        serde_json::json!({ "read": "transcript", "data_category": "meeting_transcript" }),
        move || {
            meeting_value(
                meeting
                    .list_transcript()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn read_meeting_notes(state: State<'_, AssistantRuntime>) -> Result<Vec<NoteEntry>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.notes.read",
        serde_json::json!({ "read": "notes", "data_category": "meeting_notes" }),
        move || meeting_value(meeting.read_notes().map_err(|error| error.to_string())?),
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn read_meeting_summary(state: State<'_, AssistantRuntime>) -> Result<Vec<SummaryEntry>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.notes.read",
        serde_json::json!({ "read": "summary", "data_category": "meeting_summary" }),
        move || meeting_value(meeting.read_summary().map_err(|error| error.to_string())?),
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn read_meeting_action_items(
    state: State<'_, AssistantRuntime>,
) -> Result<Vec<ActionItem>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.action_items.read",
        serde_json::json!({ "read": "action_items", "data_category": "meeting_action_items" }),
        move || {
            meeting_value(
                meeting
                    .read_action_items()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn read_meeting_decisions(
    state: State<'_, AssistantRuntime>,
) -> Result<Vec<DecisionLogEntry>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.decisions.read",
        serde_json::json!({ "read": "decisions", "data_category": "meeting_decisions" }),
        move || {
            meeting_value(
                meeting
                    .read_decisions()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn read_meeting_diagnostics(
    state: State<'_, AssistantRuntime>,
) -> Result<Vec<MeetingDiagnostic>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.diagnostics.read",
        serde_json::json!({ "read": "diagnostics", "data_category": "meeting_diagnostics" }),
        move || {
            meeting_value(
                meeting
                    .read_diagnostics()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn generate_meeting_intelligence(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    options: MeetingIntelligenceGenerationOptions,
) -> Result<MeetingIntelligenceResult, String> {
    let meeting = state.meeting_runtime.clone();
    let params = serde_json::json!({
        "artifact_types_requested": [
            "summary",
            "decisions",
            "action_items",
            "open_questions",
            "risks",
            "technical_recap",
            "follow_up_draft",
            "timeline"
        ],
        "use_local_llm_requested": options.use_local_llm,
        "max_transcript_segments": options.max_transcript_segments,
        "metadata_only": true,
        "transcript_text_included": false,
        "generated_text_included": false,
        "audit_redacted": true,
    });
    let value = governed_meeting_command(
        state.inner(),
        "meeting.intelligence.generate",
        params,
        move || {
            meeting_value(
                meeting
                    .generate_intelligence(options)
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    let result = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
            "meeting-session-updated",
        ],
    );
    Ok(result)
}

#[tauri::command]
fn read_meeting_intelligence(
    state: State<'_, AssistantRuntime>,
) -> Result<Option<MeetingIntelligenceResult>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.intelligence.read",
        serde_json::json!({
            "read": "meeting_intelligence",
            "data_category": "meeting_intelligence",
            "metadata_only": true,
            "transcript_text_included": false,
        }),
        move || {
            meeting_value(
                meeting
                    .read_intelligence()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn clear_meeting_intelligence(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<(), String> {
    let meeting = state.meeting_runtime.clone();
    governed_meeting_command(
        state.inner(),
        "meeting.intelligence.clear",
        serde_json::json!({
            "metadata_only": true,
            "transcript_text_included": false,
            "generated_text_included": false,
        }),
        move || {
            meeting
                .clear_intelligence()
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({ "ok": true, "metadata_only": true }))
        },
    )?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
            "meeting-session-updated",
        ],
    );
    Ok(())
}

#[tauri::command]
fn draft_meeting_followup(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<Option<MeetingFollowUpDraft>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.followup.draft",
        serde_json::json!({
            "metadata_only": true,
            "transcript_text_included": false,
            "generated_text_included": false,
            "send_email": false,
        }),
        move || {
            let existing = meeting
                .read_intelligence()
                .map_err(|error| error.to_string())?;
            let intelligence = match existing {
                Some(result) if result.follow_up_draft.is_some() => result,
                _ => meeting
                    .generate_intelligence(MeetingIntelligenceGenerationOptions::default())
                    .map_err(|error| error.to_string())?,
            };
            meeting_value(intelligence.follow_up_draft)
        },
    )?;
    let draft = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
            "meeting-session-updated",
        ],
    );
    Ok(draft)
}

#[tauri::command]
fn get_meeting_live_capabilities(
    state: State<'_, AssistantRuntime>,
) -> Result<MeetingLiveCapabilitySnapshot, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.session.read",
        serde_json::json!({ "read": "live_capabilities", "data_category": "meeting_capability_metadata" }),
        move || {
            meeting_value(
                meeting
                    .live_capabilities()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn pause_meeting_session(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<(), String> {
    let meeting = state.meeting_runtime.clone();
    governed_meeting_command(
        state.inner(),
        "meeting.session.pause",
        serde_json::json!({}),
        move || {
            meeting.pause_session().map_err(|error| error.to_string())?;
            Ok(serde_json::json!({ "ok": true }))
        },
    )?;
    emit_meeting_update_events(
        &window,
        &["meeting-session-updated", "meeting-diagnostics-updated"],
    );
    Ok(())
}

#[tauri::command]
fn resume_meeting_session(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<(), String> {
    let meeting = state.meeting_runtime.clone();
    governed_meeting_command(
        state.inner(),
        "meeting.session.resume",
        serde_json::json!({}),
        move || {
            meeting
                .resume_session()
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({ "ok": true }))
        },
    )?;
    emit_meeting_update_events(
        &window,
        &["meeting-session-updated", "meeting-diagnostics-updated"],
    );
    Ok(())
}

#[tauri::command]
fn stop_meeting_session(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<ExportedMeeting, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.session.stop",
        serde_json::json!({}),
        move || meeting_value(meeting.stop_session().map_err(|error| error.to_string())?),
    )?;
    let exported = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-session-updated",
            "meeting-transcript-updated",
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
        ],
    );
    Ok(exported)
}

#[tauri::command]
fn add_meeting_transcript(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    entry: TranscriptEntry,
) -> Result<(), String> {
    let meeting = state.meeting_runtime.clone();
    governed_meeting_command(
        state.inner(),
        "meeting.transcript.add",
        serde_json::json!({ "speaker": entry.speaker.clone(), "confidence": entry.confidence }),
        move || {
            meeting
                .add_transcript(entry)
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({ "ok": true }))
        },
    )?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-transcript-updated",
            "meeting-artifacts-updated",
            "meeting-session-updated",
        ],
    );
    Ok(())
}

#[tauri::command]
fn rename_meeting_speaker(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    request: RenameSpeakerRequest,
) -> Result<RenameSpeakerResult, String> {
    let meeting = state.meeting_runtime.clone();
    let params = serde_json::json!({
        "speaker_id": request.speaker_id.clone(),
        "display_name_length": request.display_name.trim().chars().count(),
        "metadata_only": true,
        "transcript_text_included": false,
    });
    let value =
        governed_meeting_command(state.inner(), "meeting.speaker.rename", params, move || {
            meeting_value(
                meeting
                    .rename_speaker(request)
                    .map_err(|error| error.to_string())?,
            )
        })?;
    let result = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-transcript-updated",
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
            "meeting-session-updated",
        ],
    );
    Ok(result)
}

#[tauri::command]
async fn transcribe_meeting_audio_file(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    request: MeetingAudioFileTranscriptionRequest,
) -> Result<MeetingAudioFileTranscriptionResult, String> {
    let meeting = state.meeting_runtime.clone();
    let desktop_agent = state.desktop_agent.clone();
    let params = meeting_file_transcription_preflight_params(&request);
    let value = desktop_agent
        .execute_confirmed_governed_direct_action_async(
            Uuid::new_v4().to_string(),
            "meeting.transcription.file",
            params,
            serde_json::json!({
                "method": "direct_debug_user_action",
                "audio_path_redacted": true,
            }),
            move || async move {
                meeting_value(
                    meeting
                        .transcribe_audio_file(request)
                        .await
                        .map_err(|error| error.to_string())?,
                )
            },
        )
        .await?;
    let result = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-transcript-updated",
            "meeting-artifacts-updated",
            "meeting-session-updated",
        ],
    );
    Ok(result)
}

#[tauri::command]
fn add_meeting_action_item(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    item: ActionItem,
) -> Result<(), String> {
    let meeting = state.meeting_runtime.clone();
    governed_meeting_command(
        state.inner(),
        "meeting.action_item.add",
        serde_json::json!({
            "description_length": item.description.chars().count(),
            "has_assignee": item.assignee.is_some(),
            "has_deadline": item.deadline.is_some(),
            "status": item.status,
        }),
        move || {
            meeting
                .add_action_item(item)
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({ "ok": true }))
        },
    )?;
    emit_meeting_update_events(
        &window,
        &["meeting-artifacts-updated", "meeting-session-updated"],
    );
    Ok(())
}

#[tauri::command]
fn add_meeting_decision(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    entry: DecisionLogEntry,
) -> Result<(), String> {
    let meeting = state.meeting_runtime.clone();
    governed_meeting_command(
        state.inner(),
        "meeting.decision.add",
        serde_json::json!({
            "decision_length": entry.decision.chars().count(),
            "rationale_length": entry.rationale.chars().count(),
            "has_made_by": entry.made_by.is_some(),
        }),
        move || {
            meeting
                .add_decision(entry)
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({ "ok": true }))
        },
    )?;
    emit_meeting_update_events(
        &window,
        &["meeting-artifacts-updated", "meeting-session-updated"],
    );
    Ok(())
}

#[tauri::command]
fn clear_meeting_session(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<(), String> {
    let meeting = state.meeting_runtime.clone();
    governed_meeting_command(
        state.inner(),
        "meeting.session.clear",
        serde_json::json!({ "scope": "runtime_session" }),
        move || {
            meeting
                .clear_runtime_session()
                .map_err(|error| error.to_string())?;
            Ok(serde_json::json!({ "ok": true }))
        },
    )?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-session-updated",
            "meeting-transcript-updated",
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
        ],
    );
    Ok(())
}

#[tauri::command]
fn detect_active_call(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
) -> Result<Option<CallInfo>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.detect",
        serde_json::json!({}),
        move || meeting_value(meeting.detect_active_call()),
    )?;
    let call: Option<CallInfo> = meeting_from_value(value)?;
    if let Some(ref ci) = call {
        if let Some(w) = window.app_handle().get_webview_window("main") {
            if ci.is_active_call {
                let _ = w.emit("meeting-call-detected", ci);
            }
        }
    }
    Ok(call)
}

#[tauri::command]
fn get_available_audio_devices(state: State<'_, AssistantRuntime>) -> Result<Vec<String>, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.audio.devices",
        serde_json::json!({}),
        move || {
            meeting_value(
                meeting
                    .available_audio_devices()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn auto_detect_audio_backend(state: State<'_, AssistantRuntime>) -> Result<CaptureBackend, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.audio.backend",
        serde_json::json!({}),
        move || meeting_value(meeting.auto_detect_audio_backend()),
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn preview_clear_meeting_data(
    state: State<'_, AssistantRuntime>,
) -> Result<MeetingDataClearPreview, String> {
    let meeting = state.meeting_runtime.clone();
    let value = governed_meeting_command(
        state.inner(),
        "meeting.clear_data.preview",
        serde_json::json!({ "scope": "all" }),
        move || {
            meeting_value(
                meeting
                    .preview_clear_all_data()
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    meeting_from_value(value)
}

#[tauri::command]
fn clear_meeting_data(
    window: WebviewWindow,
    state: State<'_, AssistantRuntime>,
    request: ClearMeetingDataRequest,
) -> Result<MeetingDataClearResult, String> {
    let meeting = state.meeting_runtime.clone();
    let scope = request.scope.clone();
    let value = confirmed_governed_meeting_command(
        state.inner(),
        "meeting.clear_data",
        serde_json::json!({
            "scope": scope,
            "confirmation": "typed_phrase_required",
        }),
        serde_json::json!({
            "method": "typed_phrase",
            "phrase_redacted": true,
        }),
        move || {
            meeting_value(
                meeting
                    .clear_all_data(request)
                    .map_err(|error| error.to_string())?,
            )
        },
    )?;
    let result = meeting_from_value(value)?;
    emit_meeting_update_events(
        &window,
        &[
            "meeting-session-updated",
            "meeting-transcript-updated",
            "meeting-artifacts-updated",
            "meeting-diagnostics-updated",
        ],
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_real_capture_config() -> MeetingConfig {
        MeetingConfig {
            platform: "teams".to_string(),
            capture_backend: CaptureBackend::Wasapi,
            transcription_model: "local".to_string(),
            sample_rate: 16_000,
            diarization_enabled: false,
            privacy_mode: "default".to_string(),
            session_mode: MeetingSessionMode::RealCapture,
            live_transcription_enabled: true,
            capture_options: meeting::types::MeetingCaptureOptions {
                system_audio: true,
                microphone: false,
                segment_transcription: true,
            },
        }
    }

    fn test_desktop_agent_runtime(name: &str) -> (DesktopAgentRuntime, PathBuf) {
        let root = std::env::temp_dir().join(format!("{name}_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp root");
        (DesktopAgentRuntime::new(root.clone()), root)
    }

    #[test]
    fn file_transcription_preflight_does_not_touch_filesystem_before_governance() {
        let request = MeetingAudioFileTranscriptionRequest {
            session_id: Some("session-with-sensitive-id".to_string()),
            audio_path: r"C:\Users\Simone\layoff_call.wav".to_string(),
            speaker: Some("speaker".to_string()),
            cleanup_after_transcription: true,
        };

        let params = meeting_file_transcription_preflight_params(&request);
        let serialized = params.to_string();

        assert_eq!(params["session_id_provided"], true);
        assert_eq!(params["path_provided"], true);
        assert_eq!(params["extension_hint"], "wav");
        assert_eq!(params["audio_path_redacted"], true);
        assert_eq!(params["speaker_provided"], true);
        assert_eq!(params["cleanup_after_transcription"], true);
        assert!(!serialized.contains("session-with-sensitive-id"));
        assert!(!serialized.contains("layoff_call"));
        assert!(!serialized.contains("Simone"));
        assert!(!params
            .as_object()
            .expect("params object")
            .contains_key("file_size_bytes"));
    }

    #[test]
    fn file_transcription_preflight_rejects_sensitive_extension_shapes() {
        assert_eq!(safe_audio_extension_hint("meeting.wav"), "wav");
        assert_eq!(safe_audio_extension_hint("meeting."), "unknown");
        assert_eq!(
            safe_audio_extension_hint("meeting.wav?token=secret"),
            "unknown"
        );
        assert_eq!(
            safe_audio_extension_hint("meeting.verylongextensionname"),
            "unknown"
        );
    }

    #[test]
    fn meeting_capture_preflight_tools_follow_selected_sources() {
        let mut config = test_real_capture_config();

        config.capture_options.system_audio = true;
        config.capture_options.microphone = false;
        assert_eq!(
            meeting_capture_preflight_tool_names(&config),
            vec!["meeting.audio.capture", "meeting.audio.capture.system"]
        );

        config.capture_options.system_audio = false;
        config.capture_options.microphone = true;
        assert_eq!(
            meeting_capture_preflight_tool_names(&config),
            vec!["meeting.audio.capture", "meeting.audio.capture.microphone"]
        );

        config.capture_options.system_audio = true;
        config.capture_options.microphone = true;
        assert_eq!(
            meeting_capture_preflight_tool_names(&config),
            vec![
                "meeting.audio.capture",
                "meeting.audio.capture.system",
                "meeting.audio.capture.microphone"
            ]
        );
    }

    #[test]
    fn meeting_segment_transcription_request_matches_live_or_capture_option() {
        let mut config = test_real_capture_config();

        config.live_transcription_enabled = false;
        config.capture_options.segment_transcription = false;
        assert!(!meeting_segment_transcription_requested(&config));

        config.capture_options.segment_transcription = true;
        assert!(meeting_segment_transcription_requested(&config));

        config.capture_options.segment_transcription = false;
        config.live_transcription_enabled = true;
        assert!(meeting_segment_transcription_requested(&config));
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn real_capture_preflight_accepts_direct_confirmation_for_audio_capture() {
        let (runtime, root) = test_desktop_agent_runtime("astra_audio_capture_confirmed");
        let config = test_real_capture_config();

        confirmed_meeting_capability_permission_check(
            &runtime,
            "meeting.audio.capture",
            meeting_capture_preflight_params("teams", &config),
            meeting_capture_start_confirmation_details("teams", &config, "meeting.audio.capture"),
        )
        .expect("confirmed audio capture preflight");

        let events = runtime.recent_audit_events(20);
        assert!(events.iter().any(|event| {
            event.tool_name == "meeting.audio.capture"
                && event.status == "direct_confirmation_accepted"
        }));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn meeting_start_preflight_checks_segment_when_capture_option_enabled_without_live_flag() {
        let (runtime, root) = test_desktop_agent_runtime("astra_segment_option_preflight");
        let mut config = test_real_capture_config();
        config.live_transcription_enabled = false;
        config.capture_options.system_audio = true;
        config.capture_options.microphone = false;
        config.capture_options.segment_transcription = true;

        confirmed_meeting_start_preflight_checks(&runtime, "teams", &config)
            .expect("start preflight checks");

        let events = runtime.recent_audit_events(30);
        assert!(events.iter().any(|event| {
            event.tool_name == "meeting.audio.capture"
                && event.status == "direct_confirmation_accepted"
        }));
        assert!(events.iter().any(|event| {
            event.tool_name == "meeting.audio.capture.system"
                && event.status == "direct_confirmation_accepted"
        }));
        assert!(events.iter().any(|event| {
            event.tool_name == "meeting.transcription.segment"
                && event.status == "direct_confirmation_accepted"
        }));
        assert!(!events
            .iter()
            .any(|event| event.tool_name == "meeting.audio.capture.microphone"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn meeting_start_preflight_checks_both_source_specific_capture_tools_for_dual_capture() {
        let (runtime, root) = test_desktop_agent_runtime("astra_dual_source_preflight");
        let mut config = test_real_capture_config();
        config.live_transcription_enabled = false;
        config.capture_options.system_audio = true;
        config.capture_options.microphone = true;
        config.capture_options.segment_transcription = false;

        confirmed_meeting_start_preflight_checks(&runtime, "teams", &config)
            .expect("start preflight checks");

        let events = runtime.recent_audit_events(30);
        for tool_name in [
            "meeting.audio.capture",
            "meeting.audio.capture.system",
            "meeting.audio.capture.microphone",
        ] {
            assert!(events.iter().any(|event| {
                event.tool_name == tool_name && event.status == "direct_confirmation_accepted"
            }));
        }
        assert!(!events
            .iter()
            .any(|event| event.tool_name == "meeting.transcription.segment"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn real_capture_preflight_accepts_direct_confirmation_for_segment_transcription() {
        let (runtime, root) = test_desktop_agent_runtime("astra_segment_confirmed");
        let config = test_real_capture_config();

        confirmed_meeting_capability_permission_check(
            &runtime,
            "meeting.transcription.segment",
            meeting_segment_transcription_preflight_params("teams", &config),
            meeting_capture_start_confirmation_details(
                "teams",
                &config,
                "meeting.transcription.segment",
            ),
        )
        .expect("confirmed segment transcription preflight");

        let events = runtime.recent_audit_events(20);
        assert!(events.iter().any(|event| {
            event.tool_name == "meeting.transcription.segment"
                && event.status == "direct_confirmation_accepted"
        }));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn approval_required_capture_preflight_does_not_fail_as_unconfirmed() {
        let (runtime, root) =
            test_desktop_agent_runtime("astra_segment_unconfirmed_then_confirmed");
        let config = test_real_capture_config();
        let params = meeting_segment_transcription_preflight_params("teams", &config);

        let unconfirmed = runtime.execute_governed_direct_action(
            "meeting-segment-unconfirmed".into(),
            "meeting.transcription.segment",
            params.clone(),
            false,
            || Ok(serde_json::json!({"permission_checked": true})),
        );
        assert!(unconfirmed
            .expect_err("unconfirmed segment preflight should require approval")
            .contains("Approval required"));

        confirmed_meeting_capability_permission_check(
            &runtime,
            "meeting.transcription.segment",
            params,
            meeting_capture_start_confirmation_details(
                "teams",
                &config,
                "meeting.transcription.segment",
            ),
        )
        .expect("confirmed segment preflight");

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn meeting_capture_confirmation_audit_is_redacted() {
        let (runtime, root) = test_desktop_agent_runtime("astra_capture_confirmation_redacted");
        let config = test_real_capture_config();

        confirmed_meeting_capability_permission_check(
            &runtime,
            "meeting.transcription.segment",
            meeting_segment_transcription_preflight_params("teams", &config),
            meeting_capture_start_confirmation_details(
                "teams",
                &config,
                "meeting.transcription.segment",
            ),
        )
        .expect("confirmed segment preflight");

        let serialized =
            serde_json::to_string(&runtime.recent_audit_events(20)).expect("audit json");
        assert!(serialized.contains("direct_confirmation_accepted"));
        assert!(serialized.contains("metadata_only"));
        assert!(serialized.contains("not_included"));
        assert!(!serialized.contains("sensitive transcript"));
        assert!(!serialized.contains("raw_audio_samples"));
        assert!(!serialized.contains(".wav"));
        assert!(!serialized.contains("C:/"));
        assert!(!serialized.contains("\\Users\\"));

        let _ = std::fs::remove_dir_all(root);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let root = project_root()
                .map_err(|message| std::io::Error::new(std::io::ErrorKind::Other, message))?;
            app.manage(AssistantRuntime::new(root));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_chat_message_stream,
            cancel_active_response,
            notify_audio_playback_started,
            notify_audio_playback_completed,
            notify_audio_session_completed,
            get_recent_request_metrics,
            get_recent_voice_turn_metrics,
            start_voice_session,
            stop_voice_session,
            report_voice_session_error,
            voice_session_audio_chunk,
            transcribe_voice_input,
            cancel_voice_input,
            list_desktop_tools,
            get_desktop_policy_snapshot,
            get_pending_desktop_approvals,
            get_recent_desktop_audit_events,
            execute_desktop_action,
            approve_desktop_action,
            reject_desktop_action,
            get_capability_manifest,
            get_screen_observation_status,
            set_screen_observation_enabled,
            capture_screen_snapshot,
            analyze_screen_context,
            get_recent_goal_loop,
            minimize_window,
            toggle_always_on_top,
            close_window,
            start_window_drag,
            set_compact_mode,
            set_expanded_mode,
            // Meeting engine commands
            get_meeting_consent_state,
            grant_meeting_consent,
            revoke_meeting_consent,
            start_meeting_session,
            get_active_meeting_session,
            get_active_meeting_state,
            get_last_completed_meeting_state,
            list_meeting_transcript,
            read_meeting_notes,
            read_meeting_summary,
            read_meeting_action_items,
            read_meeting_decisions,
            read_meeting_diagnostics,
            generate_meeting_intelligence,
            read_meeting_intelligence,
            clear_meeting_intelligence,
            draft_meeting_followup,
            get_meeting_live_capabilities,
            pause_meeting_session,
            resume_meeting_session,
            stop_meeting_session,
            add_meeting_transcript,
            rename_meeting_speaker,
            transcribe_meeting_audio_file,
            add_meeting_action_item,
            add_meeting_decision,
            clear_meeting_session,
            detect_active_call,
            get_available_audio_devices,
            auto_detect_audio_backend,
            preview_clear_meeting_data,
            clear_meeting_data,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
