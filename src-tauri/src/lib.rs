mod action_policy;
mod assistant_context;
mod assistant_memory;
mod assistant_response;
mod audio_files;
mod audit_log;
mod browser_agent;
mod capability_manifest;
mod conversation_history;
mod conversation_router;
mod desktop_agent;
mod desktop_agent_types;
mod filesystem_service;
mod metrics;
mod model_routing;
mod pending_approvals_store;
mod permissions;
mod screen_capture;
mod screen_vision;
mod semantic_intent;
mod speech_events;
mod stt_client;
mod terminal_runner;
mod text_segmentation;
mod tools_registry;
mod tts_client;
mod vad;
mod voice_metrics;
mod voice_session;

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
    DesktopActionResponse, DesktopAuditEvent, DesktopPolicySnapshot, PendingApproval,
    ScreenAnalysisRequest, ScreenAnalysisResult, ScreenCaptureResult, ScreenObservationStatus,
    ToolDescriptor,
};
use futures_util::StreamExt;
use metrics::{MetricsTracker, RequestMetricsSnapshot};
use model_routing::resolve_ollama_request;
use reqwest::Client;
use serde::{Deserialize, Serialize};
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

        Self {
            active_request_id: Arc::new(Mutex::new(None)),
            active_voice_request_id: Arc::new(Mutex::new(None)),
            audio_files,
            metrics: MetricsTracker::new(),
            stt_client: SttClient::new(project_root.clone()),
            tts_client: TtsClient::new(project_root.clone()),
            voice_metrics: VoiceMetricsTracker::new(),
            voice_session: VoiceSessionManager::new(project_root.clone()),
            conversation_history: ConversationHistoryManager::new(),
            desktop_agent: DesktopAgentRuntime::new(project_root),
            recent_artifacts: RecentArtifactMemory::default(),
            tts_segment_fingerprints: Arc::new(Mutex::new(HashMap::new())),
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
        rationale: Some("Resolved an unambiguous follow-up against session-scoped recent artifact memory".into()),
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
            minimize_window,
            toggle_always_on_top,
            close_window,
            start_window_drag,
            set_compact_mode,
            set_expanded_mode
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
