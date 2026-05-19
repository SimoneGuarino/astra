pub use super::types::MeetingRuntimeError;
use super::{
    audio_capture::{
        wasapi_backend_available, wasapi_unavailable_reason, AudioCapture, CaptureMetricsReporter,
        CapturedSegmentQueue,
    },
    call_detector::CallDetector,
    capture_controller::{
        CaptureController, CaptureControllerConfig, CaptureControllerStartRequest,
    },
    intelligence_engine::{
        build_meeting_llm_language_retry_prompt_input, build_meeting_llm_prompt_input,
        MeetingIntelligenceEngine, MeetingIntelligenceLlm, MeetingLlmPromptInput,
        MeetingLlmPromptSegment, MeetingLlmPromptStats, OllamaMeetingIntelligenceLlm,
    },
    note_organizer::NoteOrganizer,
    privacy_control::PrivacyState,
    segment_writer::CapturedMeetingSegment,
    session_memory::SessionMemoryStore,
    session_registry::SessionRegistry,
    stt_adapter::{ExistingSttClientMeetingAdapter, MeetingFileTranscriber},
    types::{
        derive_meeting_stt_completeness, normalize_meeting_app_name, ActionItem, CallInfo,
        CaptureBackend, CaptureControllerState, CaptureHealth, ClearMeetingDataRequest,
        ConsentState, DecisionLogEntry, ExportedMeeting, MeetingAudioFileTranscriptionRequest,
        MeetingAudioFileTranscriptionResult, MeetingCapabilityReadiness, MeetingCapabilityState,
        MeetingConfig, MeetingDataClearPreview, MeetingDataClearResult,
        MeetingIntelligenceGenerationOptions, MeetingIntelligenceResult, MeetingLanguage,
        MeetingLanguageSource, MeetingLiveCapabilitySnapshot, MeetingRecallDiagnostics,
        MeetingRecallEvidence, MeetingRecallEvidenceRelation, MeetingRecallIntent,
        MeetingRecallRequest, MeetingRecallResponse, MeetingRecallStatus, MeetingScreenContext,
        MeetingScreenContextAttachResponse, MeetingSession, MeetingSessionExportRequest,
        MeetingSessionExportResponse, MeetingSessionListRequest, MeetingSessionListResponse,
        MeetingSessionMode, MeetingSessionReadRequest, MeetingSessionReadResponse,
        MeetingSessionSearchRequest, MeetingSessionSearchResponse, MeetingSessionState,
        MeetingSessionType, MeetingSessionTypeSource, MeetingStatus, RenameSpeakerRequest,
        RenameSpeakerResult, SpeakerAttributionMethod, TranscriptEntry, TranscriptSource,
        CLEAR_MEETING_DATA_CONFIRMATION_PHRASE,
    },
};
use crate::stt_client::SttClient;
use chrono::Utc;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};
use uuid::Uuid;

pub const MAX_MEETING_TRANSCRIPTION_AUDIO_BYTES: u64 = 50 * 1024 * 1024;
const SUPPORTED_MEETING_AUDIO_EXTENSIONS: &[&str] = &["wav"];
const MAX_RECALL_EVIDENCE: usize = 20;
const MAX_RECALL_PROMPT_CHARS: usize = 12_000;
const MAX_RECALL_ANSWER_CHARS: usize = 2_000;

#[derive(Clone)]
pub struct MeetingRuntime {
    registry: Arc<Mutex<SessionRegistry>>,
    privacy: Arc<Mutex<PrivacyState>>,
    capture: Arc<Mutex<CaptureController>>,
    microphone_capture: Arc<Mutex<CaptureController>>,
    stt_adapter: Arc<dyn MeetingFileTranscriber>,
    intelligence_llm: Arc<dyn MeetingIntelligenceLlm>,
    organizer: NoteOrganizer,
    session_memory: SessionMemoryStore,
    meeting_storage_dir: PathBuf,
}

impl MeetingRuntime {
    pub fn new(project_root: PathBuf) -> Self {
        Self::with_file_transcriber(
            project_root,
            Arc::new(ExistingSttClientMeetingAdapter::new()),
        )
    }

    pub fn with_stt_client(project_root: PathBuf, stt_client: SttClient) -> Self {
        Self::with_file_transcriber(
            project_root,
            Arc::new(ExistingSttClientMeetingAdapter::with_stt_client(stt_client)),
        )
    }

    pub fn with_file_transcriber(
        project_root: PathBuf,
        stt_adapter: Arc<dyn MeetingFileTranscriber>,
    ) -> Self {
        Self::with_file_transcriber_and_intelligence_llm(
            project_root,
            stt_adapter,
            Arc::new(OllamaMeetingIntelligenceLlm::new()),
        )
    }

    pub fn with_file_transcriber_and_intelligence_llm(
        project_root: PathBuf,
        stt_adapter: Arc<dyn MeetingFileTranscriber>,
        intelligence_llm: Arc<dyn MeetingIntelligenceLlm>,
    ) -> Self {
        let meeting_storage_dir = project_root.join(".astra/meetings");
        Self {
            registry: Arc::new(Mutex::new(SessionRegistry::new(project_root.clone()))),
            privacy: Arc::new(Mutex::new(PrivacyState::new())),
            capture: Arc::new(Mutex::new(CaptureController::new())),
            microphone_capture: Arc::new(Mutex::new(CaptureController::new())),
            stt_adapter,
            intelligence_llm,
            organizer: NoteOrganizer::new(meeting_storage_dir.clone()),
            session_memory: SessionMemoryStore::new(meeting_storage_dir.clone()),
            meeting_storage_dir,
        }
    }

    pub fn consent_state(&self) -> Result<ConsentState, MeetingRuntimeError> {
        let privacy = self.lock_privacy()?;
        Ok(ConsentState {
            given: privacy.consent_given,
            per_app: privacy.per_app_consent.clone(),
            global_enabled: privacy.global_enabled,
        })
    }

    pub fn grant_consent(&self, app_name: &str) -> Result<ConsentState, MeetingRuntimeError> {
        let mut privacy = self.lock_privacy()?;
        privacy.grant_consent(app_name);
        Ok(ConsentState {
            given: privacy.consent_given,
            per_app: privacy.per_app_consent.clone(),
            global_enabled: privacy.global_enabled,
        })
    }

    pub fn revoke_consent(&self, app_name: &str) -> Result<ConsentState, MeetingRuntimeError> {
        let consent_state = {
            let mut privacy = self.lock_privacy()?;
            privacy.revoke_consent(app_name);
            ConsentState {
                given: privacy.consent_given,
                per_app: privacy.per_app_consent.clone(),
                global_enabled: privacy.global_enabled,
            }
        };

        let active_session = {
            let registry = self.lock_registry()?;
            registry.get_active_session().cloned()
        };
        if let Some(session) = active_session {
            let consent_still_allows_session = {
                let privacy = self.lock_privacy()?;
                privacy.can_record(&session.platform)
            };
            if !consent_still_allows_session {
                {
                    let mut capture = self.lock_capture()?;
                    capture.record_consent_revoked(&session.platform);
                }
                {
                    let mut microphone_capture = self.lock_microphone_capture()?;
                    microphone_capture.record_consent_revoked(&session.platform);
                }
                let mut registry = self.lock_registry()?;
                if registry.get_active_session().is_some() {
                    let _ = registry.update_capture_status(
                        false,
                        Some("capture stopped: consent_revoked".to_string()),
                    );
                    let _ = registry
                        .transition_to(MeetingStatus::Failed("consent_revoked".to_string()));
                }
            }
        }

        Ok(consent_state)
    }

    pub fn start_session(
        &self,
        platform: String,
        config: MeetingConfig,
    ) -> Result<MeetingSession, MeetingRuntimeError> {
        let (platform, config) = validate_meeting_config(&platform, config)?;

        {
            let privacy = self.lock_privacy()?;
            if !privacy.can_record(&platform) {
                return Err(MeetingRuntimeError::ConsentRequired { platform });
            }
        }

        match config.session_mode {
            MeetingSessionMode::Manual => {
                let mut registry = self.lock_registry()?;
                registry.start(
                    platform,
                    config,
                    MeetingStatus::Ready,
                    false,
                    Some("manual session: no audio capture started".to_string()),
                )
            }
            MeetingSessionMode::RealCapture => self.start_real_capture_session(platform, config),
        }
    }

    fn start_real_capture_session(
        &self,
        platform: String,
        config: MeetingConfig,
    ) -> Result<MeetingSession, MeetingRuntimeError> {
        let session = {
            let mut registry = self.lock_registry()?;
            registry.start(
                platform.clone(),
                config.clone(),
                MeetingStatus::Starting,
                false,
                Some("capture controller starting".to_string()),
            )?
        };

        let options = config.capture_options;
        let emit_segments = options.segment_transcription || config.live_transcription_enabled;
        let mut started_sources = Vec::new();
        let mut failures = Vec::new();

        if options.system_audio {
            match self.start_capture_source(
                &session,
                &config,
                TranscriptSource::SystemAudio,
                emit_segments,
            ) {
                Ok(health) => started_sources.push((TranscriptSource::SystemAudio, health)),
                Err(error) => failures.push((TranscriptSource::SystemAudio, error)),
            }
        }
        if options.microphone {
            match self.start_capture_source(
                &session,
                &config,
                TranscriptSource::Microphone,
                emit_segments,
            ) {
                Ok(health) => started_sources.push((TranscriptSource::Microphone, health)),
                Err(error) => failures.push((TranscriptSource::Microphone, error)),
            }
        }

        if started_sources.is_empty() {
            let _ = self.stop_all_captures();
            let mut registry = self.lock_registry()?;
            registry.clear();
            return Err(failures
                .into_iter()
                .next()
                .map(|(_, error)| error)
                .unwrap_or_else(|| MeetingRuntimeError::InvalidConfig {
                    message: "real capture requested with no enabled audio sources".to_string(),
                }));
        }

        {
            let mut registry = self.lock_registry()?;
            for (source, error) in failures {
                registry.add_diagnostic(
                    format!("{}_capture_failed", source.as_str()),
                    super::types::MeetingDiagnosticSeverity::Warning,
                    format!("{} capture did not start: {}", source.as_str(), error),
                )?;
            }
            registry.update_capture_status(
                true,
                Some(capture_status_message(&started_sources, emit_segments)),
            )?;
            registry.transition_to(MeetingStatus::Capturing)?;
            registry
                .get_active_session()
                .cloned()
                .ok_or(MeetingRuntimeError::NoActiveSession)
        }
    }

    fn start_capture_source(
        &self,
        session: &MeetingSession,
        config: &MeetingConfig,
        transcript_source: TranscriptSource,
        emit_segments: bool,
    ) -> Result<CaptureHealth, MeetingRuntimeError> {
        let controller_config =
            CaptureControllerConfig::from_meeting_config_for_source(config, transcript_source);
        let metrics = CaptureMetricsReporter::new();
        let (segment_sender, segment_task) = if emit_segments {
            let effective_pipeline = controller_config.pipeline.effective();
            let sender = CapturedSegmentQueue::new(
                effective_pipeline.effective_max_queue_depth,
                controller_config.pipeline.overflow_policy,
                metrics.clone(),
            );
            let receiver = sender.clone();
            let runtime = self.clone();
            let task = tauri::async_runtime::spawn(async move {
                while let Some(segment) = receiver.recv().await {
                    let result = runtime
                        .transcribe_captured_segment(segment, None, true)
                        .await;
                    receiver.finish_in_flight();
                    if let Err(error) = result {
                        log::warn!("managed meeting capture segment transcription failed: {error}");
                        if matches!(
                            error,
                            MeetingRuntimeError::ConsentRevoked { .. }
                                | MeetingRuntimeError::ConsentRequired { .. }
                                | MeetingRuntimeError::NoActiveSession
                        ) {
                            break;
                        }
                    }
                }
            });
            (Some(sender), Some(task))
        } else {
            (None, None)
        };

        self.with_capture_mut_for_source(transcript_source, |capture| {
            capture.start_real_capture(CaptureControllerStartRequest {
                config: controller_config,
                session_id: session.session_id.clone(),
                meeting_storage_dir: self.meeting_storage_dir.clone(),
                segment_sender,
                segment_task,
                emit_segments,
                metrics,
            })
        })?
    }

    pub async fn transcribe_audio_file(
        &self,
        request: MeetingAudioFileTranscriptionRequest,
    ) -> Result<MeetingAudioFileTranscriptionResult, MeetingRuntimeError> {
        let active_session = {
            let registry = self.lock_registry()?;
            registry
                .get_active_session()
                .cloned()
                .ok_or(MeetingRuntimeError::NoActiveSession)?
        };

        if let Some(request_session_id) = request
            .session_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if request_session_id != active_session.session_id {
                return Err(MeetingRuntimeError::InvalidConfig {
                    message: "request.session_id does not match the active meeting session"
                        .to_string(),
                });
            }
        }

        self.ensure_can_transcribe_platform(&active_session.platform)?;

        let validated = validate_audio_file_path(&request.audio_path)?;
        let managed_path =
            self.copy_audio_to_managed_segment(&active_session.session_id, &validated)?;
        let cleanup_requested = request.cleanup_after_transcription;

        self.transcribe_managed_audio_segment(ManagedAudioTranscriptionInput {
            active_session: &active_session,
            managed_path,
            audio_file_extension: validated.extension,
            file_size_bytes: validated.size,
            speaker: request.speaker,
            cleanup_requested,
            source_is_captured_segment: false,
            transcript_source: TranscriptSource::ImportedFile,
            audio_backend: Some("imported_file".to_string()),
            segment_id: None,
            start_ms: None,
            end_ms: None,
        })
        .await
    }

    pub async fn transcribe_captured_segment(
        &self,
        segment: CapturedMeetingSegment,
        speaker: Option<String>,
        cleanup_after_transcription: bool,
    ) -> Result<MeetingAudioFileTranscriptionResult, MeetingRuntimeError> {
        let active_session = {
            let registry = self.lock_registry()?;
            registry
                .get_active_session()
                .cloned()
                .ok_or(MeetingRuntimeError::NoActiveSession)?
        };

        if segment.session_id != active_session.session_id {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "captured segment does not match the active meeting session".to_string(),
            });
        }
        let segment_id = segment
            .path
            .file_stem()
            .and_then(|value| value.to_str())
            .map(|value| value.to_string());

        if let Err(error) = self.ensure_can_transcribe_platform(&active_session.platform) {
            if matches!(error, MeetingRuntimeError::ConsentRequired { .. }) {
                self.with_capture_mut_for_source(segment.transcript_source, |capture| {
                    capture.record_consent_revoked(&active_session.platform);
                })?;
            }
            let cleanup = self
                .cleanup_managed_segment_after_failure(cleanup_after_transcription, &segment.path);
            return Err(error_with_cleanup_warning(
                MeetingRuntimeError::ConsentRevoked {
                    platform: active_session.platform.clone(),
                },
                cleanup_after_transcription,
                cleanup,
            ));
        }

        let metrics_already_recorded = segment.capture_metrics_recorded;
        let validated = match self.validate_captured_segment(&segment) {
            Ok(validated) => validated,
            Err(error) => {
                self.record_captured_segment_transcription_failure(
                    &active_session.platform,
                    segment.transcript_source,
                    segment_id.as_deref(),
                    &error,
                )?;
                let cleanup = self.cleanup_managed_segment_after_failure(
                    cleanup_after_transcription,
                    &segment.path,
                );
                return Err(error_with_cleanup_warning(
                    error,
                    cleanup_after_transcription,
                    cleanup,
                ));
            }
        };
        {
            self.with_capture_mut_for_source(segment.transcript_source, |capture| {
                if !metrics_already_recorded {
                    capture.record_segment_written(segment.byte_length, segment.duration_ms);
                }
            })?;
        }

        let start_ms = segment.start_ms;
        let end_ms = segment.end_ms;
        let result = self
            .transcribe_managed_audio_segment(ManagedAudioTranscriptionInput {
                active_session: &active_session,
                managed_path: segment.path,
                audio_file_extension: validated.extension,
                file_size_bytes: validated.size,
                speaker,
                cleanup_requested: cleanup_after_transcription,
                source_is_captured_segment: true,
                transcript_source: segment.transcript_source,
                audio_backend: Some(format!("{:?}", segment.source_backend).to_ascii_lowercase()),
                segment_id: segment_id.clone(),
                start_ms,
                end_ms,
            })
            .await;

        match &result {
            Ok(_) => {
                self.with_capture_mut_for_source(segment.transcript_source, |capture| {
                    capture.record_segment_transcribed_with_id(segment_id.as_deref());
                })?;
            }
            Err(MeetingRuntimeError::ConsentRequired { .. })
            | Err(MeetingRuntimeError::ConsentRevoked { .. }) => {
                self.with_capture_mut_for_source(segment.transcript_source, |capture| {
                    capture.record_consent_revoked(&active_session.platform);
                })?;
            }
            Err(error) => {
                self.record_captured_segment_transcription_failure(
                    &active_session.platform,
                    segment.transcript_source,
                    segment_id.as_deref(),
                    error,
                )?;
            }
        }

        match result {
            Err(MeetingRuntimeError::ConsentRequired { .. }) => {
                Err(MeetingRuntimeError::ConsentRevoked {
                    platform: active_session.platform,
                })
            }
            other => other,
        }
    }

    async fn transcribe_managed_audio_segment(
        &self,
        input: ManagedAudioTranscriptionInput<'_>,
    ) -> Result<MeetingAudioFileTranscriptionResult, MeetingRuntimeError> {
        let transcript_text = match self.stt_adapter.transcribe_file(&input.managed_path).await {
            Ok(text) => text,
            Err(error) => {
                let cleanup = self.cleanup_managed_segment_after_failure(
                    input.cleanup_requested,
                    &input.managed_path,
                );
                return Err(error_with_cleanup_warning(
                    error,
                    input.cleanup_requested,
                    cleanup,
                ));
            }
        };
        let transcript_text = transcript_text.trim().to_string();
        if transcript_text.is_empty() {
            let cleanup = self.cleanup_managed_segment_after_failure(
                input.cleanup_requested,
                &input.managed_path,
            );
            let error = MeetingRuntimeError::TranscriptionUnavailable {
                reason: "Existing STT returned an empty transcript".to_string(),
            };
            return Err(error_with_cleanup_warning(
                error,
                input.cleanup_requested,
                cleanup,
            ));
        }

        if let Err(error) = self.ensure_can_transcribe_platform(&input.active_session.platform) {
            let cleanup = self.cleanup_managed_segment_after_failure(
                input.cleanup_requested,
                &input.managed_path,
            );
            return Err(error_with_cleanup_warning(
                error,
                input.cleanup_requested,
                cleanup,
            ));
        }

        let requested_speaker = input
            .speaker
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let speaker = requested_speaker
            .unwrap_or(if input.source_is_captured_segment {
                "unknown"
            } else {
                "stt"
            })
            .to_string();
        let text_length = transcript_text.chars().count();
        let result_segment_id = input
            .segment_id
            .clone()
            .unwrap_or_else(super::types::new_meeting_artifact_id);

        let transcript_index = match self.lock_registry().and_then(|mut registry| {
            match registry.get_active_session() {
                Some(still_active_session)
                    if still_active_session.session_id == input.active_session.session_id =>
                {
                    let index = registry.get_active_state().transcript.len();
                    let speaker_id =
                        if requested_speaker.is_none() && input.source_is_captured_segment {
                            Some(format!("capture_segment_{}", index + 1))
                        } else {
                            None
                        };
                    let now = Utc::now();
                    let entry = TranscriptEntry {
                        segment_id: result_segment_id.clone(),
                        session_id: input.active_session.session_id.clone(),
                        source: input.transcript_source,
                        timestamp: now,
                        created_at: now,
                        speaker,
                        speaker_id,
                        speaker_label: None,
                        speaker_confidence: None,
                        speaker_attribution_method: SpeakerAttributionMethod::Unknown,
                        text: transcript_text,
                        confidence: 0.0,
                        start_ms: input.start_ms,
                        end_ms: input.end_ms,
                        stt_model: Some(input.active_session.config.transcription_model.clone()),
                        audio_backend: input.audio_backend.clone(),
                    };
                    registry.add_transcript(entry)?;
                    Ok(registry
                        .get_active_state()
                        .transcript
                        .iter()
                        .position(|entry| entry.segment_id == result_segment_id.as_str())
                        .unwrap_or(index))
                }
                Some(_) => Err(MeetingRuntimeError::InvalidConfig {
                    message: "active meeting session changed before transcription completed"
                        .to_string(),
                }),
                None => Err(MeetingRuntimeError::NoActiveSession),
            }
        }) {
            Ok(index) => index,
            Err(error) => {
                let cleanup = self.cleanup_managed_segment_after_failure(
                    input.cleanup_requested,
                    &input.managed_path,
                );
                return Err(error_with_cleanup_warning(
                    error,
                    input.cleanup_requested,
                    cleanup,
                ));
            }
        };

        let cleanup = if input.cleanup_requested {
            self.cleanup_managed_segment_best_effort(&input.managed_path)
        } else {
            ManagedSegmentCleanupOutcome::not_requested()
        };

        Ok(MeetingAudioFileTranscriptionResult {
            transcript_added: true,
            transcript_index,
            text_length,
            audio_file_extension: input.audio_file_extension,
            file_size_bytes: input.file_size_bytes,
            stt_boundary: self.stt_adapter.boundary().to_string(),
            transcript_source: input.transcript_source,
            segment_id: result_segment_id,
            start_ms: input.start_ms,
            end_ms: input.end_ms,
            source_audio_path_redacted: true,
            managed_audio_path_redacted: true,
            cleanup_requested: input.cleanup_requested,
            cleanup_performed: cleanup.performed,
            cleanup_error: cleanup.error,
        })
    }

    pub fn get_active_session(&self) -> Result<Option<MeetingSession>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_session().cloned())
    }

    pub fn get_active_state(&self) -> Result<MeetingSessionState, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_state().clone())
    }

    pub fn get_last_completed_state(
        &self,
    ) -> Result<Option<MeetingSessionState>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_last_completed_state().cloned())
    }

    pub fn list_transcript(&self) -> Result<Vec<TranscriptEntry>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_state().transcript.clone())
    }

    pub fn read_notes(&self) -> Result<Vec<super::types::NoteEntry>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_state().notes.clone())
    }

    pub fn read_summary(&self) -> Result<Vec<super::types::SummaryEntry>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_state().summary.clone())
    }

    pub fn read_action_items(&self) -> Result<Vec<super::types::ActionItem>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_state().action_items.clone())
    }

    pub fn read_decisions(
        &self,
    ) -> Result<Vec<super::types::DecisionLogEntry>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_state().decisions.clone())
    }

    pub fn read_diagnostics(
        &self,
    ) -> Result<Vec<super::types::MeetingDiagnostic>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_active_state().diagnostics.clone())
    }

    pub async fn generate_intelligence(
        &self,
        options: MeetingIntelligenceGenerationOptions,
    ) -> Result<MeetingIntelligenceResult, MeetingRuntimeError> {
        let generation_started_at = Instant::now();
        let snapshot = {
            let mut registry = self.lock_registry()?;
            registry.begin_intelligence_generation(options)?
        };
        let input = snapshot.input.clone();

        let mut result = if input.generation_options.use_local_llm {
            let prompt_input = build_meeting_llm_prompt_input(&input);
            match self
                .intelligence_llm
                .generate_intelligence_json(prompt_input.clone())
                .await
            {
                Ok(output) => {
                    let mut first_result =
                        MeetingIntelligenceEngine::generate_with_llm_output_or_rule_based(
                            input.clone(),
                            Some(output),
                            None,
                        )?;
                    if should_retry_meeting_output_language(&first_result) {
                        first_result.diagnostics.language_retry_attempted = true;
                        first_result.diagnostics.warnings.push(
                            "Output language did not match transcript language; one local model retry was attempted".to_string(),
                        );
                        let retry_prompt =
                            build_meeting_llm_language_retry_prompt_input(&prompt_input);
                        match self
                            .intelligence_llm
                            .generate_intelligence_json(retry_prompt)
                            .await
                        {
                            Ok(retry_output) => {
                                let mut retry_result =
                                    MeetingIntelligenceEngine::generate_with_llm_output_or_rule_based(
                                        input.clone(),
                                        Some(retry_output),
                                        None,
                                    )?;
                                retry_result.diagnostics.language_retry_attempted = true;
                                if !retry_result.diagnostics.output_language_mismatch
                                    && !retry_result.diagnostics.fallback_used
                                {
                                    retry_result.diagnostics.language_retry_succeeded = true;
                                    retry_result
                                } else {
                                    first_result.diagnostics.language_retry_succeeded = false;
                                    first_result.diagnostics.warnings.push(
                                        "Language retry did not produce an acceptable corrected local model result; first valid output was kept with mismatch diagnostics".to_string(),
                                    );
                                    first_result
                                }
                            }
                            Err(error) => {
                                first_result.diagnostics.language_retry_succeeded = false;
                                first_result.diagnostics.warnings.push(format!(
                                    "Language retry failed with {}; first valid output was kept",
                                    error.reason_code()
                                ));
                                first_result
                            }
                        }
                    } else {
                        first_result
                    }
                }
                Err(error) => MeetingIntelligenceEngine::generate_with_llm_output_or_rule_based(
                    input,
                    None,
                    Some(error),
                )?,
            }
        } else {
            MeetingIntelligenceEngine::generate_with_llm_output_or_rule_based(input, None, None)?
        };
        result.diagnostics.total_generation_duration_ms = Some(
            generation_started_at
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64,
        );

        let mut registry = self.lock_registry()?;
        registry.store_intelligence_result(&snapshot, result)
    }

    pub fn read_intelligence(
        &self,
    ) -> Result<Option<MeetingIntelligenceResult>, MeetingRuntimeError> {
        let registry = self.lock_registry()?;
        Ok(registry.get_intelligence().cloned())
    }

    pub fn clear_intelligence(&self) -> Result<(), MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        registry.clear_intelligence()
    }

    pub fn pause_session(&self) -> Result<(), MeetingRuntimeError> {
        {
            let mut registry = self.lock_registry()?;
            registry.pause()?;
        }

        self.pause_capture_source(TranscriptSource::SystemAudio)?;
        self.pause_capture_source(TranscriptSource::Microphone)?;
        Ok(())
    }

    pub fn resume_session(&self) -> Result<(), MeetingRuntimeError> {
        let resumed_status = {
            let mut registry = self.lock_registry()?;
            registry.resume()?;
            registry.get_active_state().status.clone()
        };

        if matches!(
            resumed_status,
            MeetingStatus::Capturing | MeetingStatus::Transcribing
        ) {
            self.resume_capture_source(TranscriptSource::SystemAudio)?;
            self.resume_capture_source(TranscriptSource::Microphone)?;
        }
        Ok(())
    }

    pub fn stop_session(&self) -> Result<ExportedMeeting, MeetingRuntimeError> {
        {
            let mut registry = self.lock_registry()?;
            if registry.get_active_session().is_some() {
                let _ = registry.transition_to(MeetingStatus::Stopping);
            }
        }
        let stop_result = self.stop_all_captures();
        if let Err(error) = stop_result {
            let mut registry = self.lock_registry()?;
            if registry.get_active_session().is_some() {
                let _ = registry.update_capture_status(
                    false,
                    Some("capture stop failed: redacted_failure".to_string()),
                );
                let _ = registry
                    .transition_to(MeetingStatus::Failed("capture_stop_failed".to_string()));
            }
            return Err(error);
        }
        let system_capture_health = self.capture_health()?;
        let microphone_capture_health = self.microphone_capture_health()?;
        self.record_segment_stt_stop_diagnostics(
            &system_capture_health,
            &microphone_capture_health,
        )?;
        let mut exported = {
            let mut registry = self.lock_registry()?;
            registry.stop()?
        };
        apply_segment_stt_export_metadata(
            &mut exported,
            &system_capture_health,
            &microphone_capture_health,
        );
        let completed_state = {
            let registry = self.lock_registry()?;
            registry
                .get_last_completed_state()
                .cloned()
                .ok_or(MeetingRuntimeError::NoActiveSession)?
        };
        let combined_capture_health = aggregate_capture_health(
            system_capture_health.clone(),
            microphone_capture_health.clone(),
        );
        self.organizer
            .save_meeting_data(&exported)
            .map_err(|message| MeetingRuntimeError::StorageError { message })?;
        self.session_memory.archive_completed_session(
            &completed_state,
            &exported,
            &combined_capture_health,
            &system_capture_health,
            &microphone_capture_health,
        )?;
        Ok(exported)
    }

    pub fn list_archived_sessions(
        &self,
        request: MeetingSessionListRequest,
    ) -> Result<MeetingSessionListResponse, MeetingRuntimeError> {
        self.session_memory.list_sessions(request)
    }

    pub fn read_archived_session(
        &self,
        request: MeetingSessionReadRequest,
    ) -> Result<MeetingSessionReadResponse, MeetingRuntimeError> {
        self.session_memory.read_session(request)
    }

    pub fn search_archived_sessions(
        &self,
        request: MeetingSessionSearchRequest,
    ) -> Result<MeetingSessionSearchResponse, MeetingRuntimeError> {
        self.session_memory.search_sessions(request)
    }

    pub async fn answer_session_recall(
        &self,
        request: MeetingRecallRequest,
    ) -> Result<MeetingRecallResponse, MeetingRuntimeError> {
        let query_language = detect_recall_query_language(&request.query);
        let mut bounded_request = request.clone();
        bounded_request.limit = bounded_recall_limit(request.limit);
        let retrieval = self
            .session_memory
            .retrieve_recall_evidence(&bounded_request)?;
        let mut diagnostics = MeetingRecallDiagnostics {
            generator: "extractive_fallback".to_string(),
            use_local_llm: request.use_local_llm,
            llm_used: false,
            fallback_used: !request.use_local_llm,
            model_provider: None,
            model_name: None,
            llm_endpoint: None,
            query_char_count: request.query.chars().count(),
            query_language,
            recall_intent: retrieval.intent,
            searched_session_count: retrieval.searched_session_count,
            matched_session_count: retrieval.matched_session_count,
            evidence_count: retrieval.evidence.len(),
            max_evidence: bounded_request.limit,
            retrieval_truncated: retrieval.truncated,
            corrupt_archive_count: retrieval.corrupt_archive_count,
            prompt_char_count: 0,
            output_json_parse_failed: false,
            invalid_evidence_indexes: 0,
            follow_up_questions: Vec::new(),
            warnings: retrieval
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.code.clone())
                .collect(),
            generated_at: Utc::now(),
        };

        if retrieval.evidence.is_empty() {
            diagnostics.fallback_used = true;
            return Ok(MeetingRecallResponse {
                answer: no_recall_evidence_answer(query_language),
                status: MeetingRecallStatus::InsufficientEvidence,
                evidence: Vec::new(),
                sessions: Vec::new(),
                diagnostics,
            });
        }

        if request.use_local_llm {
            let prompt_input = build_recall_llm_prompt_input(
                &request.query,
                query_language,
                retrieval.intent,
                &retrieval.evidence,
            );
            diagnostics.prompt_char_count = prompt_input.prompt.chars().count();
            match self
                .intelligence_llm
                .generate_intelligence_json(prompt_input)
                .await
            {
                Ok(output) => {
                    diagnostics.llm_used = true;
                    diagnostics.generator = "local_llm".to_string();
                    diagnostics.model_provider = Some(output.provider);
                    diagnostics.model_name = Some(output.model);
                    diagnostics.llm_endpoint = output.endpoint;
                    match parse_recall_llm_output(&output.raw_json, &retrieval.evidence) {
                        Ok(parsed) => {
                            diagnostics.invalid_evidence_indexes = parsed.invalid_evidence_indexes;
                            diagnostics.follow_up_questions = parsed.follow_up_questions.clone();
                            if diagnostics.invalid_evidence_indexes > 0 {
                                diagnostics.warnings.push(
                                    "local_llm_returned_invalid_evidence_indexes".to_string(),
                                );
                            }
                            return Ok(MeetingRecallResponse {
                                answer: parsed.answer,
                                status: parsed.status,
                                evidence: parsed.evidence,
                                sessions: retrieval.sessions,
                                diagnostics,
                            });
                        }
                        Err(failure) => {
                            diagnostics.output_json_parse_failed = true;
                            diagnostics.fallback_used = true;
                            diagnostics.generator = "extractive_fallback".to_string();
                            diagnostics.invalid_evidence_indexes = failure.invalid_evidence_indexes;
                            diagnostics.warnings.push(failure.code);
                        }
                    }
                }
                Err(error) => {
                    let reason_code = error.reason_code();
                    diagnostics.fallback_used = true;
                    diagnostics.generator = "extractive_fallback".to_string();
                    diagnostics.model_provider = Some(error.provider);
                    diagnostics.model_name = error.model;
                    diagnostics.llm_endpoint = error.endpoint;
                    diagnostics
                        .warnings
                        .push(format!("local_llm_failed:{reason_code}"));
                }
            }
        }

        Ok(MeetingRecallResponse {
            answer: extractive_recall_answer(
                query_language,
                retrieval.intent,
                &retrieval.evidence,
                request.use_local_llm,
            ),
            status: if request.use_local_llm {
                MeetingRecallStatus::Degraded
            } else {
                MeetingRecallStatus::Answered
            },
            evidence: retrieval.evidence,
            sessions: retrieval.sessions,
            diagnostics,
        })
    }

    pub fn export_archived_session(
        &self,
        request: MeetingSessionExportRequest,
    ) -> Result<MeetingSessionExportResponse, MeetingRuntimeError> {
        self.session_memory.export_session(request)
    }

    pub fn rebuild_session_memory_index(
        &self,
    ) -> Result<MeetingSessionListResponse, MeetingRuntimeError> {
        self.session_memory.rebuild_index()
    }

    pub fn attach_screen_context(
        &self,
        context: MeetingScreenContext,
    ) -> Result<MeetingScreenContextAttachResponse, MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        registry.add_screen_context(context)
    }

    pub fn add_transcript(&self, mut entry: TranscriptEntry) -> Result<(), MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        let active_session = registry
            .get_active_session()
            .cloned()
            .ok_or(MeetingRuntimeError::NoActiveSession)?;
        entry.session_id = active_session.session_id;
        entry.source = TranscriptSource::Manual;
        entry.stt_model = None;
        entry.audio_backend = None;
        registry.add_transcript(entry)
    }

    pub fn rename_speaker(
        &self,
        request: RenameSpeakerRequest,
    ) -> Result<RenameSpeakerResult, MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        registry.rename_speaker(&request.speaker_id, &request.display_name)
    }

    pub fn add_action_item(&self, item: ActionItem) -> Result<(), MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        registry.add_action_item(item)
    }

    pub fn add_decision(&self, entry: DecisionLogEntry) -> Result<(), MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        registry.add_decision(entry)
    }

    pub fn clear_runtime_session(&self) -> Result<(), MeetingRuntimeError> {
        self.stop_capture_for_clear_operation("clear_runtime_session")?;
        let mut registry = self.lock_registry()?;
        registry.clear();
        Ok(())
    }

    pub fn preview_clear_all_data(&self) -> Result<MeetingDataClearPreview, MeetingRuntimeError> {
        let runtime_state_present = {
            let registry = self.lock_registry()?;
            registry.has_runtime_state()
        };
        let mut preview = self
            .organizer
            .preview_clear_all_meeting_data()
            .map_err(|message| MeetingRuntimeError::StorageError { message })?;
        preview.runtime_state_present = runtime_state_present;
        Ok(preview)
    }

    pub fn clear_all_data(
        &self,
        request: ClearMeetingDataRequest,
    ) -> Result<MeetingDataClearResult, MeetingRuntimeError> {
        let ClearMeetingDataRequest {
            scope: _scope,
            confirmation_phrase,
        } = request;
        if confirmation_phrase != CLEAR_MEETING_DATA_CONFIRMATION_PHRASE {
            return Err(MeetingRuntimeError::ConfirmationRequired {
                action: "meeting.clear_data".to_string(),
                required_phrase: CLEAR_MEETING_DATA_CONFIRMATION_PHRASE.to_string(),
            });
        }
        let capture_stop = self.stop_capture_for_clear_operation("clear_data")?;
        {
            let mut registry = self.lock_registry()?;
            registry.clear();
        }
        let mut result = self
            .organizer
            .clear_all_meeting_data()
            .map_err(|message| MeetingRuntimeError::StorageError { message })?;
        result.capture_stop_attempted = capture_stop.attempted;
        result.capture_stop_succeeded = capture_stop.succeeded;
        result.capture_stop_error_kind = capture_stop.error_kind;
        result.clear_aborted = false;
        Ok(result)
    }

    pub fn detect_active_call(&self) -> Option<CallInfo> {
        CallDetector::detect()
    }

    pub fn available_audio_devices(&self) -> Result<Vec<String>, MeetingRuntimeError> {
        AudioCapture::list_available_devices()
            .map_err(|message| MeetingRuntimeError::StorageError { message })
    }

    pub fn auto_detect_audio_backend(&self) -> CaptureBackend {
        AudioCapture::auto_detect_backend()
    }

    pub fn capture_health(&self) -> Result<CaptureHealth, MeetingRuntimeError> {
        let capture = self.lock_capture()?;
        Ok(capture.health_snapshot())
    }

    pub fn microphone_capture_health(&self) -> Result<CaptureHealth, MeetingRuntimeError> {
        let capture = self.lock_microphone_capture()?;
        Ok(capture.health_snapshot())
    }

    pub fn live_capabilities(&self) -> Result<MeetingLiveCapabilitySnapshot, MeetingRuntimeError> {
        let system_capture_health = self.capture_health()?;
        let microphone_capture_health = self.microphone_capture_health()?;
        let capture_health = aggregate_capture_health(
            system_capture_health.clone(),
            microphone_capture_health.clone(),
        );
        let stt_status = self.stt_adapter.status();
        let wasapi_available = wasapi_backend_available();
        let live_segment_transcription_ready =
            capture_health.active_handle_present && stt_status.file_transcription.available;

        Ok(MeetingLiveCapabilitySnapshot {
            manual_session: readiness(
                "meeting.session.manual",
                true,
                MeetingCapabilityState::Ready,
                None,
            ),
            audio_capture: readiness(
                "meeting.audio.capture",
                wasapi_available,
                if wasapi_available {
                    MeetingCapabilityState::Ready
                } else {
                    MeetingCapabilityState::Unavailable
                },
                if wasapi_available {
                    None
                } else {
                    Some(wasapi_unavailable_reason())
                },
            ),
            microphone_capture: readiness(
                "meeting.audio.capture.microphone",
                wasapi_available,
                if wasapi_available {
                    MeetingCapabilityState::Ready
                } else {
                    MeetingCapabilityState::Unavailable
                },
                if wasapi_available {
                    microphone_capture_health
                        .last_error
                        .clone()
                        .or_else(|| Some("Windows default microphone endpoint is checked when capture starts".to_string()))
                } else {
                    Some(wasapi_unavailable_reason())
                },
            ),
            system_audio_capture: readiness(
                "meeting.audio.capture.system",
                wasapi_available,
                if wasapi_available {
                    MeetingCapabilityState::Ready
                } else {
                    MeetingCapabilityState::Unavailable
                },
                if wasapi_available {
                    None
                } else {
                    Some(wasapi_unavailable_reason())
                },
            ),
            windows_wasapi_capture: readiness(
                "meeting.audio.capture.wasapi",
                wasapi_available,
                if wasapi_available {
                    MeetingCapabilityState::Ready
                } else {
                    MeetingCapabilityState::Unavailable
                },
                if wasapi_available {
                    None
                } else {
                    Some(wasapi_unavailable_reason())
                },
            ),
            system_capture_health,
            microphone_capture_health,
            live_transcription: readiness(
                "meeting.transcription.live",
                false,
                MeetingCapabilityState::Unavailable,
                stt_status.live_transcription.reason.clone(),
            ),
            live_segment_transcription: readiness(
                "meeting.transcription.segment",
                live_segment_transcription_ready,
                if live_segment_transcription_ready {
                    MeetingCapabilityState::Ready
                } else {
                    MeetingCapabilityState::Unavailable
                },
                if live_segment_transcription_ready {
                    None
                } else {
                    Some("Live segment transcription requires an active governed capture handle and the existing file STT bridge".to_string())
                },
            ),
            live_streaming_stt: readiness(
                "meeting.transcription.streaming",
                false,
                MeetingCapabilityState::Unavailable,
                Some("Streaming STT protocol is not implemented; only completed managed WAV files can use SttClient::transcribe(Path)".to_string()),
            ),
            chunk_streaming: stt_status.chunk_streaming.clone(),
            diarization: readiness(
                "meeting.diarization.live",
                false,
                MeetingCapabilityState::Unavailable,
                Some("No tested diarization backend is connected; captured segments use non-identifying capture segment metadata only".to_string()),
            ),
            live_summarization: readiness(
                "meeting.summarization.live",
                true,
                MeetingCapabilityState::Ready,
                Some("Rule-based transcript-derived notes, decisions, action items, and rolling summaries are active; no model-only summaries are fabricated".to_string()),
            ),
            follow_up: readiness(
                "meeting.followup.send",
                false,
                MeetingCapabilityState::Unavailable,
                Some("Follow-up sending remains disabled until draft-first outbound integrations are governed".to_string()),
            ),
            capture_health,
            stt_adapter: stt_status,
        })
    }

    fn record_captured_segment_transcription_failure(
        &self,
        platform: &str,
        source: TranscriptSource,
        segment_id: Option<&str>,
        error: &MeetingRuntimeError,
    ) -> Result<(), MeetingRuntimeError> {
        let error_kind = captured_segment_error_kind(error);
        let immediate_failure = captured_segment_error_fails_capture_immediately(error);
        let terminal_reason = {
            self.with_capture_mut_for_source(source, |capture| {
                if immediate_failure {
                    capture.record_terminal_segment_transcription_failure(&error_kind);
                    Some(format!(
                        "{}_segment_transcription_failed:{error_kind}",
                        source.as_str()
                    ))
                } else {
                    let health = capture
                        .record_segment_transcription_failure_with_id(&error_kind, segment_id);
                    let threshold = health
                        .pipeline
                        .max_consecutive_transcription_failures
                        .max(1) as u64;
                    if health.metrics.segment_transcription_failures_consecutive >= threshold {
                        let _ = capture.abort(format!(
                            "{}_segment_transcription_failure_threshold",
                            source.as_str()
                        ));
                        Some(format!(
                            "{}_segment_transcription_failure_threshold",
                            source.as_str()
                        ))
                    } else {
                        None
                    }
                }
            })?
        };
        {
            let mut registry = self.lock_registry()?;
            if registry.get_active_session().is_some() {
                registry.add_diagnostic(
                    format!("{}_segment_stt_failed", source.as_str()),
                    super::types::MeetingDiagnosticSeverity::Warning,
                    format!(
                        "{} segment transcription failed: {error_kind}",
                        source.as_str()
                    ),
                )?;
            }
        }

        if let Some(reason) = terminal_reason {
            self.mark_active_session_capture_failed(platform, &reason)?;
        }
        Ok(())
    }

    fn mark_active_session_capture_failed(
        &self,
        platform: &str,
        reason: &str,
    ) -> Result<(), MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        let matches_platform = registry.get_active_session().is_some_and(|session| {
            normalize_meeting_app_name(&session.platform) == normalize_meeting_app_name(platform)
        });
        if matches_platform {
            let _ =
                registry.update_capture_status(false, Some(format!("capture failed: {reason}")));
            let _ = registry.transition_to(MeetingStatus::Failed(reason.to_string()));
        }
        Ok(())
    }

    fn stop_capture_for_clear_operation(
        &self,
        operation: &str,
    ) -> Result<CaptureStopForClearOutcome, MeetingRuntimeError> {
        let attempted = self.any_capture_stop_needed()?;
        let stop_result = self.stop_all_captures();

        match stop_result {
            Ok(_) => Ok(CaptureStopForClearOutcome {
                attempted,
                succeeded: attempted,
                error_kind: None,
            }),
            Err(error) => {
                let error_kind = capture_stop_error_kind(&error).to_string();
                self.mark_clear_aborted_after_stop_failure(operation, &error_kind)?;
                Err(MeetingRuntimeError::ClearAbortedCaptureStopFailed {
                    operation: operation.to_string(),
                    error_kind,
                })
            }
        }
    }

    fn mark_clear_aborted_after_stop_failure(
        &self,
        operation: &str,
        error_kind: &str,
    ) -> Result<(), MeetingRuntimeError> {
        let mut registry = self.lock_registry()?;
        if registry.get_active_session().is_some() {
            let _ = registry
                .update_capture_status(false, Some(format!("{operation} aborted: {error_kind}")));
            let _ =
                registry.transition_to(MeetingStatus::Failed(format!("{operation}_{error_kind}")));
        }
        Ok(())
    }

    fn lock_registry(&self) -> Result<MutexGuard<'_, SessionRegistry>, MeetingRuntimeError> {
        self.registry
            .lock()
            .map_err(|_| MeetingRuntimeError::MutexPoisoned {
                component: "session_registry".to_string(),
            })
    }

    fn lock_privacy(&self) -> Result<MutexGuard<'_, PrivacyState>, MeetingRuntimeError> {
        self.privacy
            .lock()
            .map_err(|_| MeetingRuntimeError::MutexPoisoned {
                component: "privacy_state".to_string(),
            })
    }

    fn lock_capture(&self) -> Result<MutexGuard<'_, CaptureController>, MeetingRuntimeError> {
        self.capture
            .lock()
            .map_err(|_| MeetingRuntimeError::MutexPoisoned {
                component: "capture_controller".to_string(),
            })
    }

    fn lock_microphone_capture(
        &self,
    ) -> Result<MutexGuard<'_, CaptureController>, MeetingRuntimeError> {
        self.microphone_capture
            .lock()
            .map_err(|_| MeetingRuntimeError::MutexPoisoned {
                component: "microphone_capture_controller".to_string(),
            })
    }

    fn with_capture_mut_for_source<T>(
        &self,
        source: TranscriptSource,
        operation: impl FnOnce(&mut CaptureController) -> T,
    ) -> Result<T, MeetingRuntimeError> {
        match source {
            TranscriptSource::Microphone => {
                let mut capture = self.lock_microphone_capture()?;
                Ok(operation(&mut capture))
            }
            _ => {
                let mut capture = self.lock_capture()?;
                Ok(operation(&mut capture))
            }
        }
    }

    fn pause_capture_source(&self, source: TranscriptSource) -> Result<(), MeetingRuntimeError> {
        self.with_capture_mut_for_source(source, |capture| {
            if capture.has_active_handle()
                || matches!(capture.state(), CaptureControllerState::Capturing)
            {
                capture.pause()
            } else {
                Ok(capture.health_snapshot())
            }
        })??;
        Ok(())
    }

    fn resume_capture_source(&self, source: TranscriptSource) -> Result<(), MeetingRuntimeError> {
        self.with_capture_mut_for_source(source, |capture| {
            if capture.has_active_handle()
                || matches!(capture.state(), CaptureControllerState::Paused)
            {
                capture.resume()
            } else {
                Ok(capture.health_snapshot())
            }
        })??;
        Ok(())
    }

    fn any_capture_stop_needed(&self) -> Result<bool, MeetingRuntimeError> {
        let system_needed = {
            let capture = self.lock_capture()?;
            capture.has_active_handle()
                || matches!(
                    capture.state(),
                    CaptureControllerState::Starting
                        | CaptureControllerState::Capturing
                        | CaptureControllerState::Paused
                        | CaptureControllerState::Stopping
                )
        };
        let microphone_needed = {
            let capture = self.lock_microphone_capture()?;
            capture.has_active_handle()
                || matches!(
                    capture.state(),
                    CaptureControllerState::Starting
                        | CaptureControllerState::Capturing
                        | CaptureControllerState::Paused
                        | CaptureControllerState::Stopping
                )
        };
        Ok(system_needed || microphone_needed)
    }

    fn stop_all_captures(&self) -> Result<(), MeetingRuntimeError> {
        let system_result = {
            let mut capture = self.lock_capture()?;
            capture.stop()
        };
        let microphone_result = {
            let mut capture = self.lock_microphone_capture()?;
            capture.stop()
        };
        system_result?;
        microphone_result?;
        Ok(())
    }

    fn record_segment_stt_stop_diagnostics(
        &self,
        system_health: &CaptureHealth,
        microphone_health: &CaptureHealth,
    ) -> Result<(), MeetingRuntimeError> {
        let aggregate = aggregate_capture_health(system_health.clone(), microphone_health.clone());
        let metrics = &aggregate.metrics;
        let mut registry = self.lock_registry()?;
        if registry.get_active_session().is_none() {
            return Ok(());
        }

        if matches!(
            metrics.segment_transcription_drain_status.as_deref(),
            Some("running") | Some("completed") | Some("timed_out")
        ) {
            let severity = if metrics.drain_timeout {
                super::types::MeetingDiagnosticSeverity::Warning
            } else {
                super::types::MeetingDiagnosticSeverity::Info
            };
            registry.add_diagnostic(
                "meeting_segment_stt_drain".to_string(),
                severity,
                format!(
                    "Segment STT drain {}; queued_total={}; current_queue={}; in_flight={}; transcribed={}; failed={}",
                    metrics
                        .segment_transcription_drain_status
                        .as_deref()
                        .unwrap_or("unknown"),
                    metrics.segments_queued_total,
                    metrics.current_queue_depth,
                    metrics.segments_in_flight,
                    metrics.segments_transcribed,
                    metrics.segments_failed
                ),
            )?;
        }

        let transcript_count = registry.get_active_state().transcript.len();
        if metrics.segments_written > 0
            && metrics.segments_transcribed == 0
            && transcript_count == 0
        {
            registry.add_diagnostic(
                "meeting_segment_transcription_incomplete".to_string(),
                super::types::MeetingDiagnosticSeverity::Warning,
                format!(
                    "Audio segments were captured, but no transcript was produced before stop/export; segments_written={}; queued_total={}; current_queue={}; in_flight={}; failed={}; timeouts={}; last_error={}",
                    metrics.segments_written,
                    metrics.segments_queued_total,
                    metrics.current_queue_depth,
                    metrics.segments_in_flight,
                    metrics.segments_failed,
                    metrics.segment_transcription_timeouts,
                    metrics
                        .last_segment_transcription_error_kind
                        .as_deref()
                        .unwrap_or("none")
                ),
            )?;
        }
        Ok(())
    }

    fn ensure_can_transcribe_platform(&self, platform: &str) -> Result<(), MeetingRuntimeError> {
        let privacy = self.lock_privacy()?;
        if !privacy.can_record(platform) {
            return Err(MeetingRuntimeError::ConsentRequired {
                platform: platform.to_string(),
            });
        }
        Ok(())
    }

    fn copy_audio_to_managed_segment(
        &self,
        session_id: &str,
        audio: &ValidatedAudioFile,
    ) -> Result<PathBuf, MeetingRuntimeError> {
        let segments_dir = self.meeting_storage_dir.join(session_id).join("segments");
        std::fs::create_dir_all(&segments_dir).map_err(|error| {
            MeetingRuntimeError::StorageError {
                message: format!(
                    "create managed meeting segment directory failed: {}",
                    error.kind()
                ),
            }
        })?;
        let managed_path = segments_dir.join(format!("{}.{}", Uuid::new_v4(), audio.extension));
        let copied = std::fs::copy(&audio.canonical_path, &managed_path).map_err(|error| {
            MeetingRuntimeError::StorageError {
                message: format!("copy managed meeting segment failed: {}", error.kind()),
            }
        })?;
        if copied != audio.size {
            return Err(MeetingRuntimeError::StorageError {
                message: "copy managed meeting segment failed: copied byte count mismatch"
                    .to_string(),
            });
        }
        Ok(managed_path)
    }

    fn validate_captured_segment(
        &self,
        segment: &CapturedMeetingSegment,
    ) -> Result<ValidatedAudioFile, MeetingRuntimeError> {
        if !self.is_managed_segment_path(&segment.path) {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "captured segment path is outside managed meeting storage".to_string(),
            });
        }
        let metadata = segment
            .path
            .metadata()
            .map_err(|_| MeetingRuntimeError::InvalidConfig {
                message: "captured segment metadata cannot be read".to_string(),
            })?;
        if !metadata.is_file() {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "captured segment must be a managed file".to_string(),
            });
        }
        let extension = segment
            .path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .unwrap_or_default();
        if extension != "wav" {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "captured segment must be a managed WAV file".to_string(),
            });
        }
        let size = metadata.len();
        if size > MAX_MEETING_TRANSCRIPTION_AUDIO_BYTES {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: format!(
                    "captured segment is too large; maximum is {} bytes",
                    MAX_MEETING_TRANSCRIPTION_AUDIO_BYTES
                ),
            });
        }
        validate_wav_header(&segment.path, size)?;
        Ok(ValidatedAudioFile {
            canonical_path: segment.path.clone(),
            extension,
            size,
        })
    }

    fn cleanup_managed_segment_after_failure(
        &self,
        cleanup_requested: bool,
        managed_path: &Path,
    ) -> ManagedSegmentCleanupOutcome {
        if cleanup_requested {
            self.cleanup_managed_segment_best_effort(managed_path)
        } else {
            ManagedSegmentCleanupOutcome::not_requested()
        }
    }

    fn cleanup_managed_segment_best_effort(
        &self,
        managed_path: &Path,
    ) -> ManagedSegmentCleanupOutcome {
        if !self.is_managed_segment_path(managed_path) {
            return ManagedSegmentCleanupOutcome {
                performed: false,
                error: Some(
                    "managed meeting segment cleanup skipped: path outside managed segments"
                        .to_string(),
                ),
            };
        }

        match std::fs::remove_file(managed_path) {
            Ok(()) => ManagedSegmentCleanupOutcome {
                performed: true,
                error: None,
            },
            Err(error) => ManagedSegmentCleanupOutcome {
                performed: false,
                error: Some(format!(
                    "managed meeting segment cleanup failed: {}",
                    error.kind()
                )),
            },
        }
    }

    fn is_managed_segment_path(&self, managed_path: &Path) -> bool {
        managed_path.starts_with(&self.meeting_storage_dir)
            && managed_path
                .parent()
                .and_then(|parent| parent.file_name())
                .is_some_and(|name| name == "segments")
    }

    #[doc(hidden)]
    pub fn install_fake_active_capture_for_test(
        &self,
        stop_acknowledges: bool,
        stop_timeout: Duration,
    ) -> Result<CaptureHealth, MeetingRuntimeError> {
        let active_session = {
            let registry = self.lock_registry()?;
            registry
                .get_active_session()
                .cloned()
                .ok_or(MeetingRuntimeError::NoActiveSession)?
        };
        let controller_config =
            CaptureControllerConfig::from_meeting_config(&active_session.config);
        let health = {
            let mut capture = self.lock_capture()?;
            capture.install_fake_active_capture_for_test(
                controller_config,
                stop_acknowledges,
                stop_timeout,
            )
        };
        {
            let mut registry = self.lock_registry()?;
            registry.update_capture_status(
                health.active_handle_present,
                Some("test fake capture active".to_string()),
            )?;
        }
        Ok(health)
    }
}

#[derive(Debug)]
struct ParsedRecallLlmOutput {
    answer: String,
    status: MeetingRecallStatus,
    evidence: Vec<MeetingRecallEvidence>,
    invalid_evidence_indexes: usize,
    follow_up_questions: Vec<String>,
}

#[derive(Debug, serde::Deserialize)]
struct RawRecallLlmOutput {
    answer: Option<String>,
    status: Option<String>,
    #[serde(default)]
    used_evidence_indexes: Vec<usize>,
    #[serde(default)]
    follow_up_questions: Vec<String>,
}

#[derive(Debug)]
struct RecallLlmParseFailure {
    code: String,
    invalid_evidence_indexes: usize,
}

fn bounded_recall_limit(limit: usize) -> usize {
    if limit == 0 {
        MAX_RECALL_EVIDENCE
    } else {
        limit.min(MAX_RECALL_EVIDENCE)
    }
}

fn detect_recall_query_language(query: &str) -> MeetingLanguage {
    let lower = query.to_lowercase();
    let italian_terms = [
        "cosa",
        "quale",
        "quali",
        "quando",
        "dove",
        "abbiamo",
        "avevamo",
        "decisione",
        "deciso",
        "sessione",
        "riassumimi",
        "parlato",
        "schermo",
        "errore",
        "azioni",
    ];
    if lower.chars().any(|ch| {
        matches!(
            ch,
            '\u{00e0}' | '\u{00e8}' | '\u{00e9}' | '\u{00ec}' | '\u{00f2}' | '\u{00f9}'
        )
    }) || italian_terms.iter().any(|term| lower.contains(term))
    {
        return MeetingLanguage::Italian;
    }
    let english_terms = [
        "what", "when", "where", "which", "decide", "decided", "session", "meeting", "screen",
        "error", "action", "summary",
    ];
    if english_terms.iter().any(|term| lower.contains(term)) {
        MeetingLanguage::English
    } else {
        MeetingLanguage::Unknown
    }
}

fn no_recall_evidence_answer(language: MeetingLanguage) -> String {
    if matches!(language, MeetingLanguage::Italian) {
        "Non ho trovato evidenze negli archivi locali per rispondere a questa domanda.".to_string()
    } else {
        "I could not find evidence in archived sessions to answer this question.".to_string()
    }
}

fn extractive_recall_answer(
    language: MeetingLanguage,
    intent: MeetingRecallIntent,
    evidence: &[MeetingRecallEvidence],
    degraded_from_llm: bool,
) -> String {
    if matches!(intent, MeetingRecallIntent::ScreenContextForTopic) {
        if let Some(answer) = extractive_screen_context_answer(language, evidence) {
            return answer;
        }
    }
    let prefix = if matches!(language, MeetingLanguage::Italian) {
        if degraded_from_llm {
            "Il modello locale non ha prodotto una risposta valida; riporto le evidenze piu rilevanti."
        } else {
            "Ho trovato evidenze rilevanti negli archivi locali."
        }
    } else if degraded_from_llm {
        "The local model did not produce a valid answer, so I am returning the most relevant evidence."
    } else {
        "I found relevant evidence in archived sessions."
    };
    let mut lines = vec![format!("{prefix} ({})", evidence.len())];
    for item in evidence.iter().take(5) {
        lines.push(format!(
            "- {} / {}: {}",
            item.session_title,
            item.matched_kind,
            bounded_recall_text(&item.snippet, 220)
        ));
    }
    bounded_recall_text(&lines.join("\n"), MAX_RECALL_ANSWER_CHARS)
}

fn extractive_screen_context_answer(
    language: MeetingLanguage,
    evidence: &[MeetingRecallEvidence],
) -> Option<String> {
    let screen = evidence
        .iter()
        .find(|item| item.matched_kind == "screen_context" && !item.screen_context_ids.is_empty());
    let transcript = evidence
        .iter()
        .find(|item| item.matched_kind == "transcript" || !item.evidence_segment_ids.is_empty());

    match (screen, transcript) {
        (Some(screen), Some(_)) => {
            let approximate = matches!(
                screen.relation,
                MeetingRecallEvidenceRelation::SameSessionScreenContext
            );
            let temporal = matches!(
                screen.relation,
                MeetingRecallEvidenceRelation::TemporalScreenContext
            );
            let answer = if matches!(language, MeetingLanguage::Italian) {
                if approximate {
                    format!(
                        "Non ho una prova perfettamente sincronizzata al segmento citato, ma nella stessa sessione lo screen context archiviato mostrava: {}",
                        screen.snippet
                    )
                } else if temporal {
                    format!(
                        "Vicino al momento in cui si parlava dell'argomento, lo screen context archiviato mostrava: {}",
                        screen.snippet
                    )
                } else {
                    format!(
                        "Nel contesto schermo collegato al segmento della discussione, lo schermo mostrava: {}",
                        screen.snippet
                    )
                }
            } else if approximate {
                format!(
                    "I do not have perfectly synchronized screen evidence for the exact segment, but the same archived session shows this screen context: {}",
                    screen.snippet
                )
            } else if temporal {
                format!(
                    "Near the time this topic was discussed, the archived screen context showed: {}",
                    screen.snippet
                )
            } else {
                format!(
                    "In the screen context linked to the discussion segment, the screen showed: {}",
                    screen.snippet
                )
            };
            Some(bounded_recall_text(&answer, MAX_RECALL_ANSWER_CHARS))
        }
        (None, Some(_)) => Some(if matches!(language, MeetingLanguage::Italian) {
            "Ho trovato evidenze testuali sull'argomento, ma non ho trovato screen context archiviati per quella sessione.".to_string()
        } else {
            "I found transcript evidence for the topic, but no archived screen context was available for that session.".to_string()
        }),
        _ => None,
    }
}

fn build_recall_llm_prompt_input(
    query: &str,
    language: MeetingLanguage,
    intent: MeetingRecallIntent,
    evidence: &[MeetingRecallEvidence],
) -> MeetingLlmPromptInput {
    let mut included_segment_ids = Vec::new();
    for item in evidence {
        for segment_id in &item.evidence_segment_ids {
            if !included_segment_ids.contains(segment_id) {
                included_segment_ids.push(segment_id.clone());
            }
        }
    }
    let evidence_json = recall_evidence_json(evidence);
    let language_instruction = match language {
        MeetingLanguage::Italian => "Rispondi in italiano.",
        MeetingLanguage::English => "Answer in English.",
        MeetingLanguage::Mixed => "Prefer the user's query language.",
        MeetingLanguage::Unknown => "Use the language of the user's query when possible.",
    };
    let prompt = bounded_recall_text(
        &format!(
            r#"You are Astra's governed local work-session recall synthesizer.

Use only the evidence provided below. Do not invent facts. If the evidence is insufficient, say so.
Do not reveal chain-of-thought. Keep the answer concise and professional.
{language_instruction}
Recall intent: {intent:?}

Evidence relation rules:
- direct_match means the item directly matched the user query.
- linked_screen_context means screen context was explicitly linked to the matching transcript segment.
- temporal_screen_context means screen context was near the matching transcript segment in time.
- same_session_screen_context means screen context came from the same session only; disclose that it is not exact timing.
- Do not say "at that exact moment" unless the screen evidence relation is linked_screen_context or temporal_screen_context.

Return only strict JSON with this schema:
{{
  "answer": "string",
  "status": "answered|partial|insufficient_evidence",
  "used_evidence_indexes": [0],
  "confidence": 0.0,
  "follow_up_questions": ["string"]
}}

User recall query:
{query}

Evidence items, indexed from 0:
{evidence_json}
"#
        ),
        MAX_RECALL_PROMPT_CHARS,
    );
    MeetingLlmPromptInput {
        session_id: "meeting-recall".to_string(),
        prompt,
        segments: Vec::<MeetingLlmPromptSegment>::new(),
        stats: MeetingLlmPromptStats {
            input_segment_count: evidence.len(),
            input_truncated: false,
            input_char_count: evidence_json.chars().count(),
            max_segments: evidence.len(),
            max_chars_total: MAX_RECALL_PROMPT_CHARS,
            max_chars_per_segment: 0,
            included_segment_ids,
            detected_language: language,
            language_confidence: if matches!(language, MeetingLanguage::Unknown) {
                0.0
            } else {
                0.75
            },
            language_source: MeetingLanguageSource::TranscriptHeuristic,
            session_type: MeetingSessionType::General,
            session_type_confidence: 0.0,
            session_type_source: MeetingSessionTypeSource::Unknown,
        },
    }
}

fn recall_evidence_json(evidence: &[MeetingRecallEvidence]) -> String {
    let value = evidence
        .iter()
        .enumerate()
        .map(|(index, item)| {
            serde_json::json!({
                "index": index,
                "session_id": item.session_id,
                "session_title": item.session_title,
                "matched_kind": item.matched_kind,
                "title": item.title,
                "snippet": item.snippet,
                "relation": item.relation,
                "evidence_segment_ids": item.evidence_segment_ids,
                "screen_context_ids": item.screen_context_ids,
                "timestamp_ms": item.timestamp_ms,
                "score": item.score,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string_pretty(&value).unwrap_or_else(|_| "[]".to_string())
}

fn parse_recall_llm_output(
    raw_json: &str,
    evidence: &[MeetingRecallEvidence],
) -> Result<ParsedRecallLlmOutput, RecallLlmParseFailure> {
    let raw: RawRecallLlmOutput = serde_json::from_str(raw_json)
        .map_err(|_| recall_parse_failure("local_llm_recall_json_parse_failed", 0))?;
    let status = match raw.status.as_deref().unwrap_or("partial") {
        "answered" => MeetingRecallStatus::Answered,
        "partial" => MeetingRecallStatus::Partial,
        "insufficient_evidence" => MeetingRecallStatus::InsufficientEvidence,
        _ => return Err(recall_parse_failure("local_llm_recall_status_invalid", 0)),
    };
    let answer = bounded_recall_text(raw.answer.as_deref().unwrap_or(""), MAX_RECALL_ANSWER_CHARS);
    if answer.trim().is_empty() {
        return Err(recall_parse_failure("local_llm_recall_answer_empty", 0));
    }

    let mut invalid_evidence_indexes = 0usize;
    let mut used = Vec::new();
    for index in raw.used_evidence_indexes {
        if let Some(item) = evidence.get(index) {
            used.push(item.clone());
        } else {
            invalid_evidence_indexes = invalid_evidence_indexes.saturating_add(1);
        }
    }

    if !matches!(status, MeetingRecallStatus::InsufficientEvidence) && used.is_empty() {
        return Err(recall_parse_failure(
            "local_llm_recall_missing_valid_evidence",
            invalid_evidence_indexes,
        ));
    }

    Ok(ParsedRecallLlmOutput {
        answer,
        status,
        evidence: used,
        invalid_evidence_indexes,
        follow_up_questions: raw
            .follow_up_questions
            .into_iter()
            .map(|question| bounded_recall_text(&question, 180))
            .filter(|question| !question.trim().is_empty())
            .take(3)
            .collect(),
    })
}

fn recall_parse_failure(code: &str, invalid_evidence_indexes: usize) -> RecallLlmParseFailure {
    RecallLlmParseFailure {
        code: code.to_string(),
        invalid_evidence_indexes,
    }
}

fn bounded_recall_text(text: &str, max_chars: usize) -> String {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.chars().count() <= max_chars {
        compact
    } else {
        compact.chars().take(max_chars).collect()
    }
}

struct CaptureStopForClearOutcome {
    attempted: bool,
    succeeded: bool,
    error_kind: Option<String>,
}

struct ManagedAudioTranscriptionInput<'a> {
    active_session: &'a MeetingSession,
    managed_path: PathBuf,
    audio_file_extension: String,
    file_size_bytes: u64,
    speaker: Option<String>,
    cleanup_requested: bool,
    source_is_captured_segment: bool,
    transcript_source: TranscriptSource,
    audio_backend: Option<String>,
    segment_id: Option<String>,
    start_ms: Option<u64>,
    end_ms: Option<u64>,
}

struct ValidatedAudioFile {
    canonical_path: PathBuf,
    extension: String,
    size: u64,
}

struct ManagedSegmentCleanupOutcome {
    performed: bool,
    error: Option<String>,
}

impl ManagedSegmentCleanupOutcome {
    fn not_requested() -> Self {
        Self {
            performed: false,
            error: None,
        }
    }
}

fn error_with_cleanup_warning(
    error: MeetingRuntimeError,
    cleanup_requested: bool,
    cleanup: ManagedSegmentCleanupOutcome,
) -> MeetingRuntimeError {
    if cleanup_requested && cleanup.error.is_some() {
        MeetingRuntimeError::TranscriptionFailedWithCleanupWarning {
            reason: error.to_string(),
            cleanup_requested,
            cleanup_performed: cleanup.performed,
            cleanup_error: cleanup.error,
            managed_path_redacted: true,
        }
    } else {
        error
    }
}

fn captured_segment_error_kind(error: &MeetingRuntimeError) -> String {
    match error {
        MeetingRuntimeError::ConsentRequired { .. } => "consent_required".to_string(),
        MeetingRuntimeError::ConsentRevoked { .. } => "consent_revoked".to_string(),
        MeetingRuntimeError::TranscriptionUnavailable { reason } => {
            classify_stt_failure_reason(reason).to_string()
        }
        MeetingRuntimeError::SttUnavailable { .. } => "stt_worker_unavailable".to_string(),
        MeetingRuntimeError::TranscriptionInactive => "transcription_inactive".to_string(),
        MeetingRuntimeError::NoAudioFramesReceived { .. } => "no_audio_frames_received".to_string(),
        MeetingRuntimeError::TranscriptionFailedWithCleanupWarning { reason, .. } => {
            classify_stt_failure_reason(reason).to_string()
        }
        MeetingRuntimeError::StorageError { .. } => "storage_error".to_string(),
        MeetingRuntimeError::CaptureStreamError { .. } => "capture_stream_error".to_string(),
        MeetingRuntimeError::AudioCaptureUnavailable { .. } => {
            "audio_capture_unavailable".to_string()
        }
        MeetingRuntimeError::SegmentWriteFailed { .. } => "segment_write_failed".to_string(),
        MeetingRuntimeError::SegmentTooLarge { .. } => "segment_too_large".to_string(),
        MeetingRuntimeError::InvalidConfig { .. } => "invalid_segment".to_string(),
        MeetingRuntimeError::NoActiveSession => "no_active_session".to_string(),
        MeetingRuntimeError::SessionPaused { .. } => "session_paused".to_string(),
        MeetingRuntimeError::SessionCompleted => "session_completed".to_string(),
        MeetingRuntimeError::InvalidLifecycleTransition { .. } => "invalid_lifecycle".to_string(),
        MeetingRuntimeError::ClearAbortedCaptureStopFailed { .. } => "clear_aborted".to_string(),
        MeetingRuntimeError::CaptureStopTimedOut { .. } => "capture_stop_timed_out".to_string(),
        MeetingRuntimeError::CaptureUnavailable { .. } => "capture_unavailable".to_string(),
        MeetingRuntimeError::CaptureStartFailed { .. } => "capture_start_failed".to_string(),
        MeetingRuntimeError::CaptureStartupTimeout { .. } => "capture_startup_timeout".to_string(),
        MeetingRuntimeError::CaptureStartupChannelClosed { .. } => {
            "capture_startup_closed".to_string()
        }
        MeetingRuntimeError::CaptureDeviceUnavailable { .. } => {
            "capture_device_unavailable".to_string()
        }
        MeetingRuntimeError::PermissionDenied { .. } => "permission_denied".to_string(),
        MeetingRuntimeError::UnsupportedCapability { .. } => "unsupported_capability".to_string(),
        MeetingRuntimeError::ActiveSessionExists { .. } => "active_session_exists".to_string(),
        MeetingRuntimeError::ConfirmationRequired { .. } => "confirmation_required".to_string(),
        MeetingRuntimeError::SerializationError { .. } => "serialization_error".to_string(),
        MeetingRuntimeError::MutexPoisoned { .. } => "mutex_poisoned".to_string(),
    }
}

fn classify_stt_failure_reason(reason: &str) -> &'static str {
    let normalized = reason.to_ascii_lowercase();
    if normalized.contains("timed out") || normalized.contains("timeout") {
        "stt_timeout"
    } else if normalized.contains("unavailable") {
        "stt_worker_unavailable"
    } else if normalized.contains("not configured") || normalized.contains("config") {
        "stt_worker_unavailable"
    } else if normalized.contains("empty transcript") {
        "stt_empty_transcript"
    } else if normalized.contains("invalid") || normalized.contains("audio") {
        "stt_invalid_audio"
    } else if normalized.contains("device")
        || normalized.contains("cuda")
        || normalized.contains("cpu")
    {
        "stt_device_error"
    } else if normalized.contains("worker failed")
        || normalized.contains("protocol")
        || normalized.contains("i/o")
    {
        "stt_worker_failed"
    } else {
        "stt_unknown"
    }
}

fn captured_segment_error_fails_capture_immediately(error: &MeetingRuntimeError) -> bool {
    matches!(
        error,
        MeetingRuntimeError::StorageError { .. }
            | MeetingRuntimeError::CaptureStreamError { .. }
            | MeetingRuntimeError::SegmentWriteFailed { .. }
            | MeetingRuntimeError::SegmentTooLarge { .. }
    )
}

fn capture_stop_error_kind(error: &MeetingRuntimeError) -> &'static str {
    match error {
        MeetingRuntimeError::CaptureStopTimedOut { .. } => "capture_stop_timed_out",
        MeetingRuntimeError::CaptureStreamError { .. } => "capture_stream_error",
        MeetingRuntimeError::CaptureUnavailable { .. } => "capture_unavailable",
        MeetingRuntimeError::CaptureStartFailed { .. } => "capture_start_failed",
        MeetingRuntimeError::CaptureDeviceUnavailable { .. } => "capture_device_unavailable",
        MeetingRuntimeError::MutexPoisoned { .. } => "mutex_poisoned",
        _ => "capture_stop_failed",
    }
}

fn validate_meeting_config(
    platform_arg: &str,
    mut config: MeetingConfig,
) -> Result<(String, MeetingConfig), MeetingRuntimeError> {
    let platform = normalize_meeting_app_name(platform_arg);
    let config_platform = normalize_meeting_app_name(&config.platform);
    if platform.is_empty() || config_platform.is_empty() {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "platform and config.platform are required".to_string(),
        });
    }
    if platform != config_platform {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: format!(
                "platform argument ({platform}) does not match config.platform ({config_platform})"
            ),
        });
    }

    if config.session_mode == MeetingSessionMode::Manual && config.live_transcription_enabled {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "manual meeting sessions cannot enable live transcription".to_string(),
        });
    }
    if config.session_mode == MeetingSessionMode::Manual {
        config.capture_options = super::types::MeetingCaptureOptions::manual();
    } else {
        if config.live_transcription_enabled {
            config.capture_options.segment_transcription = true;
        }
        if !config.capture_options.any_audio_enabled() {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "real capture sessions require system_audio or microphone capture"
                    .to_string(),
            });
        }
        if config.capture_options.microphone && config.capture_backend != CaptureBackend::Wasapi {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "microphone capture is currently implemented only through Windows WASAPI"
                    .to_string(),
            });
        }
    }

    if !matches!(config.sample_rate, 16_000 | 44_100 | 48_000) {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: format!(
                "unsupported sample_rate {}; allowed values are 16000, 44100, 48000",
                config.sample_rate
            ),
        });
    }

    let transcription_model = config.transcription_model.trim().to_ascii_lowercase();
    if transcription_model.is_empty() {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "transcription_model is required".to_string(),
        });
    }
    if transcription_model != "local" {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "only local meeting transcription_model is supported".to_string(),
        });
    }

    let privacy_mode = config.privacy_mode.trim().to_ascii_lowercase();
    if privacy_mode.is_empty() {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "privacy_mode is required".to_string(),
        });
    }
    if !matches!(
        privacy_mode.as_str(),
        "default" | "redact" | "pause" | "disabled"
    ) {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: format!("unsupported privacy_mode {privacy_mode}"),
        });
    }

    config.platform = config_platform;
    config.transcription_model = transcription_model;
    config.privacy_mode = privacy_mode;
    Ok((platform, config))
}

fn validate_audio_file_path(audio_path: &str) -> Result<ValidatedAudioFile, MeetingRuntimeError> {
    let trimmed = audio_path.trim();
    if trimmed.is_empty() {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "audio_path is required".to_string(),
        });
    }
    let canonical_path =
        PathBuf::from(trimmed)
            .canonicalize()
            .map_err(|_| MeetingRuntimeError::InvalidConfig {
                message: "audio file does not exist or cannot be resolved".to_string(),
            })?;
    let metadata = canonical_path
        .metadata()
        .map_err(|_| MeetingRuntimeError::InvalidConfig {
            message: "audio file metadata cannot be read".to_string(),
        })?;
    if !metadata.is_file() {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "audio_path must point to a file".to_string(),
        });
    }
    if is_system_path(&canonical_path) {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "audio_path points to a protected system location".to_string(),
        });
    }

    let extension = canonical_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if !SUPPORTED_MEETING_AUDIO_EXTENSIONS.contains(&extension.as_str()) {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: format!(
                "unsupported audio extension {}; supported extensions: {}",
                if extension.is_empty() {
                    "none"
                } else {
                    extension.as_str()
                },
                SUPPORTED_MEETING_AUDIO_EXTENSIONS.join(", ")
            ),
        });
    }

    let size = metadata.len();
    if size > MAX_MEETING_TRANSCRIPTION_AUDIO_BYTES {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: format!(
                "audio file is too large; maximum is {} bytes",
                MAX_MEETING_TRANSCRIPTION_AUDIO_BYTES
            ),
        });
    }
    validate_wav_header(&canonical_path, size)?;

    Ok(ValidatedAudioFile {
        canonical_path,
        extension,
        size,
    })
}

fn validate_wav_header(path: &Path, size: u64) -> Result<(), MeetingRuntimeError> {
    if size < 12 {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "audio file is too small to contain a valid WAV header".to_string(),
        });
    }

    let mut header = [0_u8; 12];
    let mut file = File::open(path).map_err(|_| MeetingRuntimeError::InvalidConfig {
        message: "audio file header cannot be read".to_string(),
    })?;
    file.read_exact(&mut header)
        .map_err(|_| MeetingRuntimeError::InvalidConfig {
            message: "audio file header cannot be read".to_string(),
        })?;

    if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
        return Err(MeetingRuntimeError::InvalidConfig {
            message: "audio file is not a valid WAV file".to_string(),
        });
    }

    Ok(())
}

fn is_system_path(path: &Path) -> bool {
    #[cfg(target_os = "windows")]
    {
        ["WINDIR", "ProgramFiles", "ProgramFiles(x86)"]
            .iter()
            .filter_map(|name| std::env::var_os(name).map(PathBuf::from))
            .filter_map(|value| value.canonicalize().ok())
            .any(|system_root| path.starts_with(system_root))
    }

    #[cfg(not(target_os = "windows"))]
    {
        path.starts_with("/bin")
            || path.starts_with("/sbin")
            || path.starts_with("/usr/bin")
            || path.starts_with("/usr/sbin")
    }
}

fn should_retry_meeting_output_language(result: &MeetingIntelligenceResult) -> bool {
    result.diagnostics.llm_used
        && !result.diagnostics.fallback_used
        && result.diagnostics.output_language_mismatch
        && matches!(
            result.diagnostics.detected_language,
            MeetingLanguage::Italian | MeetingLanguage::English
        )
        && !result.diagnostics.language_retry_attempted
}

fn readiness(
    capability: &str,
    available: bool,
    state: MeetingCapabilityState,
    reason: Option<String>,
) -> MeetingCapabilityReadiness {
    MeetingCapabilityReadiness {
        capability: capability.to_string(),
        available,
        state,
        reason,
    }
}

fn aggregate_capture_health(system: CaptureHealth, microphone: CaptureHealth) -> CaptureHealth {
    let mut aggregate = system.clone();
    aggregate.active_handle_present =
        system.active_handle_present || microphone.active_handle_present;
    aggregate.backpressure_active = system.backpressure_active || microphone.backpressure_active;
    aggregate.state = if matches!(system.state, CaptureControllerState::Failed)
        || matches!(microphone.state, CaptureControllerState::Failed)
    {
        CaptureControllerState::Failed
    } else if matches!(system.state, CaptureControllerState::Capturing)
        || matches!(microphone.state, CaptureControllerState::Capturing)
    {
        CaptureControllerState::Capturing
    } else if matches!(system.state, CaptureControllerState::Paused)
        || matches!(microphone.state, CaptureControllerState::Paused)
    {
        CaptureControllerState::Paused
    } else if matches!(system.state, CaptureControllerState::Starting)
        || matches!(microphone.state, CaptureControllerState::Starting)
    {
        CaptureControllerState::Starting
    } else {
        system.state
    };
    aggregate.metrics.segments_written = system
        .metrics
        .segments_written
        .saturating_add(microphone.metrics.segments_written);
    aggregate.metrics.segments_queued = system
        .metrics
        .segments_queued
        .saturating_add(microphone.metrics.segments_queued);
    aggregate.metrics.segments_queued_total = system
        .metrics
        .segments_queued_total
        .saturating_add(microphone.metrics.segments_queued_total);
    aggregate.metrics.current_queue_depth = system
        .metrics
        .current_queue_depth
        .saturating_add(microphone.metrics.current_queue_depth);
    aggregate.metrics.segments_dequeued_total = system
        .metrics
        .segments_dequeued_total
        .saturating_add(microphone.metrics.segments_dequeued_total);
    aggregate.metrics.segments_in_flight = system
        .metrics
        .segments_in_flight
        .saturating_add(microphone.metrics.segments_in_flight);
    aggregate.metrics.segments_transcribed = system
        .metrics
        .segments_transcribed
        .saturating_add(microphone.metrics.segments_transcribed);
    aggregate.metrics.segments_failed = system
        .metrics
        .segments_failed
        .saturating_add(microphone.metrics.segments_failed);
    aggregate.metrics.segment_transcription_timeouts = system
        .metrics
        .segment_transcription_timeouts
        .saturating_add(microphone.metrics.segment_transcription_timeouts);
    aggregate.metrics.segments_dropped = system
        .metrics
        .segments_dropped
        .saturating_add(microphone.metrics.segments_dropped);
    aggregate.metrics.dropped_silence_segments = system
        .metrics
        .dropped_silence_segments
        .saturating_add(microphone.metrics.dropped_silence_segments);
    aggregate.metrics.segment_transcription_failures_total = system
        .metrics
        .segment_transcription_failures_total
        .saturating_add(microphone.metrics.segment_transcription_failures_total);
    aggregate.metrics.frames_captured = system
        .metrics
        .frames_captured
        .saturating_add(microphone.metrics.frames_captured);
    aggregate.metrics.frames_converted = system
        .metrics
        .frames_converted
        .saturating_add(microphone.metrics.frames_converted);
    aggregate.metrics.drain_timeout =
        system.metrics.drain_timeout || microphone.metrics.drain_timeout;
    aggregate.metrics.segment_transcription_drain_status =
        aggregate_drain_status(&system.metrics, &microphone.metrics);
    aggregate.metrics.last_segment_transcription_error_kind = microphone
        .metrics
        .last_segment_transcription_error_kind
        .clone()
        .or(system.metrics.last_segment_transcription_error_kind.clone());
    aggregate
}

fn aggregate_drain_status(
    system: &super::types::CaptureMetrics,
    microphone: &super::types::CaptureMetrics,
) -> Option<String> {
    if system.drain_timeout || microphone.drain_timeout {
        Some("timed_out".to_string())
    } else if matches!(
        (
            system.segment_transcription_drain_status.as_deref(),
            microphone.segment_transcription_drain_status.as_deref(),
        ),
        (Some("running"), _) | (_, Some("running"))
    ) {
        Some("running".to_string())
    } else if matches!(
        system.segment_transcription_drain_status.as_deref(),
        Some("completed")
    ) || matches!(
        microphone.segment_transcription_drain_status.as_deref(),
        Some("completed")
    ) {
        Some("completed".to_string())
    } else {
        system
            .segment_transcription_drain_status
            .clone()
            .or_else(|| microphone.segment_transcription_drain_status.clone())
    }
}

fn apply_segment_stt_export_metadata(
    exported: &mut ExportedMeeting,
    system_health: &CaptureHealth,
    microphone_health: &CaptureHealth,
) {
    let aggregate = aggregate_capture_health(system_health.clone(), microphone_health.clone());
    let metrics = aggregate.metrics;
    let completeness = derive_meeting_stt_completeness(system_health, microphone_health);
    let incomplete = completeness.overall.is_incomplete();
    if let Some(metadata) = exported.metadata.as_object_mut() {
        metadata.insert(
            "meeting_segment_transcription_incomplete".to_string(),
            serde_json::json!(incomplete),
        );
        metadata.insert(
            "stt_completeness".to_string(),
            serde_json::to_value(&completeness).unwrap_or_else(|_| {
                serde_json::json!({
                    "overall": completeness.overall.as_str(),
                    "segments_written": metrics.segments_written,
                    "segments_transcribed": metrics.segments_transcribed,
                    "segments_in_flight": metrics.segments_in_flight,
                    "timeouts": metrics.segment_transcription_timeouts,
                })
            }),
        );
        metadata.insert(
            "segment_stt_diagnostics".to_string(),
            serde_json::json!({
                "segments_written": metrics.segments_written,
                "segments_queued_total": metrics.segments_queued_total,
                "current_queue_depth": metrics.current_queue_depth,
                "segments_dequeued_total": metrics.segments_dequeued_total,
                "segments_in_flight": metrics.segments_in_flight,
                "segments_transcribed": metrics.segments_transcribed,
                "segments_failed": metrics.segments_failed,
                "segment_transcription_timeouts": metrics.segment_transcription_timeouts,
                "drain_status": metrics.segment_transcription_drain_status,
                "drain_timeout": metrics.drain_timeout,
                "last_stt_error": metrics.last_segment_transcription_error_kind,
                "last_transcription_started_segment_id": metrics.last_transcription_started_segment_id,
                "last_transcription_completed_segment_id": metrics.last_transcription_completed_segment_id,
                "last_transcription_failed_segment_id": metrics.last_transcription_failed_segment_id,
            }),
        );
    }
}

fn capture_status_message(
    started_sources: &[(TranscriptSource, CaptureHealth)],
    segment_transcription: bool,
) -> String {
    let sources = started_sources
        .iter()
        .map(|(source, health)| {
            format!(
                "{}:{:?}:segments_written={}",
                source.as_str(),
                health.state,
                health.metrics.segments_written
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "capture active sources: {sources}; segment_file_stt={}",
        if segment_transcription {
            "enabled"
        } else {
            "disabled"
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meeting::types::{
        derive_meeting_stt_source_completeness, ActionItemStatus, ArtifactGenerator,
        CaptureMetrics, ClearMeetingDataRequest, MeetingClearScope,
        MeetingIntelligenceGenerationOptions, MeetingIntelligenceStatus,
        MeetingRecallEvidenceRelation, MeetingRecallIntent, MeetingRecallRequest,
        MeetingRecallStatus, MeetingScreenContext, MeetingSessionExportFormat,
        MeetingSessionExportRequest, MeetingSessionListRequest, MeetingSessionReadRequest,
        MeetingSessionSearchRequest, MeetingStatus, MeetingSttCompletenessStatus,
        ScreenContextAttachmentMode, ScreenContextRedaction, ScreenContextSource, TranscriptSource,
        CLEAR_MEETING_DATA_CONFIRMATION_PHRASE, LOCAL_USER_SPEAKER_ID, REMOTE_SPEAKER_1_ID,
    };
    use crate::meeting::{
        intelligence_engine::{
            MeetingIntelligenceLlm, MeetingLlmError, MeetingLlmErrorKind, MeetingLlmFuture,
            MeetingLlmPromptInput, MeetingLlmRawOutput,
        },
        segment_writer::{SegmentWriter, SegmentWriterConfig},
        stt_adapter::{MeetingFileTranscriber, MeetingFileTranscriptionFuture},
    };
    use chrono::Utc;
    use std::{
        collections::VecDeque,
        path::Path,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        },
        time::Duration,
    };
    use uuid::Uuid;

    struct FixedTranscriber;

    impl MeetingFileTranscriber for FixedTranscriber {
        fn status(&self) -> super::super::types::MeetingSttAdapterStatus {
            super::super::types::MeetingSttAdapterStatus {
                state: MeetingCapabilityState::Ready,
                existing_boundary: "test".to_string(),
                file_transcription: readiness(
                    "meeting.transcription.file",
                    true,
                    MeetingCapabilityState::Ready,
                    None,
                ),
                live_transcription: readiness(
                    "meeting.transcription.live",
                    false,
                    MeetingCapabilityState::Unavailable,
                    Some("streaming STT unsupported in test adapter".to_string()),
                ),
                chunk_streaming: readiness(
                    "meeting.transcription.chunk_streaming",
                    false,
                    MeetingCapabilityState::Unavailable,
                    Some("chunk streaming unsupported in test adapter".to_string()),
                ),
                chunk_streaming_supported: false,
                emits_placeholder_transcripts: false,
                reason: None,
            }
        }

        fn transcribe_file<'a>(
            &'a self,
            audio_path: &'a Path,
        ) -> MeetingFileTranscriptionFuture<'a> {
            let stem = audio_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("segment")
                .to_string();
            Box::pin(async move {
                Ok(format!(
                    "We decided to ship this segment. Please follow up on {stem} by tomorrow."
                ))
            })
        }
    }

    struct FailingTranscriber {
        reason: &'static str,
    }

    impl MeetingFileTranscriber for FailingTranscriber {
        fn status(&self) -> super::super::types::MeetingSttAdapterStatus {
            FixedTranscriber.status()
        }

        fn transcribe_file<'a>(
            &'a self,
            _audio_path: &'a Path,
        ) -> MeetingFileTranscriptionFuture<'a> {
            Box::pin(async move {
                Err(MeetingRuntimeError::TranscriptionUnavailable {
                    reason: self.reason.to_string(),
                })
            })
        }
    }

    #[derive(Clone)]
    struct FixedMeetingLlm {
        response: Result<String, MeetingLlmErrorKind>,
        calls: Arc<AtomicUsize>,
        delay: Duration,
    }

    impl FixedMeetingLlm {
        fn json(raw: impl Into<String>) -> Self {
            Self {
                response: Ok(raw.into()),
                calls: Arc::new(AtomicUsize::new(0)),
                delay: Duration::from_millis(0),
            }
        }

        fn error(kind: MeetingLlmErrorKind) -> Self {
            Self {
                response: Err(kind),
                calls: Arc::new(AtomicUsize::new(0)),
                delay: Duration::from_millis(0),
            }
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = delay;
            self
        }
    }

    impl MeetingIntelligenceLlm for FixedMeetingLlm {
        fn generate_intelligence_json<'a>(
            &'a self,
            input: MeetingLlmPromptInput,
        ) -> MeetingLlmFuture<'a> {
            let response = self.response.clone();
            let calls = self.calls.clone();
            let delay = self.delay;
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                match response {
                    Ok(raw_json) => Ok(MeetingLlmRawOutput {
                        raw_json,
                        provider: "ollama".to_string(),
                        model: "mock-meeting-model".to_string(),
                        stats: input.stats,
                        endpoint: Some("127.0.0.1:11434".to_string()),
                        llm_generation_duration_ms: Some(1),
                    }),
                    Err(kind) => Err(MeetingLlmError {
                        kind,
                        message: "mock local model unavailable".to_string(),
                        provider: "ollama".to_string(),
                        model: Some("mock-meeting-model".to_string()),
                        stats: input.stats,
                        endpoint: Some("127.0.0.1:11434".to_string()),
                        llm_generation_duration_ms: Some(1),
                    }),
                }
            })
        }
    }

    #[derive(Clone)]
    struct SequencedMeetingLlm {
        responses: Arc<Mutex<VecDeque<String>>>,
        calls: Arc<AtomicUsize>,
    }

    impl SequencedMeetingLlm {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(
                    responses
                        .into_iter()
                        .map(ToString::to_string)
                        .collect::<VecDeque<_>>(),
                )),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl MeetingIntelligenceLlm for SequencedMeetingLlm {
        fn generate_intelligence_json<'a>(
            &'a self,
            input: MeetingLlmPromptInput,
        ) -> MeetingLlmFuture<'a> {
            let calls = self.calls.clone();
            let responses = self.responses.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                let raw_json = responses
                    .lock()
                    .expect("responses mutex")
                    .pop_front()
                    .unwrap_or_else(|| "{}".to_string());
                Ok(MeetingLlmRawOutput {
                    raw_json,
                    provider: "ollama".to_string(),
                    model: "mock-meeting-model".to_string(),
                    stats: input.stats,
                    endpoint: Some("127.0.0.1:11434".to_string()),
                    llm_generation_duration_ms: Some(1),
                })
            })
        }
    }

    #[derive(Clone)]
    struct CapturingMeetingLlm {
        response: String,
        prompts: Arc<Mutex<Vec<String>>>,
    }

    impl CapturingMeetingLlm {
        fn new(response: impl Into<String>) -> Self {
            Self {
                response: response.into(),
                prompts: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl MeetingIntelligenceLlm for CapturingMeetingLlm {
        fn generate_intelligence_json<'a>(
            &'a self,
            input: MeetingLlmPromptInput,
        ) -> MeetingLlmFuture<'a> {
            let response = self.response.clone();
            let prompts = self.prompts.clone();
            Box::pin(async move {
                prompts
                    .lock()
                    .expect("prompts mutex")
                    .push(input.prompt.clone());
                Ok(MeetingLlmRawOutput {
                    raw_json: response,
                    provider: "ollama".to_string(),
                    model: "mock-meeting-model".to_string(),
                    stats: input.stats,
                    endpoint: Some("127.0.0.1:11434".to_string()),
                    llm_generation_duration_ms: Some(1),
                })
            })
        }
    }

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!("astra_meeting_runtime_{}", Uuid::new_v4()))
    }

    fn config() -> MeetingConfig {
        MeetingConfig {
            platform: "teams".to_string(),
            capture_backend: CaptureBackend::CoreAudio,
            transcription_model: "local".to_string(),
            sample_rate: 16_000,
            diarization_enabled: false,
            privacy_mode: "default".to_string(),
            session_mode: MeetingSessionMode::RealCapture,
            live_transcription_enabled: false,
            capture_options: super::super::types::MeetingCaptureOptions::default(),
        }
    }

    fn transcript() -> TranscriptEntry {
        TranscriptEntry::sourced("", TranscriptSource::Manual, "speaker", "hello", 0.95)
    }

    fn screen_context(
        session_id: &str,
        captured_at: chrono::DateTime<Utc>,
    ) -> MeetingScreenContext {
        screen_context_with_summary(
            session_id,
            captured_at,
            "Visual Studio Code shows a GPU memory warning in the TTS module",
        )
    }

    fn screen_context_with_summary(
        session_id: &str,
        captured_at: chrono::DateTime<Utc>,
        summary: &str,
    ) -> MeetingScreenContext {
        MeetingScreenContext {
            context_id: "screen-context-test".to_string(),
            session_id: session_id.to_string(),
            captured_at,
            source: ScreenContextSource::ManualCapture,
            attachment_mode: ScreenContextAttachmentMode::CurrentMoment,
            linked_transcript_segment_ids: Vec::new(),
            linked_time_window: None,
            summary: summary.to_string(),
            structured_observation: None,
            screenshot_ref: None,
            redaction: ScreenContextRedaction::ScreenshotNotStored,
            confidence: 0.7,
            diagnostics: Vec::new(),
        }
    }

    fn transcript_at(text: &str, timestamp: chrono::DateTime<Utc>) -> TranscriptEntry {
        let mut entry = TranscriptEntry::sourced("", TranscriptSource::Manual, "Simone", text, 0.9);
        entry.timestamp = timestamp;
        entry.created_at = timestamp;
        entry
    }

    fn recall_request(query: &str, use_local_llm: bool) -> MeetingRecallRequest {
        MeetingRecallRequest {
            query: query.to_string(),
            limit: 12,
            date_from: None,
            date_to: None,
            include_transcript: true,
            include_intelligence: true,
            include_screen_context: true,
            use_local_llm,
        }
    }

    fn archive_recall_fixture(runtime: &MeetingRuntime) -> String {
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");

        runtime
            .add_transcript(TranscriptEntry::sourced(
                "",
                TranscriptSource::Manual,
                "Simone",
                "We discussed STT drain and decided to show incomplete export warnings when one segment remains in flight.",
                0.91,
            ))
            .expect("add transcript");
        let segment_id = runtime
            .get_active_state()
            .expect("active state")
            .transcript
            .first()
            .expect("transcript")
            .segment_id
            .clone();
        runtime
            .add_decision(DecisionLogEntry {
                id: "decision-recall-test".to_string(),
                session_id: session.session_id.clone(),
                timestamp: Utc::now(),
                created_at: Utc::now(),
                decision: "Ship the STT drain export warning".to_string(),
                rationale:
                    "The archive must not claim STT is complete when a segment is still in flight"
                        .to_string(),
                made_by: None,
                evidence_segment_ids: vec![segment_id],
            })
            .expect("add decision");
        runtime
            .attach_screen_context(screen_context(&session.session_id, Utc::now()))
            .expect("attach screen context");
        runtime.stop_session().expect("stop session");
        session.session_id
    }

    fn archive_bumper_screen_fixture(
        runtime: &MeetingRuntime,
        filler_offsets_minutes: &[i64],
        screen_offset_minutes: Option<i64>,
    ) -> String {
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let base = Utc::now();
        runtime
            .add_transcript(transcript_at(
                "Il bumper del fronte dell auto era il problema principale.",
                base,
            ))
            .expect("add bumper transcript");
        for (index, offset) in filler_offsets_minutes.iter().enumerate() {
            runtime
                .add_transcript(transcript_at(
                    &format!("Discussione intermedia su attrezzi da officina {index}."),
                    base + chrono::Duration::minutes(*offset),
                ))
                .expect("add filler transcript");
        }
        if let Some(offset) = screen_offset_minutes {
            runtime
                .attach_screen_context(screen_context_with_summary(
                    &session.session_id,
                    base + chrono::Duration::minutes(offset),
                    "YouTube video showing a person working on a Bugatti Veyron in a garage",
                ))
                .expect("attach screen context");
        }
        runtime.stop_session().expect("stop session");
        session.session_id
    }

    #[test]
    fn privacy_state_denies_recording_by_default() {
        let runtime = MeetingRuntime::new(temp_root());
        let state = runtime.consent_state().expect("consent state");
        assert!(!state.given);
        assert!(!state.global_enabled);
    }

    #[test]
    fn start_meeting_requires_explicit_consent() {
        let runtime = MeetingRuntime::new(temp_root());
        let result = runtime.start_session("teams".to_string(), config());
        assert!(matches!(
            result,
            Err(MeetingRuntimeError::ConsentRequired { .. })
        ));
    }

    #[test]
    fn unsupported_audio_capture_prevents_session_start_after_consent() {
        let runtime = MeetingRuntime::new(temp_root());
        runtime.grant_consent("teams").expect("grant consent");
        let result = runtime.start_session("teams".to_string(), config());
        assert!(matches!(
            result,
            Err(MeetingRuntimeError::CaptureUnavailable { .. })
        ));
        assert!(runtime
            .get_active_session()
            .expect("active session")
            .is_none());
    }

    #[test]
    fn cannot_add_transcript_without_active_session() {
        let runtime = MeetingRuntime::new(temp_root());
        let result = runtime.add_transcript(transcript());
        assert!(matches!(result, Err(MeetingRuntimeError::NoActiveSession)));
    }

    #[test]
    fn stt_completeness_derives_complete_from_finished_metrics() {
        let mut metrics = CaptureMetrics {
            segments_written: 3,
            segments_transcribed: 3,
            segment_transcription_drain_status: Some("completed".to_string()),
            ..CaptureMetrics::default()
        };
        assert_eq!(
            derive_meeting_stt_source_completeness(&metrics).status,
            MeetingSttCompletenessStatus::Complete
        );

        metrics.segments_written = 0;
        metrics.segments_transcribed = 0;
        metrics.dropped_silence_segments = 4;
        assert_eq!(
            derive_meeting_stt_source_completeness(&metrics).status,
            MeetingSttCompletenessStatus::CompleteNoSpeech
        );
    }

    #[test]
    fn stt_completeness_derives_incomplete_drain_timeout_pending_and_failed() {
        let drain_timeout = CaptureMetrics {
            segments_written: 19,
            segments_transcribed: 18,
            segments_in_flight: 1,
            drain_timeout: true,
            segment_transcription_drain_status: Some("timed_out".to_string()),
            ..CaptureMetrics::default()
        };
        assert_eq!(
            derive_meeting_stt_source_completeness(&drain_timeout).status,
            MeetingSttCompletenessStatus::IncompleteDrainTimeout
        );

        let pending = CaptureMetrics {
            current_queue_depth: 1,
            ..CaptureMetrics::default()
        };
        assert_eq!(
            derive_meeting_stt_source_completeness(&pending).status,
            MeetingSttCompletenessStatus::IncompletePendingQueue
        );

        let failed = CaptureMetrics {
            segments_written: 2,
            segments_transcribed: 1,
            segments_failed: 1,
            ..CaptureMetrics::default()
        };
        assert_eq!(
            derive_meeting_stt_source_completeness(&failed).status,
            MeetingSttCompletenessStatus::IncompleteFailedSegments
        );
    }

    #[test]
    fn export_metadata_and_markdown_mark_partial_drain_timeout_incomplete() {
        let mut exported = ExportedMeeting {
            session_id: "session-partial-drain".to_string(),
            platform: "teams".to_string(),
            started_at: Utc::now(),
            ended_at: Utc::now(),
            participants: Vec::new(),
            transcript: vec![transcript()],
            summary: Vec::new(),
            action_items: Vec::new(),
            decisions: Vec::new(),
            notes: Vec::new(),
            intelligence: None,
            screen_contexts: Vec::new(),
            metadata: serde_json::json!({}),
        };
        let mut system_health = CaptureHealth::default();
        system_health.metrics.segments_written = 19;
        system_health.metrics.segments_queued_total = 19;
        system_health.metrics.segments_dequeued_total = 19;
        system_health.metrics.segments_transcribed = 18;
        system_health.metrics.segments_in_flight = 1;
        system_health.metrics.segment_transcription_timeouts = 1;
        system_health.metrics.drain_timeout = true;
        system_health.metrics.segment_transcription_drain_status = Some("timed_out".to_string());

        let mut microphone_health = CaptureHealth::default();
        microphone_health.metrics.dropped_silence_segments = 3;
        microphone_health.metrics.segment_transcription_drain_status =
            Some("completed".to_string());

        apply_segment_stt_export_metadata(&mut exported, &system_health, &microphone_health);

        assert_eq!(
            exported
                .metadata
                .get("meeting_segment_transcription_incomplete")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            exported
                .metadata
                .get("stt_completeness")
                .and_then(|value| value.get("overall"))
                .and_then(|value| value.as_str()),
            Some("incomplete_drain_timeout")
        );
        assert_eq!(
            exported
                .metadata
                .get("stt_completeness")
                .and_then(|value| value.get("microphone"))
                .and_then(|value| value.get("status"))
                .and_then(|value| value.as_str()),
            Some("complete_no_speech")
        );

        let markdown = NoteOrganizer::new(temp_root())
            .to_markdown(&exported)
            .expect("markdown export");
        assert!(markdown.contains("STT completeness: incomplete (drain timed out)"));
        assert!(markdown.contains("System audio STT: incomplete (drain timed out)"));
        assert!(markdown.contains("18/19 transcribed"));
        assert!(markdown.contains("1 in-flight"));
        assert!(markdown.contains("1 timeout"));
        assert!(markdown.contains("Microphone STT: complete/no speech"));
    }

    #[test]
    fn manual_transcript_is_marked_manual_and_derives_artifacts() {
        let runtime = MeetingRuntime::new(temp_root());
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");

        let mut entry = TranscriptEntry::sourced(
            "",
            TranscriptSource::Unknown,
            "Simone",
            "We decided to ship the scoped milestone because it is ready. Please follow up by tomorrow.",
            0.8,
        );
        entry.stt_model = Some("should_be_cleared_for_manual".to_string());
        entry.audio_backend = Some("should_be_cleared_for_manual".to_string());

        runtime.add_transcript(entry).expect("add transcript");

        let state = runtime.get_active_state().expect("read active state");
        let transcript = state.transcript.first().expect("transcript entry");

        assert_eq!(transcript.source, TranscriptSource::Manual);
        assert_eq!(transcript.speaker_id.as_deref(), Some("manual_simone"));
        assert_eq!(transcript.speaker_label.as_deref(), Some("Simone"));
        assert_eq!(
            transcript.speaker_attribution_method,
            SpeakerAttributionMethod::UserAssigned
        );
        assert!(transcript.stt_model.is_none());
        assert!(transcript.audio_backend.is_none());
        assert!(state
            .notes
            .iter()
            .any(|note| note.evidence_segment_ids.contains(&transcript.segment_id)));
        assert!(state
            .action_items
            .iter()
            .any(|item| item.evidence_segment_ids.contains(&transcript.segment_id)));
        assert!(state.decisions.iter().any(|decision| decision
            .evidence_segment_ids
            .contains(&transcript.segment_id)));
    }

    #[tokio::test]
    async fn dual_source_segment_stt_is_source_tagged_and_timeline_ordered() {
        let root = temp_root();
        let runtime =
            MeetingRuntime::with_file_transcriber(root.clone(), Arc::new(FixedTranscriber));
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");

        let storage_dir = root.join(".astra").join("meetings");
        let samples = vec![100_i16; 16_000];
        let system_writer = SegmentWriter::new(
            storage_dir.clone(),
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::SystemAudio,
                ..SegmentWriterConfig::default()
            },
        );
        let mic_writer = SegmentWriter::new(
            storage_dir,
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::Microphone,
                ..SegmentWriterConfig::default()
            },
        );

        let mut mic_segment = mic_writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("mic segment");
        mic_segment.sequence_number = 2;
        mic_segment.start_ms = Some(2_000);
        mic_segment.end_ms = Some(3_000);

        let mut system_segment = system_writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("system segment");
        system_segment.sequence_number = 1;
        system_segment.start_ms = Some(0);
        system_segment.end_ms = Some(1_000);

        runtime
            .transcribe_captured_segment(mic_segment, Some("mic".to_string()), false)
            .await
            .expect("mic transcription");
        runtime
            .transcribe_captured_segment(system_segment, Some("system".to_string()), false)
            .await
            .expect("system transcription");

        let state = runtime.get_active_state().expect("active state");
        assert_eq!(state.transcript.len(), 2);
        assert_eq!(state.transcript[0].source, TranscriptSource::SystemAudio);
        assert_eq!(state.transcript[0].start_ms, Some(0));
        assert_eq!(
            state.transcript[0].speaker_id.as_deref(),
            Some(REMOTE_SPEAKER_1_ID)
        );
        assert_eq!(
            state.transcript[0].speaker_label.as_deref(),
            Some("Speaker 1")
        );
        assert_eq!(state.transcript[1].source, TranscriptSource::Microphone);
        assert_eq!(state.transcript[1].start_ms, Some(2_000));
        assert_eq!(
            state.transcript[1].speaker_id.as_deref(),
            Some(LOCAL_USER_SPEAKER_ID)
        );
        assert_eq!(state.transcript[1].speaker_label.as_deref(), Some("You"));
        assert!(state.notes.iter().any(|note| {
            note.evidence_segment_ids
                .contains(&state.transcript[0].segment_id)
        }));
        assert!(state.notes.iter().any(|note| {
            note.evidence_segment_ids
                .contains(&state.transcript[1].segment_id)
        }));
    }

    #[tokio::test]
    async fn captured_segment_success_records_completed_segment_metrics() {
        let root = temp_root();
        let runtime =
            MeetingRuntime::with_file_transcriber(root.clone(), Arc::new(FixedTranscriber));
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let storage_dir = root.join(".astra").join("meetings");
        let writer = SegmentWriter::new(
            storage_dir,
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::SystemAudio,
                ..SegmentWriterConfig::default()
            },
        );
        let samples = vec![100_i16; 16_000];
        let segment = writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("segment");
        let segment_id = segment
            .path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap()
            .to_string();

        runtime
            .transcribe_captured_segment(segment, None, false)
            .await
            .expect("transcription");

        let health = runtime.capture_health().expect("capture health");
        assert_eq!(health.metrics.segments_transcribed, 1);
        assert_eq!(
            health
                .metrics
                .last_transcription_completed_segment_id
                .as_deref(),
            Some(segment_id.as_str())
        );
    }

    #[tokio::test]
    async fn captured_segment_failure_records_failed_segment_metrics_and_timeout_kind() {
        let root = temp_root();
        let runtime = MeetingRuntime::with_file_transcriber(
            root.clone(),
            Arc::new(FailingTranscriber {
                reason: "Existing STT worker timed out",
            }),
        );
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let storage_dir = root.join(".astra").join("meetings");
        let writer = SegmentWriter::new(
            storage_dir,
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::SystemAudio,
                ..SegmentWriterConfig::default()
            },
        );
        let samples = vec![100_i16; 16_000];
        let segment = writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("segment");
        let segment_id = segment
            .path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap()
            .to_string();

        let result = runtime
            .transcribe_captured_segment(segment, None, false)
            .await;
        assert!(result.is_err());

        let health = runtime.capture_health().expect("capture health");
        assert_eq!(health.metrics.segments_failed, 1);
        assert_eq!(health.metrics.segment_transcription_timeouts, 1);
        assert_eq!(
            health
                .metrics
                .last_segment_transcription_error_kind
                .as_deref(),
            Some("stt_timeout")
        );
        assert_eq!(
            health
                .metrics
                .last_transcription_failed_segment_id
                .as_deref(),
            Some(segment_id.as_str())
        );
    }

    #[tokio::test]
    async fn stop_export_records_captured_but_untranscribed_segment_diagnostics() {
        let root = temp_root();
        let runtime = MeetingRuntime::with_file_transcriber(
            root.clone(),
            Arc::new(FailingTranscriber {
                reason: "Existing STT worker timed out",
            }),
        );
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let storage_dir = root.join(".astra").join("meetings");
        let writer = SegmentWriter::new(
            storage_dir,
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::SystemAudio,
                ..SegmentWriterConfig::default()
            },
        );
        let samples = vec![100_i16; 16_000];
        let segment = writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("segment");

        let result = runtime
            .transcribe_captured_segment(segment, None, false)
            .await;
        assert!(result.is_err());

        let exported = runtime.stop_session().expect("stop session");
        assert!(exported.transcript.is_empty());
        assert_eq!(
            exported
                .metadata
                .get("meeting_segment_transcription_incomplete")
                .and_then(|value| value.as_bool()),
            Some(true)
        );
        assert_eq!(
            exported
                .metadata
                .get("stt_completeness")
                .and_then(|value| value.get("overall"))
                .and_then(|value| value.as_str()),
            Some("incomplete_failed_segments")
        );
        let diagnostics = exported
            .metadata
            .get("segment_stt_diagnostics")
            .expect("segment STT diagnostics");
        assert_eq!(
            diagnostics
                .get("segments_written")
                .and_then(|value| value.as_u64()),
            Some(1)
        );
        assert_eq!(
            diagnostics
                .get("segments_failed")
                .and_then(|value| value.as_u64()),
            Some(1)
        );
        assert_eq!(
            diagnostics
                .get("last_stt_error")
                .and_then(|value| value.as_str()),
            Some("stt_timeout")
        );
        let list = runtime
            .list_archived_sessions(MeetingSessionListRequest {
                limit: 10,
                cursor: None,
                date_from: None,
                date_to: None,
                has_intelligence: None,
                query: None,
            })
            .expect("list archived sessions");
        let item = list.sessions.first().expect("archived session");
        assert_eq!(item.stt_completeness_status, "incomplete_failed_segments");
        assert!(item
            .stt_completeness_detail
            .contains("incomplete_failed_segments"));
    }

    #[tokio::test]
    async fn completed_session_is_archived_listed_read_searchable_and_exportable() {
        let root = temp_root();
        let runtime =
            MeetingRuntime::with_file_transcriber(root.clone(), Arc::new(FixedTranscriber));
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");

        let storage_dir = root.join(".astra").join("meetings");
        let samples = vec![100_i16; 16_000];
        let writer = SegmentWriter::new(
            storage_dir.clone(),
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::SystemAudio,
                ..SegmentWriterConfig::default()
            },
        );
        let segment = writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("system segment");
        runtime
            .transcribe_captured_segment(segment, None, false)
            .await
            .expect("system transcription");
        let active_after_transcript = runtime.get_active_state().expect("active state");
        let linked_entry_id = active_after_transcript
            .transcript
            .first()
            .expect("transcript entry")
            .segment_id
            .clone();
        let attached_screen = runtime
            .attach_screen_context(screen_context(&session.session_id, Utc::now()))
            .expect("attach screen context");
        assert_eq!(attached_screen.context.screenshot_ref, None);
        assert!(attached_screen
            .context
            .linked_transcript_segment_ids
            .contains(&linked_entry_id));
        runtime
            .rename_speaker(RenameSpeakerRequest {
                speaker_id: REMOTE_SPEAKER_1_ID.to_string(),
                display_name: "Marco".to_string(),
            })
            .expect("rename speaker");
        runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions::default())
            .await
            .expect("generate rule-based intelligence");

        let exported = runtime.stop_session().expect("stop session");
        let archive_dir = storage_dir.join("sessions").join(&exported.session_id);
        assert!(archive_dir.join("session.json").exists());
        assert!(archive_dir.join("transcript.json").exists());
        assert!(archive_dir.join("intelligence.json").exists());
        assert!(archive_dir.join("screen_context.json").exists());
        assert!(archive_dir.join("export.md").exists());

        let list = runtime
            .list_archived_sessions(MeetingSessionListRequest {
                limit: 10,
                cursor: None,
                date_from: None,
                date_to: None,
                has_intelligence: Some(true),
                query: None,
            })
            .expect("list archived sessions");
        assert_eq!(list.sessions.len(), 1);
        let item = list.sessions.first().expect("session list item");
        assert_eq!(item.session_id, exported.session_id);
        assert_eq!(item.transcript_count, 1);
        assert!(item.intelligence_present);
        assert!(item.speakers_preview.iter().any(|name| name == "Marco"));
        assert_eq!(item.stt_completeness_status, "complete");
        assert_eq!(item.screen_context_count, 1);

        let read = runtime
            .read_archived_session(MeetingSessionReadRequest {
                session_id: exported.session_id.clone(),
                include_transcript: true,
                include_intelligence: true,
                include_diagnostics: true,
            })
            .expect("read archived session");
        let archive = read.archive;
        assert_eq!(archive.state.transcript.len(), 1);
        assert_eq!(archive.state.screen_contexts.len(), 1);
        assert_eq!(archive.screen_contexts.len(), 1);
        assert!(archive.state.screen_contexts[0]
            .linked_transcript_segment_ids
            .contains(&linked_entry_id));
        assert_eq!(
            archive.state.transcript[0].speaker_label.as_deref(),
            Some("Marco")
        );
        let transcript_ids = archive
            .state
            .transcript
            .iter()
            .map(|entry| entry.segment_id.as_str())
            .collect::<std::collections::HashSet<_>>();
        let intelligence = archive
            .state
            .intelligence
            .as_ref()
            .expect("archived intelligence");
        let summary = intelligence.summary.as_ref().expect("summary");
        assert!(summary
            .evidence_segment_ids
            .iter()
            .all(|id| transcript_ids.contains(id.as_str())));

        let transcript_search = runtime
            .search_archived_sessions(MeetingSessionSearchRequest {
                query: "ship".to_string(),
                limit: 20,
            })
            .expect("search transcript");
        assert!(transcript_search
            .results
            .iter()
            .any(|result| result.matched_kind == "transcript"));

        let action_search = runtime
            .search_archived_sessions(MeetingSessionSearchRequest {
                query: "tomorrow".to_string(),
                limit: 20,
            })
            .expect("search action items");
        assert!(action_search.results.iter().any(|result| {
            matches!(
                result.matched_kind.as_str(),
                "action_item" | "intelligence_action_item"
            )
        }));

        let screen_search = runtime
            .search_archived_sessions(MeetingSessionSearchRequest {
                query: "gpu memory".to_string(),
                limit: 20,
            })
            .expect("search screen context");
        assert!(screen_search.results.iter().any(|result| {
            result.matched_kind == "screen_context"
                && result.screen_context_id.as_deref() == Some("screen-context-test")
        }));

        let markdown = runtime
            .export_archived_session(MeetingSessionExportRequest {
                session_id: exported.session_id.clone(),
                format: MeetingSessionExportFormat::Markdown,
            })
            .expect("export markdown");
        assert_eq!(
            markdown.filename,
            format!("{}_session.md", exported.session_id)
        );
        assert!(markdown.content.contains("# Meeting Session Recap"));
        assert!(markdown.content.contains("## Summary"));
        assert!(markdown.content.contains("## Action Items"));
        assert!(markdown.content.contains("## Screen Context"));
        assert!(markdown.content.contains("GPU memory warning"));
        assert!(markdown.content.contains("## Transcript"));

        let json = runtime
            .export_archived_session(MeetingSessionExportRequest {
                session_id: exported.session_id.clone(),
                format: MeetingSessionExportFormat::Json,
            })
            .expect("export json");
        assert!(json.content.contains(&exported.session_id));
        assert!(json.content.contains("\"schema_version\""));
        assert!(json.content.contains("\"screen_contexts\""));
    }

    #[test]
    fn screen_context_without_transcript_records_diagnostic_without_crashing() {
        let root = temp_root();
        let runtime = MeetingRuntime::new(root.clone());
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");

        let attached = runtime
            .attach_screen_context(screen_context(&session.session_id, Utc::now()))
            .expect("attach screen context");

        assert!(attached.context.linked_transcript_segment_ids.is_empty());
        assert!(attached
            .context
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "screen_context_no_transcript_link"));
        let state = runtime.get_active_state().expect("active state");
        assert_eq!(state.screen_contexts.len(), 1);
    }

    #[test]
    fn corrupt_archive_does_not_crash_reindex_or_search() {
        let root = temp_root();
        let runtime = MeetingRuntime::new(root.clone());
        let bad_dir = root
            .join(".astra")
            .join("meetings")
            .join("sessions")
            .join("bad-session");
        std::fs::create_dir_all(&bad_dir).expect("bad archive dir");
        std::fs::write(bad_dir.join("session.json"), "{not valid json").expect("bad archive");

        let rebuilt = runtime
            .rebuild_session_memory_index()
            .expect("rebuild index");
        assert!(rebuilt.sessions.is_empty());
        assert!(rebuilt
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "archive_corrupt"));

        let searched = runtime
            .search_archived_sessions(MeetingSessionSearchRequest {
                query: "decision".to_string(),
                limit: 10,
            })
            .expect("search corrupt archives");
        assert!(searched.results.is_empty());
        assert_eq!(searched.corrupt_archive_count, 0);
    }

    #[tokio::test]
    async fn recall_query_with_transcript_match_returns_evidence() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("STT drain", false))
            .await
            .expect("recall answer");

        assert_eq!(response.status, MeetingRecallStatus::Answered);
        assert!(response
            .evidence
            .iter()
            .any(|item| item.matched_kind == "transcript"));
        assert!(!response.answer.is_empty());
    }

    #[tokio::test]
    async fn recall_query_with_decision_artifact_returns_evidence() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("Ship STT drain export warning", false))
            .await
            .expect("recall answer");

        assert!(response
            .evidence
            .iter()
            .any(|item| item.matched_kind == "decision"));
    }

    #[tokio::test]
    async fn recall_query_with_screen_context_match_returns_screen_context_evidence() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("GPU memory warning", false))
            .await
            .expect("recall answer");

        assert!(response.evidence.iter().any(|item| {
            item.matched_kind == "screen_context"
                && item
                    .screen_context_ids
                    .iter()
                    .any(|id| id == "screen-context-test")
        }));
    }

    #[tokio::test]
    async fn recall_no_evidence_returns_insufficient_evidence() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("unrelated banana invoice", false))
            .await
            .expect("recall answer");

        assert_eq!(response.status, MeetingRecallStatus::InsufficientEvidence);
        assert!(response.evidence.is_empty());
    }

    #[tokio::test]
    async fn recall_local_llm_valid_json_is_validated_and_answered() {
        let llm = FixedMeetingLlm::json(
            r#"{"answer":"We decided to ship the STT drain export warning.","status":"answered","used_evidence_indexes":[0],"confidence":0.9,"follow_up_questions":[]}"#,
        );
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("STT drain", true))
            .await
            .expect("recall answer");

        assert_eq!(response.status, MeetingRecallStatus::Answered);
        assert!(response.diagnostics.llm_used);
        assert_eq!(response.evidence.len(), 1);
        assert!(response.answer.contains("STT drain"));
    }

    #[tokio::test]
    async fn recall_local_llm_invalid_json_degrades_to_fallback() {
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(FixedMeetingLlm::json("not json")),
        );
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("STT drain", true))
            .await
            .expect("recall answer");

        assert_eq!(response.status, MeetingRecallStatus::Degraded);
        assert!(response.diagnostics.fallback_used);
        assert!(response.diagnostics.output_json_parse_failed);
        assert!(!response.evidence.is_empty());
    }

    #[tokio::test]
    async fn recall_local_llm_invalid_evidence_index_is_rejected() {
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(FixedMeetingLlm::json(
                r#"{"answer":"Unsupported answer","status":"answered","used_evidence_indexes":[999],"confidence":0.9,"follow_up_questions":[]}"#,
            )),
        );
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("STT drain", true))
            .await
            .expect("recall answer");

        assert_eq!(response.status, MeetingRecallStatus::Degraded);
        assert_eq!(response.diagnostics.invalid_evidence_indexes, 1);
        assert!(response
            .diagnostics
            .warnings
            .iter()
            .any(|warning| warning == "local_llm_recall_missing_valid_evidence"));
    }

    #[tokio::test]
    async fn recall_retrieval_is_bounded_by_limit() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_recall_fixture(&runtime);
        let mut request = recall_request("STT drain warning", false);
        request.limit = 1;

        let response = runtime
            .answer_session_recall(request)
            .await
            .expect("recall answer");

        assert!(response.evidence.len() <= 1);
        assert_eq!(response.diagnostics.max_evidence, 1);
    }

    #[tokio::test]
    async fn recall_italian_query_prompts_italian_answer_language() {
        let llm = CapturingMeetingLlm::new(
            r#"{"answer":"Abbiamo deciso di mostrare un avviso sullo STT drain.","status":"answered","used_evidence_indexes":[0],"confidence":0.8,"follow_up_questions":[]}"#,
        );
        let prompts = llm.prompts.clone();
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        archive_recall_fixture(&runtime);

        let response = runtime
            .answer_session_recall(recall_request("Cosa abbiamo deciso sullo STT drain?", true))
            .await
            .expect("recall answer");

        assert_eq!(
            response.diagnostics.query_language,
            MeetingLanguage::Italian
        );
        let prompt = prompts
            .lock()
            .expect("prompts mutex")
            .first()
            .expect("prompt")
            .clone();
        assert!(prompt.contains("Rispondi in italiano."));
    }

    #[tokio::test]
    async fn recall_screen_topic_expands_to_directly_linked_screen_context() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_bumper_screen_fixture(&runtime, &[], Some(1));

        let response = runtime
            .answer_session_recall(recall_request(
                "quando stavamo parlando del bumper del auto, cosa stavamo guardando?",
                false,
            ))
            .await
            .expect("recall answer");

        assert_eq!(
            response.diagnostics.recall_intent,
            MeetingRecallIntent::ScreenContextForTopic
        );
        assert!(response
            .evidence
            .iter()
            .any(|item| item.matched_kind == "transcript"
                && item.snippet.to_lowercase().contains("bumper")));
        assert!(response.evidence.iter().any(|item| {
            item.matched_kind == "screen_context"
                && item.relation == MeetingRecallEvidenceRelation::LinkedScreenContext
                && item.snippet.contains("Bugatti Veyron")
        }));
        assert!(response.answer.contains("Bugatti Veyron"));
    }

    #[tokio::test]
    async fn recall_screen_topic_expands_to_temporal_screen_context() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_bumper_screen_fixture(&runtime, &[2, 3, 4], Some(5));

        let response = runtime
            .answer_session_recall(recall_request(
                "quando stavamo parlando del bumper del auto, cosa stavamo guardando?",
                false,
            ))
            .await
            .expect("recall answer");

        assert!(response.evidence.iter().any(|item| {
            item.matched_kind == "screen_context"
                && item.relation == MeetingRecallEvidenceRelation::TemporalScreenContext
        }));
        assert!(response.answer.contains("Vicino al momento"));
    }

    #[tokio::test]
    async fn recall_screen_topic_expands_to_same_session_screen_context_with_uncertainty() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_bumper_screen_fixture(&runtime, &[30, 31, 32], Some(33));

        let response = runtime
            .answer_session_recall(recall_request(
                "quando stavamo parlando del bumper del auto, cosa stavamo guardando?",
                false,
            ))
            .await
            .expect("recall answer");

        assert!(response.evidence.iter().any(|item| {
            item.matched_kind == "screen_context"
                && item.relation == MeetingRecallEvidenceRelation::SameSessionScreenContext
        }));
        assert!(response
            .answer
            .contains("Non ho una prova perfettamente sincronizzata"));
        assert!(response.answer.contains("Bugatti Veyron"));
        assert!(response
            .diagnostics
            .warnings
            .iter()
            .any(|warning| warning == "session_level_screen_context"));
    }

    #[tokio::test]
    async fn recall_screen_topic_with_transcript_but_no_screen_context_says_so() {
        let runtime =
            MeetingRuntime::with_file_transcriber(temp_root(), Arc::new(FixedTranscriber));
        archive_bumper_screen_fixture(&runtime, &[], None);

        let response = runtime
            .answer_session_recall(recall_request(
                "quando stavamo parlando del bumper del auto, cosa stavamo guardando?",
                false,
            ))
            .await
            .expect("recall answer");

        assert_eq!(response.status, MeetingRecallStatus::Answered);
        assert!(response
            .evidence
            .iter()
            .all(|item| item.matched_kind != "screen_context"));
        assert!(response
            .answer
            .contains("non ho trovato screen context archiviati"));
        assert!(response
            .diagnostics
            .warnings
            .iter()
            .any(|warning| warning == "screen_context_not_available"));
    }

    #[tokio::test]
    async fn speaker_rename_updates_metadata_without_mutating_transcript_text() {
        let root = temp_root();
        let runtime =
            MeetingRuntime::with_file_transcriber(root.clone(), Arc::new(FixedTranscriber));
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");

        let storage_dir = root.join(".astra").join("meetings");
        let samples = vec![100_i16; 16_000];
        let writer = SegmentWriter::new(
            storage_dir,
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::SystemAudio,
                ..SegmentWriterConfig::default()
            },
        );
        let segment = writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("system segment");
        runtime
            .transcribe_captured_segment(segment, None, false)
            .await
            .expect("system transcription");

        let before = runtime.get_active_state().expect("active state");
        let original_text = before.transcript[0].text.clone();
        let original_segment_id = before.transcript[0].segment_id.clone();
        assert_eq!(
            before.transcript[0].speaker_id.as_deref(),
            Some(REMOTE_SPEAKER_1_ID)
        );

        let result = runtime
            .rename_speaker(RenameSpeakerRequest {
                speaker_id: REMOTE_SPEAKER_1_ID.to_string(),
                display_name: "Marco".to_string(),
            })
            .expect("rename speaker");
        assert_eq!(result.renamed_entries, 1);

        let after = runtime.get_active_state().expect("active state");
        let entry = after.transcript.first().expect("transcript");
        assert_eq!(entry.text, original_text);
        assert_eq!(entry.segment_id, original_segment_id);
        assert_eq!(entry.speaker_label.as_deref(), Some("Marco"));
        assert_eq!(
            entry.speaker_attribution_method,
            SpeakerAttributionMethod::UserAssigned
        );
        assert_eq!(after.speaker_rename_count, 1);
        assert!(after.notes.iter().any(|note| {
            note.content.contains("[Marco]")
                && note.evidence_segment_ids.contains(&original_segment_id)
        }));
        assert!(after
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "speaker_label_renamed"));
    }

    #[tokio::test]
    async fn generated_intelligence_is_evidence_linked_and_does_not_mutate_transcript() {
        let runtime = MeetingRuntime::new(temp_root());
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        runtime
            .add_transcript(TranscriptEntry::sourced(
                "",
                TranscriptSource::Manual,
                "Simone",
                "We decided to generate transcript-backed intelligence. Please update src/types/meeting.ts by tomorrow. What blocker remains if cargo test fails?",
                0.9,
            ))
            .expect("add transcript");

        let before = runtime.get_active_state().expect("before");
        let before_text = before.transcript[0].text.clone();
        let before_segment_id = before.transcript[0].segment_id.clone();
        let result = runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions::default())
            .await
            .expect("generate intelligence");

        assert_eq!(result.status, MeetingIntelligenceStatus::Generated);
        assert!(result
            .summary
            .as_ref()
            .is_some_and(|summary| { summary.evidence_segment_ids.contains(&before_segment_id) }));
        assert!(result
            .action_items
            .iter()
            .all(|item| !item.evidence_segment_ids.is_empty()));
        assert!(result.follow_up_draft.is_some());
        assert!(result.technical_recap.as_ref().is_some_and(|recap| recap
            .mentioned_files
            .iter()
            .any(|file| file.ends_with(".ts"))));

        let after = runtime.get_active_state().expect("after");
        assert_eq!(after.transcript[0].text, before_text);
        assert_eq!(after.transcript[0].segment_id, before_segment_id);
        assert!(after.intelligence.is_some());
    }

    #[tokio::test]
    async fn clear_intelligence_removes_derived_intelligence_only() {
        let runtime = MeetingRuntime::new(temp_root());
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        runtime
            .add_transcript(TranscriptEntry::sourced(
                "",
                TranscriptSource::Manual,
                "Simone",
                "Please follow up on the meeting intelligence export.",
                0.9,
            ))
            .expect("add transcript");
        runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions::default())
            .await
            .expect("generate intelligence");
        runtime.clear_intelligence().expect("clear intelligence");

        let state = runtime.get_active_state().expect("state");
        assert!(!state.transcript.is_empty());
        assert!(state.intelligence.is_none());
        assert!(state
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "meeting_intelligence_cleared"));
    }

    #[tokio::test]
    async fn local_llm_success_stores_validated_artifacts() {
        let raw = r#"{
            "summary": {
                "text": "Abbiamo deciso di collegare il modello locale.",
                "bullets": ["Adapter Ollama collegato"],
                "evidence_segment_ids": ["seg-it"],
                "confidence": 0.91
            },
            "follow_up_draft": {
                "subject": "Riepilogo riunione Astra",
                "body": "Ciao,\nabbiamo deciso di collegare il modello locale e validare gli output con evidenza.",
                "evidence_segment_ids": ["seg-it"],
                "confidence": 0.88
            }
        }"#;
        let llm = FixedMeetingLlm::json(raw);
        let calls = llm.calls.clone();
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let mut entry = TranscriptEntry::sourced(
            "",
            TranscriptSource::Manual,
            "Simone",
            "Abbiamo deciso di collegare il modello locale Ollama alla meeting intelligence.",
            0.9,
        );
        entry.segment_id = "seg-it".to_string();
        runtime.add_transcript(entry).expect("add transcript");

        let result = runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions {
                use_local_llm: true,
                max_transcript_segments: 120,
            })
            .await
            .expect("generate");

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.status, MeetingIntelligenceStatus::Generated);
        assert!(result.diagnostics.llm_used);
        assert!(!result.diagnostics.fallback_used);
        assert!(matches!(
            result.summary.as_ref().map(|summary| &summary.generator),
            Some(ArtifactGenerator::LocalLlm { model, .. }) if model == "mock-meeting-model"
        ));
        assert!(result
            .follow_up_draft
            .as_ref()
            .is_some_and(|draft| draft.body.contains("Ciao")));
    }

    #[tokio::test]
    async fn language_retry_success_stores_corrected_output() {
        let first = r#"{
            "summary": {
                "text": "Here is the meeting summary and follow up from the session.",
                "bullets": ["The team decided the next step."],
                "evidence_segment_ids": ["seg-it"],
                "confidence": 0.8
            }
        }"#;
        let second = r#"{
            "summary": {
                "text": "Durante la sessione e stata confermata la sintesi italiana.",
                "bullets": ["La bozza di follow-up resta in italiano."],
                "evidence_segment_ids": ["seg-it"],
                "confidence": 0.86
            }
        }"#;
        let llm = SequencedMeetingLlm::new(vec![first, second]);
        let calls = llm.calls.clone();
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let mut entry = TranscriptEntry::sourced(
            "",
            TranscriptSource::Manual,
            "Simone",
            "Abbiamo deciso che il riepilogo e la bozza di follow-up devono essere in italiano.",
            0.9,
        );
        entry.segment_id = "seg-it".to_string();
        runtime.add_transcript(entry).expect("add transcript");

        let result = runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions {
                use_local_llm: true,
                max_transcript_segments: 120,
            })
            .await
            .expect("generate");

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(result.diagnostics.language_retry_attempted);
        assert!(result.diagnostics.language_retry_succeeded);
        assert!(!result.diagnostics.output_language_mismatch);
        assert_eq!(result.diagnostics.output_language, MeetingLanguage::Italian);
        assert!(result
            .summary
            .as_ref()
            .is_some_and(|summary| summary.text.starts_with("Durante")));
    }

    #[tokio::test]
    async fn language_retry_invalid_keeps_first_valid_output_with_diagnostic() {
        let first = r#"{
            "summary": {
                "text": "Here is the meeting summary and follow up from the session.",
                "bullets": ["The team decided the next step."],
                "evidence_segment_ids": ["seg-it"],
                "confidence": 0.8
            }
        }"#;
        let llm = SequencedMeetingLlm::new(vec![first, "{not valid json"]);
        let calls = llm.calls.clone();
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let mut entry = TranscriptEntry::sourced(
            "",
            TranscriptSource::Manual,
            "Simone",
            "Abbiamo deciso che il riepilogo e la bozza di follow-up devono essere in italiano.",
            0.9,
        );
        entry.segment_id = "seg-it".to_string();
        runtime.add_transcript(entry).expect("add transcript");

        let result = runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions {
                use_local_llm: true,
                max_transcript_segments: 120,
            })
            .await
            .expect("generate");

        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert!(result.diagnostics.language_retry_attempted);
        assert!(!result.diagnostics.language_retry_succeeded);
        assert!(result.diagnostics.output_language_mismatch);
        assert!(result
            .diagnostics
            .warnings
            .iter()
            .any(|warning| warning.contains("first valid output was kept")));
    }

    #[tokio::test]
    async fn local_llm_error_falls_back_truthfully() {
        let llm = FixedMeetingLlm::error(MeetingLlmErrorKind::Unavailable);
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        runtime
            .add_transcript(TranscriptEntry::sourced(
                "",
                TranscriptSource::Manual,
                "Simone",
                "Please generate grounded intelligence from this transcript.",
                0.9,
            ))
            .expect("add transcript");

        let result = runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions {
                use_local_llm: true,
                max_transcript_segments: 120,
            })
            .await
            .expect("fallback");

        assert_eq!(result.status, MeetingIntelligenceStatus::Degraded);
        assert!(!result.diagnostics.llm_used);
        assert!(result.diagnostics.fallback_used);
        assert_eq!(
            result.diagnostics.model_unavailable_reason.as_deref(),
            Some("local_llm_unavailable")
        );
        assert!(result.summary.is_some());
    }

    #[tokio::test]
    async fn rule_based_generation_does_not_call_llm_when_disabled() {
        let llm = FixedMeetingLlm::json("{}");
        let calls = llm.calls.clone();
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        runtime
            .add_transcript(TranscriptEntry::sourced(
                "",
                TranscriptSource::Manual,
                "Simone",
                "Please follow up on the local model adapter.",
                0.9,
            ))
            .expect("add transcript");

        let result = runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions {
                use_local_llm: false,
                max_transcript_segments: 120,
            })
            .await
            .expect("generate");

        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(!result.diagnostics.llm_used);
        assert!(!result.diagnostics.fallback_used);
        assert!(matches!(
            result.diagnostics.generator,
            ArtifactGenerator::RuleBased
        ));
    }

    #[tokio::test]
    async fn intelligence_generation_does_not_hold_registry_lock_during_llm_call() {
        let raw = r#"{
            "summary": {
                "text": "Generated from the first snapshot.",
                "bullets": ["Snapshot one"],
                "evidence_segment_ids": ["seg-lock-1"],
                "confidence": 0.8
            }
        }"#;
        let llm = FixedMeetingLlm::json(raw).with_delay(Duration::from_millis(150));
        let runtime = MeetingRuntime::with_file_transcriber_and_intelligence_llm(
            temp_root(),
            Arc::new(FixedTranscriber),
            Arc::new(llm),
        );
        runtime.grant_consent("teams").expect("grant consent");
        runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let mut first = TranscriptEntry::sourced(
            "",
            TranscriptSource::Manual,
            "Simone",
            "The first snapshot should be enough for the model.",
            0.9,
        );
        first.segment_id = "seg-lock-1".to_string();
        runtime.add_transcript(first).expect("add first transcript");

        let generator_runtime = runtime.clone();
        let task = tokio::spawn(async move {
            generator_runtime
                .generate_intelligence(MeetingIntelligenceGenerationOptions {
                    use_local_llm: true,
                    max_transcript_segments: 120,
                })
                .await
        });
        tokio::time::sleep(Duration::from_millis(25)).await;
        let add_result = tokio::time::timeout(Duration::from_millis(100), async {
            runtime.add_transcript(TranscriptEntry::sourced(
                "",
                TranscriptSource::Manual,
                "Simone",
                "This transcript arrives while the local model is still generating.",
                0.9,
            ))
        })
        .await;
        assert!(add_result.is_ok(), "registry lock was held during LLM call");
        assert!(add_result.expect("timeout result").is_ok());

        let result = task.await.expect("join").expect("generate");
        assert!(result.diagnostics.transcript_changed_during_generation);
        assert_eq!(result.diagnostics.snapshot_transcript_segment_count, 1);
    }

    #[tokio::test]
    async fn speaker_rename_refreshes_intelligence_labels_without_changing_evidence() {
        let root = temp_root();
        let runtime =
            MeetingRuntime::with_file_transcriber(root.clone(), Arc::new(FixedTranscriber));
        runtime.grant_consent("teams").expect("grant consent");
        let session = runtime
            .start_session(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
            )
            .expect("start manual session");
        let storage_dir = root.join(".astra").join("meetings");
        let samples = vec![100_i16; 16_000];
        let writer = SegmentWriter::new(
            storage_dir,
            SegmentWriterConfig {
                sample_rate: 16_000,
                channels: 1,
                transcript_source: TranscriptSource::SystemAudio,
                ..SegmentWriterConfig::default()
            },
        );
        let segment = writer
            .write_pcm_i16_segment(&session.session_id, &samples)
            .expect("system segment");
        runtime
            .transcribe_captured_segment(segment, None, false)
            .await
            .expect("system transcription");

        let generated = runtime
            .generate_intelligence(MeetingIntelligenceGenerationOptions::default())
            .await
            .expect("generate intelligence");
        let evidence_before = generated
            .timeline
            .first()
            .expect("timeline")
            .evidence_segment_ids
            .clone();
        runtime
            .rename_speaker(RenameSpeakerRequest {
                speaker_id: REMOTE_SPEAKER_1_ID.to_string(),
                display_name: "Marco".to_string(),
            })
            .expect("rename");

        let after = runtime
            .read_intelligence()
            .expect("read")
            .expect("intelligence");
        assert_eq!(
            after
                .timeline
                .first()
                .and_then(|item| item.speaker_display_name.as_deref()),
            Some("Marco")
        );
        assert_eq!(
            after
                .timeline
                .first()
                .expect("timeline")
                .evidence_segment_ids,
            evidence_before
        );
    }

    #[test]
    fn initial_diagnostics_report_truthful_speaker_attribution_limits() {
        let root = temp_root();
        let mut registry = SessionRegistry::new(root);
        registry
            .start(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
                MeetingStatus::Ready,
                false,
                Some("manual session".to_string()),
            )
            .expect("registry start");
        let state = registry.get_active_state();
        assert!(state
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "diarization_unsupported"));
        assert!(state
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "speaker_attribution_source_default"));
    }

    #[test]
    fn cannot_add_transcript_while_paused() {
        let root = temp_root();
        let mut registry = SessionRegistry::new(root);
        registry
            .start(
                "teams".to_string(),
                MeetingConfig {
                    session_mode: MeetingSessionMode::Manual,
                    ..config()
                },
                MeetingStatus::Ready,
                false,
                Some("manual session".to_string()),
            )
            .expect("registry start");
        registry.pause().expect("pause");
        let result = registry.add_transcript(transcript());
        assert!(result.is_err());
        assert!(matches!(
            registry.get_active_state().status,
            MeetingStatus::Paused
        ));
    }

    #[test]
    fn clear_meeting_data_removes_persisted_files() {
        let root = temp_root();
        let runtime = MeetingRuntime::new(root.clone());
        let meeting_dir = root.join(".astra").join("meetings").join("old_session");
        std::fs::create_dir_all(&meeting_dir).expect("meeting dir");
        std::fs::write(meeting_dir.join("notes.json"), "{}").expect("meeting file");

        let result = runtime
            .clear_all_data(ClearMeetingDataRequest {
                scope: MeetingClearScope::All,
                confirmation_phrase: CLEAR_MEETING_DATA_CONFIRMATION_PHRASE.to_string(),
            })
            .expect("clear data");
        assert_eq!(result.persisted_entries_removed, 1);
        assert!(!meeting_dir.exists());
    }

    #[test]
    fn manual_action_item_requires_active_mutable_session() {
        let runtime = MeetingRuntime::new(temp_root());
        let item = ActionItem {
            id: super::super::types::new_meeting_artifact_id(),
            session_id: String::new(),
            timestamp: Utc::now(),
            created_at: Utc::now(),
            title: "follow up".to_string(),
            description: "follow up".to_string(),
            assignee: None,
            deadline: None,
            status: ActionItemStatus::Open,
            evidence_segment_ids: Vec::new(),
        };
        let result = runtime.add_action_item(item);
        assert!(matches!(result, Err(MeetingRuntimeError::NoActiveSession)));
    }
}
