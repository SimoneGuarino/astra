use chrono::Utc;
use personal_ai_lib::meeting::{
    audio_capture::{
        AudioCapture, CaptureMetricsReporter, CapturedSegmentEnqueueOutcome, CapturedSegmentQueue,
    },
    audio_quality::{
        analyze_mono_i16_final_flush, analyze_mono_i16_segment,
        convert_interleaved_f32_to_i16_mono, downmix_interleaved_i16_to_mono,
    },
    call_detector::CallDetector,
    capture_controller::{CaptureController, CaptureControllerConfig},
    follow_up_sender::FollowUpSender,
    note_organizer::NoteOrganizer,
    privacy_control::PrivacyState,
    runtime::{MeetingRuntime, MeetingRuntimeError, MAX_MEETING_TRANSCRIPTION_AUDIO_BYTES},
    segment_writer::{SegmentWriter, SegmentWriterConfig},
    session_registry::SessionRegistry,
    stt_adapter::{
        ExistingSttClientMeetingAdapter, MeetingFileTranscriber, MeetingFileTranscriptionFuture,
        MeetingSttEngine,
    },
    transcription_stream::TranscriptionStream,
    types::{
        normalize_meeting_app_name, AudioFrame, AudioSampleFormat, CallDetectionState,
        CaptureBackend, CaptureControllerState, CaptureHealthStatus, CaptureOverflowPolicy,
        CapturePipelineConfig, ClearMeetingDataRequest, ExportedMeeting, MeetingAudioChunk,
        MeetingAudioFileTranscriptionRequest, MeetingAudioSegment, MeetingCapabilityReadiness,
        MeetingCapabilityState, MeetingClearScope, MeetingConfig, MeetingSessionMode,
        MeetingStatus, MeetingSttAdapterStatus, SpeakerAttributionMethod, TranscriptEntry,
        TranscriptSource, CLEAR_MEETING_DATA_CONFIRMATION_PHRASE,
        DEFAULT_CAPTURE_MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES,
        DEFAULT_CAPTURE_MAX_SEGMENTS_PER_SESSION, DEFAULT_CAPTURE_SEGMENT_DURATION_MS,
        REMOTE_SPEAKER_1_ID,
    },
    wasapi_loopback::wait_for_startup_result,
};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn temp_root(name: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!("astra_{name}_{suffix}"))
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
        capture_options: personal_ai_lib::meeting::types::MeetingCaptureOptions::default(),
    }
}

fn manual_config() -> MeetingConfig {
    MeetingConfig {
        session_mode: MeetingSessionMode::Manual,
        ..config()
    }
}

fn manual_config_for(platform: &str) -> MeetingConfig {
    MeetingConfig {
        platform: platform.to_string(),
        ..manual_config()
    }
}

fn transcript() -> TranscriptEntry {
    TranscriptEntry::sourced("", TranscriptSource::Manual, "speaker", "hello", 0.95)
}

struct FakeMeetingFileTranscriber {
    mode: FakeTranscriberMode,
}

enum FakeTranscriberMode {
    Return(String),
    Fail,
    FailNTimesThenReturn {
        remaining_failures: Arc<AtomicUsize>,
        text: String,
    },
    ReplaceManagedPathWithDirectory(String),
    ReplaceManagedPathWithDirectoryThenFail,
    WaitForRelease {
        text: String,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    },
}

impl MeetingFileTranscriber for FakeMeetingFileTranscriber {
    fn status(&self) -> MeetingSttAdapterStatus {
        MeetingSttAdapterStatus {
            state: MeetingCapabilityState::Ready,
            existing_boundary: "SttClient::transcribe(Path)".to_string(),
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
                Some("fake test transcriber; live streaming unsupported".to_string()),
            ),
            chunk_streaming: readiness(
                "meeting.transcription.chunk_streaming",
                false,
                MeetingCapabilityState::Unavailable,
                Some("fake test transcriber; chunk streaming unsupported".to_string()),
            ),
            chunk_streaming_supported: false,
            emits_placeholder_transcripts: false,
            reason: None,
        }
    }

    fn transcribe_file<'a>(&'a self, audio_path: &'a Path) -> MeetingFileTranscriptionFuture<'a> {
        Box::pin(async move {
            match &self.mode {
                FakeTranscriberMode::Return(text) => Ok(text.clone()),
                FakeTranscriberMode::Fail => Err(MeetingRuntimeError::TranscriptionUnavailable {
                    reason: "fake STT failed".to_string(),
                }),
                FakeTranscriberMode::FailNTimesThenReturn {
                    remaining_failures,
                    text,
                } => {
                    let should_fail = remaining_failures
                        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                            remaining.checked_sub(1)
                        })
                        .is_ok();
                    if should_fail {
                        Err(MeetingRuntimeError::TranscriptionUnavailable {
                            reason: "fake STT failed".to_string(),
                        })
                    } else {
                        Ok(text.clone())
                    }
                }
                FakeTranscriberMode::ReplaceManagedPathWithDirectory(text) => {
                    std::fs::remove_file(audio_path).map_err(|error| {
                        MeetingRuntimeError::StorageError {
                            message: format!(
                                "test setup could not remove managed file: {}",
                                error.kind()
                            ),
                        }
                    })?;
                    std::fs::create_dir(audio_path).map_err(|error| {
                        MeetingRuntimeError::StorageError {
                            message: format!(
                                "test setup could not create directory conflict: {}",
                                error.kind()
                            ),
                        }
                    })?;
                    Ok(text.clone())
                }
                FakeTranscriberMode::ReplaceManagedPathWithDirectoryThenFail => {
                    std::fs::remove_file(audio_path).map_err(|error| {
                        MeetingRuntimeError::StorageError {
                            message: format!(
                                "test setup could not remove managed file: {}",
                                error.kind()
                            ),
                        }
                    })?;
                    std::fs::create_dir(audio_path).map_err(|error| {
                        MeetingRuntimeError::StorageError {
                            message: format!(
                                "test setup could not create directory conflict: {}",
                                error.kind()
                            ),
                        }
                    })?;
                    Err(MeetingRuntimeError::TranscriptionUnavailable {
                        reason: "fake STT failed".to_string(),
                    })
                }
                FakeTranscriberMode::WaitForRelease {
                    text,
                    started,
                    release,
                } => {
                    started.notify_one();
                    release.notified().await;
                    Ok(text.clone())
                }
            }
        })
    }
}

fn runtime_with_fake_transcriber(name: &str, text: &str) -> (MeetingRuntime, PathBuf) {
    runtime_with_fake_transcriber_mode(name, FakeTranscriberMode::Return(text.to_string()))
}

fn runtime_with_fake_transcriber_mode(
    name: &str,
    mode: FakeTranscriberMode,
) -> (MeetingRuntime, PathBuf) {
    let root = temp_root(name);
    let runtime = MeetingRuntime::with_file_transcriber(
        root.clone(),
        Arc::new(FakeMeetingFileTranscriber { mode }),
    );
    (runtime, root)
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

fn write_test_audio(root: &Path, file_name: &str) -> PathBuf {
    let path = root.join(file_name);
    std::fs::create_dir_all(root).expect("audio root");
    std::fs::write(&path, b"RIFF....WAVEfmt ").expect("audio file");
    path
}

fn start_manual_session(runtime: &MeetingRuntime) -> String {
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start")
        .session_id
}

fn file_request(path: &Path) -> MeetingAudioFileTranscriptionRequest {
    MeetingAudioFileTranscriptionRequest {
        session_id: None,
        audio_path: path.to_string_lossy().to_string(),
        speaker: Some("stt".to_string()),
        cleanup_after_transcription: false,
    }
}

#[test]
fn meeting_session_mode_defaults_to_manual() {
    assert_eq!(MeetingSessionMode::default(), MeetingSessionMode::Manual);

    let registry = SessionRegistry::new(temp_root("meeting_default_mode"));
    assert_eq!(
        registry.get_active_state().session.session_mode,
        MeetingSessionMode::Manual
    );
}

#[test]
fn privacy_state_denies_recording_by_default() {
    let runtime = MeetingRuntime::new(temp_root("meeting_privacy"));
    let state = runtime.consent_state().expect("consent state");

    assert!(!state.given);
    assert!(!state.global_enabled);
}

#[test]
fn start_meeting_requires_explicit_consent() {
    let runtime = MeetingRuntime::new(temp_root("meeting_consent"));

    let result = runtime.start_session("teams".to_string(), config());

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::ConsentRequired { .. })
    ));
}

#[test]
fn unsupported_audio_capture_does_not_mark_capture_running() {
    let mut capture = AudioCapture::new(CaptureBackend::Default, "default".into(), 16_000);

    let result = capture.start();

    assert!(result.is_err());
    assert!(!capture.is_running());
}

#[test]
fn meeting_start_with_unsupported_capture_does_not_create_active_capture_session() {
    let runtime = MeetingRuntime::new(temp_root("meeting_unsupported_capture"));
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
fn manual_session_still_works_without_audio_capture() {
    let runtime = MeetingRuntime::new(temp_root("meeting_manual_without_capture"));
    runtime.grant_consent("teams").expect("grant consent");

    let session = runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");

    assert_eq!(session.session_mode, MeetingSessionMode::Manual);
    assert_eq!(session.status, MeetingStatus::Ready);
    assert!(!session.capture_active);
    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.state, CaptureControllerState::Idle);
    assert!(!health.active_handle_present);
}

#[test]
fn real_capture_unsupported_does_not_create_fake_capture_state() {
    let runtime = MeetingRuntime::new(temp_root("meeting_no_fake_capture"));
    runtime.grant_consent("teams").expect("grant consent");

    let result = runtime.start_session("teams".to_string(), config());

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureUnavailable { .. })
    ));
    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.state, CaptureControllerState::Unsupported);
    assert_eq!(health.status, CaptureHealthStatus::Unsupported);
    assert!(!health.active_handle_present);
}

#[test]
fn capture_controller_stop_without_start_is_safe() {
    let mut controller = CaptureController::new();

    let health = controller.stop().expect("stop without start");

    assert_eq!(health.state, CaptureControllerState::Idle);
    assert_eq!(health.status, CaptureHealthStatus::Idle);
    assert!(!health.active_handle_present);
}

#[test]
fn capture_controller_pause_without_capture_is_rejected() {
    let mut controller = CaptureController::new();

    let result = controller.pause();

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::UnsupportedCapability { .. })
    ));
}

#[test]
fn unsupported_capture_backend_does_not_create_active_capture_handle() {
    let mut controller = CaptureController::new();
    let controller_config = CaptureControllerConfig::from_meeting_config(&config());

    let result = controller.start(controller_config);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureUnavailable { .. })
    ));
    assert!(!controller.health_snapshot().active_handle_present);
}

#[test]
fn unsupported_start_never_leaves_active_handle() {
    let mut controller = CaptureController::new();
    let controller_config = CaptureControllerConfig::from_meeting_config(&config());

    let result = controller.start(controller_config);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureUnavailable { .. })
    ));
    let health = controller.health_snapshot();
    assert_eq!(health.state, CaptureControllerState::Unsupported);
    assert!(!health.active_handle_present);
}

#[test]
fn capture_controller_abort_without_handle_is_terminal() {
    let mut controller = CaptureController::new();

    let health = controller
        .abort("test abort".to_string())
        .expect("abort without handle");

    assert_eq!(health.state, CaptureControllerState::Failed);
    assert_eq!(health.status, CaptureHealthStatus::Failed);
    assert!(!health.active_handle_present);
    assert_eq!(health.last_error.as_deref(), Some("test abort"));
}

#[test]
fn stop_is_idempotent_and_health_is_truthful() {
    let mut controller = CaptureController::new();

    let first = controller.stop().expect("first stop");
    let second = controller.stop().expect("second stop");

    assert_eq!(first.state, CaptureControllerState::Idle);
    assert_eq!(second.state, CaptureControllerState::Idle);
    assert_eq!(second.status, CaptureHealthStatus::Idle);
    assert!(!second.active_handle_present);
    assert!(!controller.health_snapshot().active_handle_present);
}

#[test]
fn unsupported_capture_backend_does_not_mark_session_capturing() {
    let runtime = MeetingRuntime::new(temp_root("meeting_no_capturing_state"));
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
    assert_ne!(
        runtime.get_active_state().expect("active state").status,
        MeetingStatus::Capturing
    );
}

#[test]
fn non_windows_capture_backend_returns_typed_unsupported() {
    let mut controller = CaptureController::new();
    let mut real_capture_config = config();
    real_capture_config.capture_backend = CaptureBackend::CoreAudio;
    let controller_config = CaptureControllerConfig::from_meeting_config(&real_capture_config);

    let result = controller.start(controller_config);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureUnavailable {
            backend: CaptureBackend::CoreAudio,
            ..
        })
    ));
    let health = controller.health_snapshot();
    assert_eq!(health.state, CaptureControllerState::Unsupported);
    assert_eq!(health.status, CaptureHealthStatus::Unsupported);
    assert!(!health.active_handle_present);
}

#[test]
fn segment_writer_writes_valid_wav_header() {
    let root = temp_root("meeting_segment_writer_header");
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );

    let segment = writer
        .write_pcm_i16_segment("session_1", &[0, 128, -128, 0])
        .expect("write segment");
    let bytes = std::fs::read(&segment.path).expect("read segment");

    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(&bytes[36..40], b"data");
    assert_eq!(segment.byte_length, 52);
    assert!(segment.managed_path_redacted);
}

#[test]
fn segment_writer_rejects_oversized_segment() {
    let root = temp_root("meeting_segment_writer_large");
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig {
            max_bytes: 45,
            ..SegmentWriterConfig::default()
        },
    );

    let result = writer.write_pcm_i16_segment("session_1", &[0]);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::SegmentTooLarge { .. })
    ));
}

#[test]
fn cannot_add_transcript_without_active_session() {
    let runtime = MeetingRuntime::new(temp_root("meeting_no_session"));

    let result = runtime.add_transcript(transcript());

    assert!(matches!(result, Err(MeetingRuntimeError::NoActiveSession)));
}

#[test]
fn session_lifecycle_rejects_invalid_transitions() {
    let mut registry = SessionRegistry::new(temp_root("meeting_invalid_transition"));
    registry
        .start(
            "teams".to_string(),
            manual_config(),
            MeetingStatus::Ready,
            false,
            Some("manual session: no audio capture started".to_string()),
        )
        .expect("manual session start");

    let result = registry.transition_to(MeetingStatus::ConsentRequired);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidLifecycleTransition { .. })
    ));
}

#[test]
fn session_registry_does_not_silently_ignore_metadata_write_failure() {
    let root = temp_root("meeting_metadata_write_failure");
    std::fs::create_dir_all(root.join(".astra")).expect("state dir");
    std::fs::write(root.join(".astra").join("meetings"), b"not a directory")
        .expect("meetings file conflict");
    let mut registry = SessionRegistry::new(root);

    let result = registry.start(
        "teams".to_string(),
        manual_config(),
        MeetingStatus::Ready,
        false,
        Some("manual session".to_string()),
    );

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::StorageError { .. })
    ));
    assert!(registry.get_active_session().is_none());
}

#[test]
fn session_registry_propagates_partial_state_write_failure() {
    let root = temp_root("meeting_partial_write_failure");
    let mut registry = SessionRegistry::new(root.clone());
    registry
        .start(
            "teams".to_string(),
            manual_config(),
            MeetingStatus::Ready,
            false,
            Some("manual session".to_string()),
        )
        .expect("manual session start");
    std::fs::remove_dir_all(root.join(".astra").join("meetings")).expect("remove meetings dir");
    std::fs::write(root.join(".astra").join("meetings"), b"not a directory")
        .expect("meetings file conflict");

    let result = registry.add_transcript(transcript());

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::StorageError { .. })
    ));
}

#[test]
fn cannot_add_transcript_while_paused() {
    let runtime = MeetingRuntime::new(temp_root("meeting_paused"));
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");
    runtime.pause_session().expect("pause");

    let result = runtime.add_transcript(transcript());

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::SessionPaused { .. })
    ));
    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.status, MeetingStatus::Paused);
    assert_eq!(state.paused_from, Some(MeetingStatus::Ready));
}

#[test]
fn pause_session_preserves_previous_state() {
    let runtime = MeetingRuntime::new(temp_root("meeting_pause_previous"));
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");
    runtime
        .add_transcript(transcript())
        .expect("manual transcript");

    runtime.pause_session().expect("pause");

    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.status, MeetingStatus::Paused);
    assert_eq!(state.paused_from, Some(MeetingStatus::Transcribing));
}

#[test]
fn resume_session_restores_previous_state() {
    let runtime = MeetingRuntime::new(temp_root("meeting_resume_previous"));
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");
    runtime
        .add_transcript(transcript())
        .expect("manual transcript");
    runtime.pause_session().expect("pause");

    runtime.resume_session().expect("resume");

    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.status, MeetingStatus::Transcribing);
    assert_eq!(state.paused_from, None);
}

#[test]
fn can_add_manual_transcript_only_in_valid_state_if_manual_mode_exists() {
    let runtime = MeetingRuntime::new(temp_root("meeting_manual_transcript"));
    runtime.grant_consent("teams").expect("grant consent");
    let session = runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");

    assert_eq!(session.session_mode, MeetingSessionMode::Manual);
    assert!(!session.capture_active);
    runtime
        .add_transcript(transcript())
        .expect("manual transcript");

    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.status, MeetingStatus::Transcribing);
    assert_eq!(state.transcript.len(), 1);
}

#[test]
fn manual_transcript_does_not_require_live_stt_permission() {
    let runtime = MeetingRuntime::new(temp_root("meeting_manual_no_live_stt"));
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");

    runtime
        .add_transcript(transcript())
        .expect("manual transcript does not use live STT");

    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.transcript.len(), 1);
}

#[test]
fn live_transcription_request_remains_unsupported_without_capture_backend() {
    let runtime = MeetingRuntime::new(temp_root("meeting_live_stt_unsupported"));
    runtime.grant_consent("teams").expect("grant consent");
    let mut live_config = config();
    live_config.live_transcription_enabled = true;

    let result = runtime.start_session("teams".to_string(), live_config);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureUnavailable { .. })
    ));
    let capabilities = runtime.live_capabilities().expect("capabilities");
    assert_eq!(
        capabilities.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.stt_adapter.chunk_streaming_supported);
}

#[test]
fn clear_meeting_data_requires_confirmation() {
    let root = temp_root("meeting_clear_data_requires_confirmation");
    let runtime = MeetingRuntime::new(root.clone());
    let meeting_dir = root.join(".astra").join("meetings").join("old_session");
    std::fs::create_dir_all(&meeting_dir).expect("meeting dir");
    std::fs::write(meeting_dir.join("notes.json"), "{}").expect("meeting file");

    let result = runtime.clear_all_data(ClearMeetingDataRequest {
        scope: MeetingClearScope::All,
        confirmation_phrase: "DELETE".to_string(),
    });

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::ConfirmationRequired { .. })
    ));
    assert!(meeting_dir.exists());
}

#[test]
fn clear_meeting_data_confirmed_removes_persisted_files() {
    let root = temp_root("meeting_clear_data");
    let runtime = MeetingRuntime::new(root.clone());
    let meeting_dir = root.join(".astra").join("meetings").join("old_session");
    std::fs::create_dir_all(&meeting_dir).expect("meeting dir");
    std::fs::write(meeting_dir.join("notes.json"), "{}").expect("meeting file");

    let preview = runtime
        .preview_clear_all_data()
        .expect("preview clear data");
    assert_eq!(preview.persisted_entries, 1);

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
fn grant_multiple_apps_then_revoke_one_keeps_other_allowed() {
    let mut privacy = PrivacyState::new();
    privacy.grant_consent("Microsoft Teams");
    privacy.grant_consent(" zoom ");

    privacy.revoke_consent("teams");

    assert!(privacy.global_enabled);
    assert!(privacy.consent_given);
    assert!(!privacy.can_record("microsoft_teams"));
    assert!(privacy.can_record("Zoom"));
}

#[test]
fn revoking_last_allowed_app_disables_global_consent() {
    let mut privacy = PrivacyState::new();
    privacy.grant_consent("Teams");

    privacy.revoke_consent("Microsoft Teams");

    assert!(!privacy.global_enabled);
    assert!(!privacy.consent_given);
    assert!(!privacy.can_record("teams"));
}

#[test]
fn meeting_app_name_normalization_handles_case_whitespace_and_aliases() {
    assert_eq!(normalize_meeting_app_name(" Teams "), "teams");
    assert_eq!(normalize_meeting_app_name("Microsoft Teams"), "teams");
    assert_eq!(normalize_meeting_app_name("microsoft_teams"), "teams");
    assert_eq!(normalize_meeting_app_name("Google Meet"), "google_meet");
    assert_eq!(normalize_meeting_app_name("Custom App"), "custom_app");
}

#[test]
fn start_session_rejects_platform_config_mismatch() {
    let runtime = MeetingRuntime::new(temp_root("meeting_platform_mismatch"));
    runtime.grant_consent("teams").expect("grant consent");

    let result = runtime.start_session("teams".to_string(), manual_config_for("zoom"));

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
    assert!(runtime
        .get_active_session()
        .expect("active session")
        .is_none());
}

#[test]
fn manual_mode_rejects_live_transcription_enabled() {
    let runtime = MeetingRuntime::new(temp_root("meeting_manual_live_invalid"));
    runtime.grant_consent("teams").expect("grant consent");
    let mut config = manual_config();
    config.live_transcription_enabled = true;

    let result = runtime.start_session("teams".to_string(), config);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[test]
fn start_rejects_empty_platform() {
    let runtime = MeetingRuntime::new(temp_root("meeting_empty_platform"));

    let result = runtime.start_session(" ".to_string(), manual_config());

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[test]
fn start_rejects_platform_config_mismatch() {
    let runtime = MeetingRuntime::new(temp_root("meeting_platform_mismatch_exact"));
    runtime.grant_consent("teams").expect("grant consent");

    let result = runtime.start_session("teams".to_string(), manual_config_for("zoom"));

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[test]
fn start_rejects_invalid_sample_rate() {
    let runtime = MeetingRuntime::new(temp_root("meeting_invalid_sample_rate"));
    runtime.grant_consent("teams").expect("grant consent");
    let mut config = manual_config();
    config.sample_rate = 0;

    let result = runtime.start_session("teams".to_string(), config);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[test]
fn start_rejects_empty_transcription_model() {
    let runtime = MeetingRuntime::new(temp_root("meeting_empty_model"));
    runtime.grant_consent("teams").expect("grant consent");
    let mut config = manual_config();
    config.transcription_model.clear();

    let result = runtime.start_session("teams".to_string(), config);

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[test]
fn start_session_accepts_equivalent_normalized_platform_names() {
    let runtime = MeetingRuntime::new(temp_root("meeting_platform_normalized"));
    runtime
        .grant_consent("Microsoft Teams")
        .expect("grant consent");

    let session = runtime
        .start_session(" teams ".to_string(), manual_config_for("microsoft_teams"))
        .expect("manual session start");

    assert_eq!(session.platform, "teams");
    assert_eq!(session.config.platform, "teams");
    assert_eq!(session.session_mode, MeetingSessionMode::Manual);
}

#[test]
fn completed_state_is_not_confused_with_active_session() {
    let runtime = MeetingRuntime::new(temp_root("meeting_completed_state"));
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");
    runtime
        .add_transcript(transcript())
        .expect("manual transcript");

    let exported = runtime.stop_session().expect("stop session");

    assert_eq!(exported.transcript.len(), 1);
    assert!(runtime
        .get_active_session()
        .expect("active session")
        .is_none());
    let active_state = runtime.get_active_state().expect("active state");
    assert_eq!(active_state.status, MeetingStatus::Idle);
    let completed_state = runtime
        .get_last_completed_state()
        .expect("last completed state")
        .expect("completed state");
    assert_eq!(completed_state.status, MeetingStatus::Stopped);
    assert_eq!(completed_state.transcript.len(), 1);
}

#[tokio::test]
async fn transcribe_file_requires_active_session() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_no_session", "hello");
    let audio = write_test_audio(&root, "sample.wav");

    let result = runtime.transcribe_audio_file(file_request(&audio)).await;

    assert!(matches!(result, Err(MeetingRuntimeError::NoActiveSession)));
}

#[tokio::test]
async fn transcribe_file_requires_consent() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_no_consent", "hello");
    let audio = write_test_audio(&root, "sample.wav");
    start_manual_session(&runtime);
    runtime.revoke_consent("teams").expect("revoke consent");

    let result = runtime.transcribe_audio_file(file_request(&audio)).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::ConsentRequired { .. })
    ));
}

#[tokio::test]
async fn transcribe_file_rejects_session_id_mismatch() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_session_mismatch", "hello");
    let audio = write_test_audio(&root, "sample.wav");
    start_manual_session(&runtime);
    let mut request = file_request(&audio);
    request.session_id = Some("different-session".to_string());

    let result = runtime.transcribe_audio_file(request).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[tokio::test]
async fn transcribe_file_rejects_missing_file() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_missing", "hello");
    start_manual_session(&runtime);
    let missing = root.join("missing.wav");

    let result = runtime.transcribe_audio_file(file_request(&missing)).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[tokio::test]
async fn transcribe_file_rejects_unsupported_extension() {
    let (runtime, root) =
        runtime_with_fake_transcriber("meeting_file_unsupported_extension", "hello");
    let audio = write_test_audio(&root, "sample.mp3");
    start_manual_session(&runtime);

    let result = runtime.transcribe_audio_file(file_request(&audio)).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[tokio::test]
async fn wav_header_validation_rejects_fake_wav() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_fake_wav", "hello");
    std::fs::create_dir_all(&root).expect("audio root");
    let audio = root.join("fake.wav");
    std::fs::write(&audio, b"not a real WAV header").expect("fake wav");
    start_manual_session(&runtime);

    let result = runtime.transcribe_audio_file(file_request(&audio)).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[tokio::test]
async fn too_small_wav_file_is_rejected() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_too_small_wav", "hello");
    std::fs::create_dir_all(&root).expect("audio root");
    let audio = root.join("too_small.wav");
    std::fs::write(&audio, b"RIFF").expect("too small wav");
    start_manual_session(&runtime);

    let result = runtime.transcribe_audio_file(file_request(&audio)).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[tokio::test]
async fn transcribe_file_rejects_too_large_file() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_too_large", "hello");
    std::fs::create_dir_all(&root).expect("audio root");
    let audio = root.join("large.wav");
    let file = std::fs::File::create(&audio).expect("large audio");
    file.set_len(MAX_MEETING_TRANSCRIPTION_AUDIO_BYTES + 1)
        .expect("set large file length");
    start_manual_session(&runtime);

    let result = runtime.transcribe_audio_file(file_request(&audio)).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::InvalidConfig { .. })
    ));
}

#[tokio::test]
async fn valid_minimal_wav_header_passes_validation_step() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_valid_wav", "hello");
    let audio = write_test_audio(&root, "sample.wav");
    start_manual_session(&runtime);

    let result = runtime
        .transcribe_audio_file(file_request(&audio))
        .await
        .expect("valid wav transcription");

    assert!(result.transcript_added);
    assert_eq!(result.audio_file_extension, "wav");
    assert_eq!(result.file_size_bytes, 16);
    assert!(result.source_audio_path_redacted);
}

#[tokio::test]
async fn transcribe_file_adds_transcript_from_existing_stt_output() {
    let (runtime, root) =
        runtime_with_fake_transcriber("meeting_file_adds_transcript", "real transcript text");
    let audio = write_test_audio(&root, "sample.wav");
    let session_id = start_manual_session(&runtime);
    let mut request = file_request(&audio);
    request.session_id = Some(session_id);
    request.speaker = Some("recorder".to_string());

    let result = runtime
        .transcribe_audio_file(request)
        .await
        .expect("file transcription");

    assert!(result.transcript_added);
    assert_eq!(result.transcript_index, 0);
    assert_eq!(result.text_length, "real transcript text".chars().count());
    assert_eq!(result.audio_file_extension, "wav");
    assert_eq!(result.file_size_bytes, 16);
    assert_eq!(result.stt_boundary, "SttClient::transcribe(Path)");
    assert!(result.source_audio_path_redacted);
    assert!(result.managed_audio_path_redacted);
    assert!(!result.cleanup_requested);
    assert!(!result.cleanup_performed);
    assert!(result.cleanup_error.is_none());

    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.transcript.len(), 1);
    assert_eq!(state.transcript[0].speaker, "recorder");
    assert_eq!(state.transcript[0].text, "real transcript text");
    assert_eq!(state.transcript[0].confidence, 0.0);
}

#[tokio::test]
async fn captured_segment_uses_existing_stt_file_bridge() {
    let (runtime, root) =
        runtime_with_fake_transcriber("meeting_captured_segment_bridge", "captured transcript");
    let session_id = start_manual_session(&runtime);
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );
    let segment = writer
        .write_pcm_i16_segment(&session_id, &[0, 100, -100, 0])
        .expect("write captured segment");

    let result = runtime
        .transcribe_captured_segment(segment, Some("capture".to_string()), true)
        .await
        .expect("captured segment transcription");

    assert!(result.transcript_added);
    assert_eq!(result.text_length, "captured transcript".chars().count());
    assert_eq!(result.audio_file_extension, "wav");
    assert_eq!(result.stt_boundary, "SttClient::transcribe(Path)");
    assert!(result.cleanup_requested);
    assert!(result.cleanup_performed);
    assert!(result.managed_audio_path_redacted);

    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.transcript.len(), 1);
    assert_eq!(state.transcript[0].speaker, "capture");
    assert_eq!(state.transcript[0].text, "captured transcript");

    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.metrics.segments_written, 1);
    assert_eq!(health.metrics.segments_transcribed, 1);
    assert_eq!(
        health.last_segment_status.as_deref(),
        Some("segment_transcribed")
    );
}

#[tokio::test]
async fn revoked_consent_stops_capture_before_transcribing_next_segment() {
    let (runtime, root) =
        runtime_with_fake_transcriber("meeting_captured_segment_consent_revoked", "should not add");
    let session_id = start_manual_session(&runtime);
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );
    let segment = writer
        .write_pcm_i16_segment(&session_id, &[0, 100, -100, 0])
        .expect("write captured segment");
    runtime.revoke_consent("teams").expect("revoke consent");

    let result = runtime
        .transcribe_captured_segment(segment, Some("capture".to_string()), true)
        .await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::ConsentRevoked { .. })
    ));
    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.status, CaptureHealthStatus::ConsentRevoked);
    assert_eq!(
        health.last_segment_status.as_deref(),
        Some("consent_revoked")
    );
    assert_eq!(
        runtime.get_active_state().expect("state").transcript.len(),
        0
    );
}

#[test]
fn consent_revoke_stops_active_capture_immediately() {
    let (runtime, _root) =
        runtime_with_fake_transcriber("meeting_consent_revoke_active_capture", "unused");
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");
    let installed = runtime
        .install_fake_active_capture_for_test(true, Duration::from_millis(25))
        .expect("install fake active capture");
    assert!(installed.active_handle_present);

    let started = Instant::now();
    runtime.revoke_consent("teams").expect("revoke consent");
    assert!(
        started.elapsed() < Duration::from_millis(250),
        "consent revoke should not wait on unbounded capture shutdown"
    );

    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.status, CaptureHealthStatus::ConsentRevoked);
    assert!(!health.active_handle_present);
    assert_eq!(
        health.last_segment_status.as_deref(),
        Some("consent_revoked")
    );
    let session = runtime
        .get_active_session()
        .expect("active session")
        .expect("session remains inspectable");
    assert!(!session.capture_active);
    assert_eq!(
        session.capture_backend_status.as_deref(),
        Some("capture stopped: consent_revoked")
    );
    assert!(matches!(
        session.status,
        MeetingStatus::Failed(ref reason) if reason == "consent_revoked"
    ));
}

#[tokio::test]
async fn consent_revoke_prevents_late_segment_transcription() {
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let root = temp_root("meeting_late_segment_consent");
    let runtime = MeetingRuntime::with_file_transcriber(
        root.clone(),
        Arc::new(FakeMeetingFileTranscriber {
            mode: FakeTranscriberMode::WaitForRelease {
                text: "late transcript should not be inserted".to_string(),
                started: started.clone(),
                release: release.clone(),
            },
        }),
    );
    let session_id = start_manual_session(&runtime);
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );
    let segment = writer
        .write_pcm_i16_segment(&session_id, &[0, 100, -100, 0])
        .expect("write captured segment");
    let runtime_for_task = runtime.clone();

    let task = tokio::spawn(async move {
        runtime_for_task
            .transcribe_captured_segment(segment, Some("capture".to_string()), true)
            .await
    });
    started.notified().await;
    runtime.revoke_consent("teams").expect("revoke consent");
    release.notify_one();

    let result = task.await.expect("transcription task joins");
    assert!(matches!(
        result,
        Err(MeetingRuntimeError::ConsentRevoked { .. })
    ));
    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.transcript.len(), 0);
    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.status, CaptureHealthStatus::ConsentRevoked);
}

#[test]
fn wasapi_startup_error_preserves_cause() {
    let (ok_tx, ok_rx) = mpsc::channel();
    ok_tx.send(Ok(())).expect("send startup ok");
    assert!(
        wait_for_startup_result(&ok_rx, Duration::from_millis(10), CaptureBackend::Wasapi).is_ok()
    );

    let (error_tx, error_rx) = mpsc::channel();
    error_tx
        .send(Err(MeetingRuntimeError::CaptureStartFailed {
            backend: CaptureBackend::Wasapi,
            reason: "activate audio client failed: 0x88890004".to_string(),
        }))
        .expect("send startup error");
    let error =
        wait_for_startup_result(&error_rx, Duration::from_millis(10), CaptureBackend::Wasapi)
            .expect_err("typed startup error");
    assert!(matches!(
        error,
        MeetingRuntimeError::CaptureStartFailed { ref reason, .. }
            if reason.contains("activate audio client")
                && reason.contains("0x88890004")
    ));

    let (timeout_tx, timeout_rx) = mpsc::channel::<Result<(), MeetingRuntimeError>>();
    let timeout = wait_for_startup_result(
        &timeout_rx,
        Duration::from_millis(5),
        CaptureBackend::Wasapi,
    )
    .expect_err("startup timeout");
    assert!(matches!(
        timeout,
        MeetingRuntimeError::CaptureStartupTimeout { .. }
    ));
    drop(timeout_tx);

    let (closed_tx, closed_rx) = mpsc::channel::<Result<(), MeetingRuntimeError>>();
    drop(closed_tx);
    let closed = wait_for_startup_result(
        &closed_rx,
        Duration::from_millis(10),
        CaptureBackend::Wasapi,
    )
    .expect_err("startup channel closed");
    assert!(matches!(
        closed,
        MeetingRuntimeError::CaptureStartupChannelClosed { .. }
    ));
}

#[test]
fn capture_stop_timeout_is_bounded_and_terminal() {
    let mut controller = CaptureController::new();
    let controller_config = CaptureControllerConfig::from_meeting_config(&config());
    controller.install_fake_active_capture_for_test(
        controller_config,
        false,
        Duration::from_millis(25),
    );

    let started = Instant::now();
    let result = controller.stop();
    assert!(
        started.elapsed() < Duration::from_millis(250),
        "fake non-acknowledging stop must return within the configured bound"
    );
    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureStopTimedOut { .. })
    ));

    let health = controller.health_snapshot();
    assert_eq!(health.state, CaptureControllerState::Failed);
    assert_eq!(health.status, CaptureHealthStatus::StopTimedOut);
    assert!(!health.active_handle_present);
    assert_eq!(
        health.last_segment_status.as_deref(),
        Some("capture_stop_timed_out")
    );
}

#[test]
fn capture_stop_timeout_preserves_typed_error() {
    let mut capture = AudioCapture::new(CaptureBackend::Wasapi, "default".into(), 16_000);
    capture.install_fake_running_capture_for_test(Duration::from_millis(25));

    let result = capture.stop();

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureStopTimedOut {
            backend: CaptureBackend::Wasapi,
            timeout_ms: 25,
        })
    ));
    assert!(!capture.is_running());
}

#[test]
fn audio_capture_handle_preserves_capture_stop_timeout() {
    let mut controller = CaptureController::new();
    let controller_config = CaptureControllerConfig::from_meeting_config(&config());
    controller.install_fake_active_capture_for_test(
        controller_config,
        false,
        Duration::from_millis(25),
    );

    let result = controller.stop();

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::CaptureStopTimedOut { .. })
    ));
    assert!(!matches!(
        result,
        Err(MeetingRuntimeError::CaptureUnavailable { .. })
    ));
}

#[tokio::test]
async fn one_segment_transcription_failure_records_warning_without_failing_capture() {
    let (runtime, root) =
        runtime_with_fake_transcriber_mode("meeting_segment_warning", FakeTranscriberMode::Fail);
    let session_id = start_manual_session(&runtime);
    runtime
        .install_fake_active_capture_for_test(true, Duration::from_millis(25))
        .expect("fake active capture");
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );
    let segment = writer
        .write_pcm_i16_segment(&session_id, &[0, 100, -100, 0])
        .expect("write captured segment");

    let result = runtime
        .transcribe_captured_segment(segment, Some("capture".to_string()), true)
        .await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::TranscriptionUnavailable { .. })
    ));
    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.state, CaptureControllerState::Capturing);
    assert_eq!(health.status, CaptureHealthStatus::Healthy);
    assert!(health.active_handle_present);
    assert_eq!(health.metrics.segment_transcription_failures_total, 1);
    assert_eq!(health.metrics.segment_transcription_failures_consecutive, 1);
    assert_eq!(
        health
            .metrics
            .last_segment_transcription_error_kind
            .as_deref(),
        Some("transcription_unavailable")
    );
    assert_eq!(
        runtime.get_active_state().expect("state").transcript.len(),
        0
    );
}

#[tokio::test]
async fn consecutive_segment_transcription_failures_trip_failure_threshold() {
    let (runtime, root) =
        runtime_with_fake_transcriber_mode("meeting_segment_threshold", FakeTranscriberMode::Fail);
    let session_id = start_manual_session(&runtime);
    runtime
        .install_fake_active_capture_for_test(true, Duration::from_millis(25))
        .expect("fake active capture");
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );

    for offset in 0..DEFAULT_CAPTURE_MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES {
        let sample = i16::try_from(offset).unwrap_or_default();
        let segment = writer
            .write_pcm_i16_segment(&session_id, &[sample, 100, -100, sample])
            .expect("write captured segment");
        let result = runtime
            .transcribe_captured_segment(segment, Some("capture".to_string()), true)
            .await;
        assert!(result.is_err());
    }

    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.state, CaptureControllerState::Failed);
    assert_eq!(health.status, CaptureHealthStatus::Failed);
    assert!(!health.active_handle_present);
    assert_eq!(
        health.metrics.segment_transcription_failures_consecutive,
        DEFAULT_CAPTURE_MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES as u64
    );
    assert_eq!(
        runtime
            .get_active_session()
            .expect("active session")
            .expect("session")
            .capture_backend_status
            .as_deref(),
        Some("capture failed: segment_transcription_failure_threshold")
    );
}

#[tokio::test]
async fn successful_segment_transcription_resets_consecutive_failure_count() {
    let remaining_failures = Arc::new(AtomicUsize::new(1));
    let (runtime, root) = runtime_with_fake_transcriber_mode(
        "meeting_segment_failure_reset",
        FakeTranscriberMode::FailNTimesThenReturn {
            remaining_failures,
            text: "recovered transcript".to_string(),
        },
    );
    let session_id = start_manual_session(&runtime);
    runtime
        .install_fake_active_capture_for_test(true, Duration::from_millis(25))
        .expect("fake active capture");
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );

    let first = writer
        .write_pcm_i16_segment(&session_id, &[0, 100, -100, 0])
        .expect("write first segment");
    assert!(runtime
        .transcribe_captured_segment(first, Some("capture".to_string()), true)
        .await
        .is_err());
    assert_eq!(
        runtime
            .capture_health()
            .expect("capture health")
            .metrics
            .segment_transcription_failures_consecutive,
        1
    );

    let second = writer
        .write_pcm_i16_segment(&session_id, &[1, 101, -101, 1])
        .expect("write second segment");
    runtime
        .transcribe_captured_segment(second, Some("capture".to_string()), true)
        .await
        .expect("second segment transcribes");

    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.metrics.segment_transcription_failures_total, 1);
    assert_eq!(health.metrics.segment_transcription_failures_consecutive, 0);
    assert_eq!(health.metrics.segments_transcribed, 1);
    assert_eq!(
        runtime.get_active_state().expect("state").transcript.len(),
        1
    );
}

#[test]
fn consent_revocation_still_stops_capture_immediately() {
    let (runtime, _root) =
        runtime_with_fake_transcriber("meeting_consent_revoke_still_immediate", "unused");
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");
    runtime
        .install_fake_active_capture_for_test(true, Duration::from_millis(25))
        .expect("fake active capture");

    runtime.revoke_consent("teams").expect("revoke consent");

    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.status, CaptureHealthStatus::ConsentRevoked);
    assert!(!health.active_handle_present);
}

#[test]
fn clear_data_aborts_or_reports_when_capture_stop_times_out() {
    let root = temp_root("meeting_clear_data_stop_timeout");
    let runtime = MeetingRuntime::new(root.clone());
    let meeting_dir = root.join(".astra").join("meetings").join("old_session");
    std::fs::create_dir_all(&meeting_dir).expect("meeting dir");
    std::fs::write(meeting_dir.join("notes.json"), "{}").expect("meeting file");
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");
    runtime
        .install_fake_active_capture_for_test(false, Duration::from_millis(25))
        .expect("fake active capture");

    let result = runtime.clear_all_data(ClearMeetingDataRequest {
        scope: MeetingClearScope::All,
        confirmation_phrase: CLEAR_MEETING_DATA_CONFIRMATION_PHRASE.to_string(),
    });

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::ClearAbortedCaptureStopFailed {
            error_kind,
            ..
        }) if error_kind == "capture_stop_timed_out"
    ));
    assert!(meeting_dir.exists());
    let health = runtime.capture_health().expect("capture health");
    assert_eq!(health.status, CaptureHealthStatus::StopTimedOut);
    assert!(!health.active_handle_present);
    let session = runtime
        .get_active_session()
        .expect("active session")
        .expect("session remains for inspection");
    assert_eq!(
        session.capture_backend_status.as_deref(),
        Some("clear_data aborted: capture_stop_timed_out")
    );
}

#[test]
fn meeting_audit_summary_excludes_paths_filenames_transcripts_samples_devices() {
    let metrics = CaptureMetricsReporter::new();
    metrics.record_segment_written(52, 15_000);
    metrics.record_segment_transcription_failure("transcription_unavailable");
    let serialized = serde_json::to_string(&metrics.snapshot()).expect("metrics json");

    assert!(!serialized.contains("raw_audio"));
    assert!(!serialized.contains("samples"));
    assert!(!serialized.contains("device"));
    assert!(!serialized.contains(".wav"));
    assert!(!serialized.contains("sample.wav"));
    assert!(!serialized.contains("sensitive transcript"));
    assert!(!serialized.contains("C:/"));
    assert!(!serialized.contains("\\Users\\"));
    assert!(serialized.contains("transcription_unavailable"));
}

#[test]
fn backpressure_reject_newest_records_drop_metrics() {
    let root = temp_root("meeting_backpressure_reject_newest");
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );
    let first = writer
        .write_pcm_i16_segment("session_1", &[0, 100, -100, 0])
        .expect("first segment");
    let second = writer
        .write_pcm_i16_segment("session_1", &[1, 101, -101, 1])
        .expect("second segment");
    let metrics = CaptureMetricsReporter::new();
    let queue = CapturedSegmentQueue::new(1, CaptureOverflowPolicy::RejectNewest, metrics.clone());

    assert!(matches!(
        queue.try_send(first),
        CapturedSegmentEnqueueOutcome::Enqueued { depth: 1 }
    ));
    let outcome = queue.try_send(second);
    let dropped_path = match outcome {
        CapturedSegmentEnqueueOutcome::DroppedNewest { segment } => segment.path,
        other => panic!("expected newest drop, got {other:?}"),
    };
    let _ = std::fs::remove_file(dropped_path);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.queue_full_events, 1);
    assert_eq!(snapshot.segments_dropped, 1);
    assert!(snapshot.backpressure_active);
    assert_eq!(
        snapshot.last_overflow_policy_applied,
        Some(CaptureOverflowPolicy::RejectNewest)
    );
    let serialized = serde_json::to_string(&snapshot).expect("metrics json");
    assert!(!serialized.contains("session_1"));
    assert!(!serialized.contains(".wav"));
}

#[test]
fn backpressure_stop_capture_policy_stops_safely() {
    let root = temp_root("meeting_backpressure_stop_capture");
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );
    let first = writer
        .write_pcm_i16_segment("session_1", &[0, 100, -100, 0])
        .expect("first segment");
    let second = writer
        .write_pcm_i16_segment("session_1", &[1, 101, -101, 1])
        .expect("second segment");
    let metrics = CaptureMetricsReporter::new();
    let queue = CapturedSegmentQueue::new(1, CaptureOverflowPolicy::StopCapture, metrics.clone());

    assert!(matches!(
        queue.try_send(first),
        CapturedSegmentEnqueueOutcome::Enqueued { depth: 1 }
    ));
    let outcome = queue.try_send(second);
    let stopped_path = match outcome {
        CapturedSegmentEnqueueOutcome::StopCapture { segment } => segment.path,
        other => panic!("expected stop capture overflow, got {other:?}"),
    };
    let _ = std::fs::remove_file(stopped_path);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.queue_full_events, 1);
    assert_eq!(snapshot.segments_dropped, 1);
    assert!(snapshot.backpressure_active);
    assert_eq!(
        snapshot.last_overflow_policy_applied,
        Some(CaptureOverflowPolicy::StopCapture)
    );
    assert_eq!(
        snapshot.last_segment_status.as_deref(),
        Some("segment_dropped:stop_capture")
    );
}

#[test]
fn segment_defaults_are_coherent_and_report_effective_config() {
    let defaults = CapturePipelineConfig::default();
    let effective = defaults.effective();
    assert_eq!(
        defaults.chunk_duration_ms,
        DEFAULT_CAPTURE_SEGMENT_DURATION_MS
    );
    assert_eq!(
        defaults.max_segments_per_session,
        DEFAULT_CAPTURE_MAX_SEGMENTS_PER_SESSION
    );
    assert_eq!(
        defaults.max_consecutive_transcription_failures,
        DEFAULT_CAPTURE_MAX_CONSECUTIVE_TRANSCRIPTION_FAILURES
    );
    assert_eq!(
        effective.effective_segment_duration_ms,
        DEFAULT_CAPTURE_SEGMENT_DURATION_MS
    );
    assert!(!effective.duration_clamped);
    assert!(effective.estimated_max_session_duration_ms >= 2 * 60 * 60 * 1_000);

    let too_low = CapturePipelineConfig {
        chunk_duration_ms: 1_000,
        ..CapturePipelineConfig::default()
    }
    .effective();
    assert_eq!(too_low.effective_segment_duration_ms, 10_000);
    assert!(too_low.duration_clamped);

    let too_high = CapturePipelineConfig {
        chunk_duration_ms: 60_000,
        ..CapturePipelineConfig::default()
    }
    .effective();
    assert_eq!(too_high.effective_segment_duration_ms, 30_000);
    assert!(too_high.duration_clamped);
}

#[test]
fn audio_quality_downmix_f32_to_i16_mono_is_bounded() {
    let output = convert_interleaved_f32_to_i16_mono(&[1.5, 1.5, -1.5, -1.5], 2);

    assert_eq!(output.samples.len(), 2);
    assert_eq!(output.samples[0], i16::MAX);
    assert_eq!(output.samples[1], -i16::MAX);
    assert_eq!(output.clipped_samples, 4);
    assert!(output.peak_abs > 0);
}

#[test]
fn audio_quality_downmixes_stereo_to_mono_pcm16() {
    let output = downmix_interleaved_i16_to_mono(&[1000, -1000, 2000, 0], 2);

    assert_eq!(output.samples, vec![0, 1000]);
    assert_eq!(output.input_frames, 2);
    assert_eq!(output.output_frames, 2);
    assert_eq!(output.clipped_samples, 0);
    assert_eq!(output.normalization_gain_bps, 10_000);
}

#[test]
fn audio_quality_rms_vad_drops_pure_silence() {
    let config = CapturePipelineConfig::default();
    let silence = vec![0_i16; 16_000];

    let analysis = analyze_mono_i16_segment(&silence, 16_000, &config);

    assert!(!analysis.speech_detected);
    assert!(analysis.should_drop);
    assert_eq!(analysis.speech_ratio_bps, 0);
    assert_eq!(analysis.silence_ratio_bps, 10_000);
}

#[test]
fn vad_pure_silence_segment_is_dropped_before_stt() {
    let config = CapturePipelineConfig::default();
    let silence = vec![0_i16; 16_000];

    let analysis = analyze_mono_i16_segment(&silence, 16_000, &config);

    assert!(!analysis.speech_detected);
    assert!(analysis.should_drop);
    assert_eq!(analysis.speech_ratio_bps, 0);
    assert_eq!(analysis.silence_ratio_bps, 10_000);
}

#[test]
fn audio_quality_vad_keeps_short_valid_speech() {
    let config = CapturePipelineConfig::default();
    let mut speech_inside_silence = vec![0_i16; 16_000 * 15];
    speech_inside_silence
        .iter_mut()
        .take(1_600)
        .for_each(|sample| *sample = 3_000);

    let analysis = analyze_mono_i16_segment(&speech_inside_silence, 16_000, &config);

    assert!(analysis.speech_detected);
    assert!(!analysis.should_drop);
    assert!(analysis.speech_ratio_bps < config.vad_min_speech_ratio_bps);
}

#[test]
fn audio_quality_vad_drops_low_rms_noise() {
    let config = CapturePipelineConfig::default();
    let noise = vec![100_i16; 16_000];

    let analysis = analyze_mono_i16_segment(&noise, 16_000, &config);

    assert!(!analysis.speech_detected);
    assert!(analysis.should_drop);
}

#[test]
fn audio_quality_vad_keeps_clipped_short_speech() {
    let config = CapturePipelineConfig::default();
    let mut clipped_speech_inside_silence = vec![0_i16; 16_000 * 15];
    clipped_speech_inside_silence
        .iter_mut()
        .take(320)
        .for_each(|sample| *sample = i16::MAX);

    let analysis = analyze_mono_i16_segment(&clipped_speech_inside_silence, 16_000, &config);

    assert!(analysis.speech_detected);
    assert!(!analysis.should_drop);
}

#[test]
fn final_flush_keeps_short_speech() {
    let config = CapturePipelineConfig::default();
    let mut final_phrase = vec![0_i16; 16_000];
    final_phrase
        .iter_mut()
        .rev()
        .take(800)
        .for_each(|sample| *sample = 300);

    let normal = analyze_mono_i16_segment(&final_phrase, 16_000, &config);
    let final_flush = analyze_mono_i16_final_flush(&final_phrase, 16_000, &config);

    assert!(normal.should_drop);
    assert!(final_flush.speech_detected);
    assert!(!final_flush.should_drop);
}

#[test]
fn final_flush_drops_pure_silence() {
    let config = CapturePipelineConfig::default();
    let silence = vec![0_i16; 1_600];

    let analysis = analyze_mono_i16_final_flush(&silence, 16_000, &config);

    assert!(!analysis.speech_detected);
    assert!(analysis.should_drop);
}

#[test]
fn vad_speech_segment_is_kept_for_stt() {
    let config = CapturePipelineConfig::default();
    let speech = vec![3_000_i16; 16_000];

    let analysis = analyze_mono_i16_segment(&speech, 16_000, &config);

    assert!(analysis.speech_detected);
    assert!(!analysis.should_drop);
    assert!(analysis.speech_ratio_bps >= config.vad_min_speech_ratio_bps);
}

#[test]
fn meeting_debug_metrics_do_not_expose_audio_samples() {
    let metrics = CaptureMetricsReporter::new();
    metrics.record_wasapi_endpoint_acquired();
    metrics.record_wasapi_mix_format(48_000, 2, "float32");
    metrics.record_wasapi_buffer_frame_count(960);
    metrics.record_wasapi_stream_initialized();
    metrics.record_wasapi_stream_started();
    metrics.record_wasapi_packet(480);
    metrics.record_audio_conversion(480, 0, 1_000, 300, 10_000);
    metrics.record_backend_error(
        "capture_start_failed",
        "activate audio client failed: 0x88890004",
    );

    let snapshot = metrics.snapshot();
    assert!(snapshot.wasapi_endpoint_acquired);
    assert!(snapshot.wasapi_mix_format_detected);
    assert_eq!(snapshot.wasapi_sample_rate, Some(48_000));
    assert_eq!(snapshot.wasapi_channel_count, Some(2));
    assert_eq!(snapshot.wasapi_sample_format.as_deref(), Some("float32"));
    assert_eq!(snapshot.frames_captured, 480);
    assert_eq!(snapshot.frames_converted, 480);

    let serialized = serde_json::to_string(&snapshot).expect("metrics json");
    assert!(!serialized.contains("device friendly"));
    assert!(!serialized.contains("Headphones"));
    assert!(!serialized.contains(".wav"));
    assert!(!serialized.contains("C:/"));
    assert!(!serialized.contains("\\Users\\"));
}

#[test]
fn meeting_vad_metrics_are_redacted_and_count_based() {
    let metrics = CaptureMetricsReporter::new();
    metrics.record_vad_analysis(4, 746, 53, 9_947);
    metrics.record_silence_segment_dropped(16_000, 0, 50, 0, 10_000);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.vad_speech_frames, 4);
    assert_eq!(snapshot.vad_silence_frames, 796);
    assert_eq!(snapshot.last_speech_ratio_bps, 0);
    assert_eq!(snapshot.last_silence_ratio_bps, 10_000);
    assert_eq!(snapshot.dropped_silence_segments, 1);

    let serialized = serde_json::to_string(&snapshot).expect("metrics json");
    assert!(serialized.contains("vad_speech_frames"));
    assert!(serialized.contains("last_speech_ratio_bps"));
    assert!(!serialized.contains("samples"));
    assert!(!serialized.contains("raw_audio"));
    assert!(!serialized.contains(".wav"));
    assert!(!serialized.contains("C:/"));
    assert!(!serialized.contains("\\Users\\"));
}

#[tokio::test]
async fn captured_segment_uses_non_diarization_speaker_label() {
    let (runtime, root) =
        runtime_with_fake_transcriber("meeting_unknown_speaker_turn", "captured transcript");
    let session_id = start_manual_session(&runtime);
    let writer = SegmentWriter::new(
        root.join(".astra").join("meetings"),
        SegmentWriterConfig::default(),
    );
    let segment = writer
        .write_pcm_i16_segment(&session_id, &[0, 100, -100, 0])
        .expect("write captured segment");

    runtime
        .transcribe_captured_segment(segment, None, true)
        .await
        .expect("captured segment transcription");

    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.transcript[0].speaker, "Speaker 1");
    assert_eq!(
        state.transcript[0].speaker_id.as_deref(),
        Some(REMOTE_SPEAKER_1_ID)
    );
    assert_eq!(
        state.transcript[0].speaker_attribution_method,
        SpeakerAttributionMethod::SourceDefault
    );
    assert_ne!(state.transcript[0].speaker, "speaker_1");
}

#[test]
fn diarization_and_live_summary_are_truthfully_unavailable_without_real_backends() {
    let runtime = MeetingRuntime::new(temp_root("meeting_truthful_live_status"));
    let capabilities = runtime.live_capabilities().expect("capabilities");

    assert_eq!(
        capabilities.diarization.state,
        MeetingCapabilityState::Unavailable
    );
    assert_eq!(
        capabilities.live_summarization.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(capabilities
        .diarization
        .reason
        .as_deref()
        .is_some_and(|reason| reason.contains("No tested diarization backend")));
    assert!(capabilities
        .live_summarization
        .reason
        .as_deref()
        .is_some_and(|reason| reason.contains("No governed MeetingRuntime live-summary adapter")));
}

#[test]
fn ui_state_or_dto_distinguishes_backend_available_from_permission_ready() {
    let source = include_str!("../../src/components/MeetingDebugPanel.tsx");

    assert!(source.contains("type MeetingStartReadiness"));
    assert!(source.contains("canRequestStart"));
    assert!(source.contains("requiresApprovalOrConfirmation"));
    assert!(source.contains("hardBlockers"));
    assert!(source.contains("softGates"));
    assert!(source.contains("backendAvailable"));
    assert!(source.contains("audioPermissionReady"));
    assert!(source.contains("segmentPermissionReady"));
    assert!(source.contains("approvalRequired"));
    assert!(source.contains("Recording ready for governed start"));
    assert!(source.contains("Confirmation/audit will occur when you click Start."));
    assert!(source.contains(
        "No pending approval exists because the request is blocked before approval creation."
    ));
    assert!(source.contains("toolReady(audioCaptureTool)"));
    assert!(source.contains("toolReady(segmentTranscriptionTool)"));
    assert!(source.contains("Hardware validation"));
    assert!(source.contains("Silence dropped"));
    assert!(source.contains("Clipping count"));
    assert!(source.contains("Diarization:"));
    assert!(source.contains("Summary:"));
}

#[test]
fn start_meeting_session_uses_confirmed_preflight_for_approval_gated_capture() {
    let source = include_str!("../src/lib.rs");

    assert!(source.contains("confirmed_meeting_capability_permission_check"));
    assert!(source.contains("execute_confirmed_governed_direct_action"));
    assert!(source.contains("meeting_control_center_explicit_start"));
    assert!(source.contains("\"metadata_only\": true"));
    assert!(source.contains("\"raw_audio\": \"not_included\""));
    assert!(source.contains("\"transcript_text\": \"not_included\""));
    assert!(!source.contains("unavailable_meeting_capability_permission_check"));
}

#[test]
fn capture_audit_does_not_include_paths_filenames_or_samples() {
    let metrics = CaptureMetricsReporter::new();
    metrics.record_segment_written(52, 15_000);
    metrics.record_queue_full(CaptureOverflowPolicy::RejectNewest);
    metrics.record_segment_dropped(CaptureOverflowPolicy::RejectNewest);
    let serialized = serde_json::to_string(&metrics.snapshot()).expect("metrics json");

    assert!(!serialized.contains("raw_audio"));
    assert!(!serialized.contains("samples"));
    assert!(!serialized.contains(".wav"));
    assert!(!serialized.contains("C:/"));
    assert!(!serialized.contains("\\Users\\"));
    assert!(serialized.contains("segment_dropped:reject_newest"));
    assert_eq!(metrics.snapshot().segments_written, 1);
}

#[tokio::test]
async fn transcribe_file_does_not_audit_raw_transcript_text() {
    let (runtime, root) =
        runtime_with_fake_transcriber("meeting_file_result_redacted", "sensitive transcript text");
    let audio = write_test_audio(&root, "sample.wav");
    start_manual_session(&runtime);

    let result = runtime
        .transcribe_audio_file(file_request(&audio))
        .await
        .expect("file transcription");
    let serialized = serde_json::to_string(&result).expect("result json");

    assert!(!serialized.contains("sensitive transcript text"));
    assert!(!serialized.contains("meeting_file_result_redacted"));
    assert_eq!(
        result.text_length,
        "sensitive transcript text".chars().count()
    );
}

#[tokio::test]
async fn file_transcription_result_reports_cleanup_success_when_cleanup_works() {
    let (runtime, root) = runtime_with_fake_transcriber("meeting_file_cleanup", "hello");
    let audio = write_test_audio(&root, "sample.wav");
    let session_id = start_manual_session(&runtime);
    let mut request = file_request(&audio);
    request.cleanup_after_transcription = true;

    let result = runtime
        .transcribe_audio_file(request)
        .await
        .expect("file transcription");

    assert!(result.transcript_added);
    assert!(result.cleanup_requested);
    assert!(result.cleanup_performed);
    assert!(result.cleanup_error.is_none());

    let segments_dir = root
        .join(".astra")
        .join("meetings")
        .join(session_id)
        .join("segments");
    let remaining = std::fs::read_dir(&segments_dir)
        .expect("segments dir")
        .count();
    assert_eq!(remaining, 0);
}

#[tokio::test]
async fn cleanup_failure_after_success_returns_warning_not_error() {
    let (runtime, root) = runtime_with_fake_transcriber_mode(
        "meeting_file_cleanup_warning",
        FakeTranscriberMode::ReplaceManagedPathWithDirectory("hello".to_string()),
    );
    let audio = write_test_audio(&root, "sample.wav");
    start_manual_session(&runtime);
    let mut request = file_request(&audio);
    request.cleanup_after_transcription = true;

    let result = runtime
        .transcribe_audio_file(request)
        .await
        .expect("cleanup warning should not fail successful transcription");

    assert!(result.transcript_added);
    assert!(result.cleanup_requested);
    assert!(!result.cleanup_performed);
    assert!(result.cleanup_error.is_some());
    let state = runtime.get_active_state().expect("state");
    assert_eq!(state.transcript.len(), 1);
}

#[tokio::test]
async fn managed_copy_is_best_effort_cleaned_when_stt_fails_and_cleanup_was_requested() {
    let (runtime, root) = runtime_with_fake_transcriber_mode(
        "meeting_file_stt_fail_cleanup",
        FakeTranscriberMode::Fail,
    );
    let audio = write_test_audio(&root, "sample.wav");
    let session_id = start_manual_session(&runtime);
    let mut request = file_request(&audio);
    request.cleanup_after_transcription = true;

    let result = runtime.transcribe_audio_file(request).await;

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::TranscriptionUnavailable { .. })
    ));
    let segments_dir = root
        .join(".astra")
        .join("meetings")
        .join(session_id)
        .join("segments");
    let remaining = std::fs::read_dir(&segments_dir)
        .expect("segments dir")
        .count();
    assert_eq!(remaining, 0);
}

#[tokio::test]
async fn cleanup_failure_on_failure_path_is_reported_or_audited() {
    let (runtime, root) = runtime_with_fake_transcriber_mode(
        "meeting_file_stt_fail_cleanup_warning",
        FakeTranscriberMode::ReplaceManagedPathWithDirectoryThenFail,
    );
    let audio = write_test_audio(&root, "sample.wav");
    start_manual_session(&runtime);
    let mut request = file_request(&audio);
    request.cleanup_after_transcription = true;

    let result = runtime.transcribe_audio_file(request).await;

    match result {
        Err(MeetingRuntimeError::TranscriptionFailedWithCleanupWarning {
            reason,
            cleanup_requested,
            cleanup_performed,
            cleanup_error,
            managed_path_redacted,
        }) => {
            assert!(reason.contains("fake STT failed"));
            assert!(cleanup_requested);
            assert!(!cleanup_performed);
            assert!(cleanup_error.is_some());
            assert!(managed_path_redacted);
        }
        other => panic!("expected cleanup warning error, got {other:?}"),
    }
}

#[tokio::test]
async fn full_file_path_is_not_returned_in_file_transcription_result() {
    let (runtime, root) =
        runtime_with_fake_transcriber("meeting_file_path_redacted", "safe transcript");
    let audio = write_test_audio(&root, "sample.wav");
    start_manual_session(&runtime);

    let result = runtime
        .transcribe_audio_file(file_request(&audio))
        .await
        .expect("file transcription");
    let serialized = serde_json::to_string(&result).expect("result json");

    assert!(!serialized.contains("meeting_file_path_redacted"));
    assert!(!serialized.contains("sample.wav"));
    assert!(!serialized.contains("safe transcript"));
    assert_eq!(result.audio_file_extension, "wav");
    assert!(result.source_audio_path_redacted);
    assert!(result.managed_audio_path_redacted);
}

#[test]
fn manual_transcript_still_works_without_file_stt_permission() {
    let runtime = MeetingRuntime::new(temp_root("meeting_manual_without_file_stt_permission"));
    runtime.grant_consent("teams").expect("grant consent");
    runtime
        .start_session("teams".to_string(), manual_config())
        .expect("manual session start");

    runtime
        .add_transcript(transcript())
        .expect("manual transcript");

    assert_eq!(
        runtime.get_active_state().expect("state").transcript.len(),
        1
    );
}

#[test]
fn meeting_transcription_live_remains_unavailable() {
    let runtime = MeetingRuntime::new(temp_root("meeting_live_unavailable_exact"));

    let capabilities = runtime.live_capabilities().expect("capabilities");

    assert!(!capabilities.live_transcription.available);
    assert_eq!(
        capabilities.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert_eq!(
        capabilities.stt_adapter.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.stt_adapter.chunk_streaming_supported);
}

#[test]
fn file_transcription_capability_is_distinct_from_live_and_chunk_capabilities() {
    let (runtime, _root) =
        runtime_with_fake_transcriber("meeting_file_capability_distinct", "hello");

    let capabilities = runtime.live_capabilities().expect("capabilities");

    assert_eq!(
        capabilities.stt_adapter.file_transcription.state,
        MeetingCapabilityState::Ready
    );
    assert!(capabilities.stt_adapter.file_transcription.available);
    assert_eq!(capabilities.stt_adapter.reason, None);
    assert_eq!(
        capabilities.stt_adapter.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.stt_adapter.live_transcription.available);
    assert_eq!(
        capabilities.stt_adapter.chunk_streaming.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.stt_adapter.chunk_streaming.available);
    assert_eq!(
        capabilities.stt_adapter.existing_boundary,
        "SttClient::transcribe(Path)"
    );
    assert_eq!(
        capabilities.windows_wasapi_capture.state,
        if cfg!(target_os = "windows") {
            MeetingCapabilityState::Ready
        } else {
            MeetingCapabilityState::Unavailable
        }
    );
    assert_eq!(
        capabilities.live_segment_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert_eq!(
        capabilities.live_streaming_stt.state,
        MeetingCapabilityState::Unavailable
    );
    assert_eq!(
        capabilities.chunk_streaming.state,
        MeetingCapabilityState::Unavailable
    );
}

#[test]
fn stt_status_file_ready_has_no_top_level_unsupported_reason() {
    let (runtime, _root) = runtime_with_fake_transcriber("meeting_file_status_ready", "hello");

    let status = runtime
        .live_capabilities()
        .expect("capabilities")
        .stt_adapter;

    assert_eq!(status.state, MeetingCapabilityState::Ready);
    assert_eq!(
        status.file_transcription.state,
        MeetingCapabilityState::Ready
    );
    assert!(status.file_transcription.available);
    assert!(status.reason.is_none());
}

#[test]
fn stt_status_live_and_chunk_remain_unavailable_with_specific_reasons() {
    let (runtime, _root) = runtime_with_fake_transcriber("meeting_live_chunk_status", "hello");

    let status = runtime
        .live_capabilities()
        .expect("capabilities")
        .stt_adapter;

    assert_eq!(
        status.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert_eq!(
        status.chunk_streaming.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(status.live_transcription.reason.is_some());
    assert!(status.chunk_streaming.reason.is_some());
    assert!(status.reason.is_none());
}

#[test]
fn live_transcription_and_followup_remain_unavailable_without_live_backend() {
    let runtime = MeetingRuntime::new(temp_root("meeting_live_permissions_not_default"));

    let capabilities = runtime.live_capabilities().expect("capabilities");

    assert_eq!(
        capabilities.audio_capture.available,
        cfg!(target_os = "windows")
    );
    assert_eq!(
        capabilities.audio_capture.state,
        if cfg!(target_os = "windows") {
            MeetingCapabilityState::Ready
        } else {
            MeetingCapabilityState::Unavailable
        }
    );
    assert!(!capabilities.live_transcription.available);
    assert_eq!(
        capabilities.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.follow_up.available);
    assert_eq!(
        capabilities.follow_up.state,
        MeetingCapabilityState::Unavailable
    );
}

#[test]
fn call_detector_no_false_positive_on_plain_browser() {
    let result = CallDetector::detect_from_process_names(&["chrome.exe"])
        .expect("browser process should produce a weak detection");

    assert!(!result.is_active_call);
    assert_eq!(result.detection_state, CallDetectionState::Detected);
    assert!(result.window_title.is_empty());
}

#[test]
fn exported_meeting_uses_real_sha256() {
    assert_eq!(
        NoteOrganizer::sha256_hex("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn placeholder_transcription_is_not_returned_as_real_text() {
    let mut stream = TranscriptionStream::new("local".to_string(), None);
    let frame = AudioFrame {
        timestamp: Utc::now(),
        samples: vec![0.1, 0.2],
        device_id: "test".to_string(),
    };

    let result = stream.process_segment(&[frame], None);

    assert!(result.is_err());
    assert!(stream.get_transcript().is_empty());
}

#[test]
fn stt_adapter_does_not_emit_placeholder_transcript() {
    let adapter = ExistingSttClientMeetingAdapter::new();
    let segment = MeetingAudioSegment {
        session_id: Some("meeting".to_string()),
        chunks: vec![MeetingAudioChunk {
            sample_rate: 16_000,
            channels: 1,
            format: AudioSampleFormat::F32Pcm,
            monotonic_timestamp_ms: 1,
            sequence_number: 1,
            duration_ms: 1_000,
            source_backend: CaptureBackend::Wasapi,
            transcript_source: TranscriptSource::SystemAudio,
            byte_length: 64_000,
            frame_count: 16_000,
        }],
        total_duration_ms: 1_000,
        source_backend: CaptureBackend::Wasapi,
        transcript_source: TranscriptSource::SystemAudio,
        contains_raw_audio: false,
    };

    let result = adapter.transcribe_segment(&segment, Some("speaker".to_string()));

    assert!(matches!(
        result,
        Err(MeetingRuntimeError::TranscriptionUnavailable { .. })
    ));
}

#[test]
fn stt_adapter_uses_existing_stt_boundary_or_reports_unsupported() {
    let adapter = ExistingSttClientMeetingAdapter::new();
    let status = MeetingSttEngine::status(&adapter);

    assert_eq!(status.state, MeetingCapabilityState::Unavailable);
    assert_eq!(status.existing_boundary, "SttClient::transcribe(Path)");
    assert_eq!(
        status.file_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert_eq!(
        status.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert_eq!(
        status.chunk_streaming.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!status.chunk_streaming_supported);
    assert!(!status.emits_placeholder_transcripts);
}

#[test]
fn meeting_capabilities_distinguish_file_vs_wasapi_vs_streaming() {
    let runtime = MeetingRuntime::new(temp_root("meeting_live_capabilities"));

    let capabilities = runtime.live_capabilities().expect("capabilities");

    assert!(capabilities.manual_session.available);
    assert_eq!(
        capabilities.manual_session.state,
        MeetingCapabilityState::Ready
    );
    assert_eq!(
        capabilities.audio_capture.available,
        cfg!(target_os = "windows")
    );
    assert_eq!(
        capabilities.audio_capture.state,
        if cfg!(target_os = "windows") {
            MeetingCapabilityState::Ready
        } else {
            MeetingCapabilityState::Unavailable
        }
    );
    assert_eq!(
        capabilities.windows_wasapi_capture.available,
        cfg!(target_os = "windows")
    );
    assert_eq!(
        capabilities.windows_wasapi_capture.state,
        if cfg!(target_os = "windows") {
            MeetingCapabilityState::Ready
        } else {
            MeetingCapabilityState::Unavailable
        }
    );
    assert!(!capabilities.live_transcription.available);
    assert_eq!(
        capabilities.live_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.live_segment_transcription.available);
    assert_eq!(
        capabilities.live_segment_transcription.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.live_streaming_stt.available);
    assert_eq!(
        capabilities.live_streaming_stt.state,
        MeetingCapabilityState::Unavailable
    );
    assert!(!capabilities.chunk_streaming.available);
    assert_eq!(
        capabilities.chunk_streaming.state,
        MeetingCapabilityState::Unavailable
    );
}

#[test]
fn follow_up_sender_is_not_supported() {
    let mut sender = FollowUpSender::new(
        "smtp.example.test".to_string(),
        587,
        "astra".to_string(),
        "astra@example.test".to_string(),
    );
    let exported = ExportedMeeting {
        session_id: "session".to_string(),
        platform: "teams".to_string(),
        started_at: Utc::now(),
        ended_at: Utc::now(),
        participants: Vec::new(),
        transcript: Vec::new(),
        summary: Vec::new(),
        action_items: Vec::new(),
        decisions: Vec::new(),
        notes: Vec::new(),
        intelligence: None,
        metadata: serde_json::json!({}),
    };

    let result = sender.send(&exported, vec!["user@example.test".to_string()]);

    assert!(result.is_err());
    assert!(sender.send_failed);
}
