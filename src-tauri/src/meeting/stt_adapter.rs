//! Meeting STT adapter boundary.
//!
//! Astra already has a file-based `SttClient` used by voice sessions. Meeting
//! audio is expected to arrive as bounded chunks/segments in the future. Live
//! chunk transcription remains unsupported, but completed audio files can be
//! sent through the existing `SttClient::transcribe(Path)` boundary.

use super::types::{
    MeetingAudioSegment, MeetingCapabilityReadiness, MeetingCapabilityState, MeetingRuntimeError,
    MeetingSttAdapterStatus, TranscriptEntry,
};
use crate::stt_client::{SttClient, SttClientError};
use std::{future::Future, path::Path, pin::Pin};

const EXISTING_STT_BOUNDARY: &str = "SttClient::transcribe(Path)";
const UNSUPPORTED_REASON: &str =
    "Existing SttClient accepts completed audio file paths; meeting chunk streaming is not wired yet";
const MISSING_STT_CLIENT_REASON: &str =
    "Meeting file transcription requires the existing SttClient to be attached";

pub type MeetingFileTranscriptionFuture<'a> =
    Pin<Box<dyn Future<Output = Result<String, MeetingRuntimeError>> + Send + 'a>>;

pub trait MeetingFileTranscriber: Send + Sync {
    fn status(&self) -> MeetingSttAdapterStatus;

    fn request_warm_up(&self) -> Result<(), MeetingRuntimeError> {
        Ok(())
    }

    fn transcribe_file<'a>(&'a self, audio_path: &'a Path) -> MeetingFileTranscriptionFuture<'a>;

    fn boundary(&self) -> &'static str {
        EXISTING_STT_BOUNDARY
    }
}

pub trait MeetingSttEngine {
    fn status(&self) -> MeetingSttAdapterStatus;

    fn transcribe_segment(
        &self,
        segment: &MeetingAudioSegment,
        speaker: Option<String>,
    ) -> Result<Option<TranscriptEntry>, MeetingRuntimeError>;
}

#[derive(Default, Clone)]
pub struct ExistingSttClientMeetingAdapter {
    stt_client: Option<SttClient>,
}

impl ExistingSttClientMeetingAdapter {
    pub fn new() -> Self {
        Self { stt_client: None }
    }

    pub fn with_stt_client(stt_client: SttClient) -> Self {
        Self {
            stt_client: Some(stt_client),
        }
    }

    pub fn unsupported_reason(&self) -> &'static str {
        UNSUPPORTED_REASON
    }
}

impl MeetingFileTranscriber for ExistingSttClientMeetingAdapter {
    fn status(&self) -> MeetingSttAdapterStatus {
        let file_available = self.stt_client.is_some();
        let file_state = if file_available {
            MeetingCapabilityState::Ready
        } else {
            MeetingCapabilityState::Unavailable
        };
        let file_reason = if file_available {
            None
        } else {
            Some(MISSING_STT_CLIENT_REASON.to_string())
        };

        MeetingSttAdapterStatus {
            state: file_state.clone(),
            existing_boundary: EXISTING_STT_BOUNDARY.to_string(),
            file_transcription: readiness(
                "meeting.transcription.file",
                file_available,
                file_state,
                file_reason,
            ),
            live_transcription: readiness(
                "meeting.transcription.live",
                false,
                MeetingCapabilityState::Unavailable,
                Some(self.unsupported_reason().to_string()),
            ),
            chunk_streaming: readiness(
                "meeting.transcription.chunk_streaming",
                false,
                MeetingCapabilityState::Unavailable,
                Some(self.unsupported_reason().to_string()),
            ),
            chunk_streaming_supported: false,
            emits_placeholder_transcripts: false,
            reason: if file_available {
                None
            } else {
                Some(MISSING_STT_CLIENT_REASON.to_string())
            },
        }
    }

    fn request_warm_up(&self) -> Result<(), MeetingRuntimeError> {
        let client = self
            .stt_client
            .as_ref()
            .ok_or_else(|| MeetingRuntimeError::SttUnavailable {
                reason: MISSING_STT_CLIENT_REASON.to_string(),
            })?;
        client.request_warm_up().map_err(|error| {
            MeetingRuntimeError::TranscriptionUnavailable {
                reason: sanitize_stt_error(&error),
            }
        })
    }

    fn transcribe_file<'a>(&'a self, audio_path: &'a Path) -> MeetingFileTranscriptionFuture<'a> {
        Box::pin(async move {
            let client =
                self.stt_client
                    .as_ref()
                    .ok_or_else(|| MeetingRuntimeError::SttUnavailable {
                        reason: MISSING_STT_CLIENT_REASON.to_string(),
                    })?;
            client.transcribe(audio_path).await.map_err(|error| {
                MeetingRuntimeError::TranscriptionUnavailable {
                    reason: sanitize_stt_error(&error),
                }
            })
        })
    }
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

impl MeetingSttEngine for ExistingSttClientMeetingAdapter {
    fn status(&self) -> MeetingSttAdapterStatus {
        <Self as MeetingFileTranscriber>::status(self)
    }

    fn transcribe_segment(
        &self,
        segment: &MeetingAudioSegment,
        _speaker: Option<String>,
    ) -> Result<Option<TranscriptEntry>, MeetingRuntimeError> {
        if segment.chunks.is_empty() {
            return Ok(None);
        }

        Err(MeetingRuntimeError::TranscriptionUnavailable {
            reason: self.unsupported_reason().to_string(),
        })
    }
}

fn sanitize_stt_error(error: &SttClientError) -> String {
    match error {
        SttClientError::Cancelled => "Existing STT request was cancelled".to_string(),
        SttClientError::Config(_) => "Existing STT worker is not configured".to_string(),
        SttClientError::Io(_) => "Existing STT worker I/O failed".to_string(),
        SttClientError::Protocol(_) => "Existing STT worker protocol failed".to_string(),
        SttClientError::Timeout => "Existing STT worker timed out".to_string(),
        SttClientError::WorkerFailed(_) => "Existing STT worker failed".to_string(),
        SttClientError::WorkerUnavailable => "Existing STT worker is unavailable".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{ExistingSttClientMeetingAdapter, MeetingSttEngine};
    use crate::meeting::types::{
        AudioSampleFormat, CaptureBackend, MeetingAudioChunk, MeetingAudioSegment,
        MeetingCapabilityState, MeetingRuntimeError, TranscriptSource,
    };

    fn segment() -> MeetingAudioSegment {
        MeetingAudioSegment {
            session_id: Some("meeting".to_string()),
            chunks: vec![MeetingAudioChunk {
                sample_rate: 16_000,
                channels: 1,
                format: AudioSampleFormat::F32Pcm,
                monotonic_timestamp_ms: 42,
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
        }
    }

    #[test]
    fn stt_adapter_does_not_emit_placeholder_transcripts() {
        let adapter = ExistingSttClientMeetingAdapter::new();

        let result = adapter.transcribe_segment(&segment(), Some("speaker".to_string()));

        assert!(matches!(
            result,
            Err(MeetingRuntimeError::TranscriptionUnavailable { .. })
        ));
    }

    #[test]
    fn stt_adapter_uses_existing_stt_boundary_or_reports_unsupported() {
        let adapter = ExistingSttClientMeetingAdapter::new();
        let status = adapter.status();

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
        assert_eq!(
            status.reason.as_deref(),
            Some(super::MISSING_STT_CLIENT_REASON)
        );
        assert_eq!(
            status.live_transcription.reason.as_deref(),
            Some(super::UNSUPPORTED_REASON)
        );
        assert_eq!(
            status.chunk_streaming.reason.as_deref(),
            Some(super::UNSUPPORTED_REASON)
        );
    }
}
