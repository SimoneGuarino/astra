use serde::{Deserialize, Serialize};

use crate::vad::VadFrameSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStartRequest {
    pub message: String,
    #[serde(default)]
    pub input_modality: AssistantInputModality,
    #[serde(default)]
    pub audio_response: AssistantAudioResponsePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartChatResponse {
    pub request_id: String,
    pub model: String,
    pub audio_response_enabled: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AssistantInputModality {
    #[default]
    Typed,
    Voice,
}

impl AssistantInputModality {
    pub fn as_source(self) -> &'static str {
        match self {
            Self::Typed => "typed",
            Self::Voice => "voice",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AssistantAudioResponsePolicy {
    #[default]
    Auto,
    Enabled,
    Disabled,
}

pub fn resolve_audio_response_enabled(
    input_modality: AssistantInputModality,
    audio_response: AssistantAudioResponsePolicy,
    allow_typed_audio: bool,
) -> bool {
    match audio_response {
        AssistantAudioResponsePolicy::Disabled => false,
        AssistantAudioResponsePolicy::Enabled => {
            input_modality == AssistantInputModality::Voice || allow_typed_audio
        }
        AssistantAudioResponsePolicy::Auto => input_modality == AssistantInputModality::Voice,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantRequestStartedEvent {
    pub request_id: String,
    pub model: String,
    pub source: String,
    pub user_message: Option<String>,
    pub audio_response_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantRequestFinishedEvent {
    pub request_id: String,
    pub full_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantRequestSettledEvent {
    pub request_id: String,
    pub had_tts_failures: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantErrorEvent {
    pub request_id: String,
    pub stage: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunkEvent {
    pub request_id: String,
    pub chunk: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSegmentQueuedEvent {
    pub request_id: String,
    pub segment_id: String,
    pub sequence: u32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSegmentReadyEvent {
    pub request_id: String,
    pub segment_id: String,
    pub sequence: u32,
    pub output_path: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSegmentFailedEvent {
    pub request_id: String,
    pub segment_id: String,
    pub sequence: u32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPlaybackEvent {
    pub request_id: String,
    pub segment_id: String,
    pub sequence: u32,
    pub output_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSessionCompletedRequest {
    pub request_id: String,
    #[serde(default)]
    pub had_failures: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTranscriptionRequest {
    pub audio_bytes: Vec<u8>,
    pub mime_type: String,
    pub auto_submit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTranscriptionStartedEvent {
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTranscriptionFinishedEvent {
    pub request_id: String,
    pub text: String,
    pub auto_submit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceTranscriptionResponse {
    pub request_id: String,
    pub text: String,
    pub auto_submit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSessionStartResponse {
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSessionAudioChunk {
    pub session_id: String,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VoiceSessionStateEvent {
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
    pub state: String,
    pub mode: String,
    pub reason: String,
    pub conversation_expires_in_ms: Option<u128>,
    pub vad: VadFrameSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSessionTranscriptEvent {
    pub session_id: String,
    pub turn_id: String,
    pub text: String,
    pub accepted: bool,
    pub reason: String,
    pub action: String,
    pub response_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantInterruptedEvent {
    pub request_id: Option<String>,
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_chat_defaults_to_audio_disabled() {
        let request: ChatStartRequest =
            serde_json::from_value(serde_json::json!({"message": "hello"})).expect("request");

        assert_eq!(request.input_modality, AssistantInputModality::Typed);
        assert_eq!(request.audio_response, AssistantAudioResponsePolicy::Auto);
        assert!(!resolve_audio_response_enabled(
            request.input_modality,
            request.audio_response,
            false
        ));
    }

    #[test]
    fn voice_chat_auto_enables_audio() {
        assert!(resolve_audio_response_enabled(
            AssistantInputModality::Voice,
            AssistantAudioResponsePolicy::Auto,
            false
        ));
    }

    #[test]
    fn start_chat_response_serializes_audio_policy() {
        let serialized = serde_json::to_value(StartChatResponse {
            request_id: "request".to_string(),
            model: "model".to_string(),
            audio_response_enabled: false,
        })
        .expect("json");

        assert_eq!(serialized["audio_response_enabled"], false);
    }
}
