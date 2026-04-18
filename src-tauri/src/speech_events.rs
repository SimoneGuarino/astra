use serde::{Deserialize, Serialize};

use crate::vad::VadFrameSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStartRequest {
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartChatResponse {
    pub request_id: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantRequestStartedEvent {
    pub request_id: String,
    pub model: String,
    pub source: String,
    pub user_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantRequestFinishedEvent {
    pub request_id: String,
    pub full_text: String,
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
