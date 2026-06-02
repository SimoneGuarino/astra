use serde::{Deserialize, Serialize};

use crate::vad::VadFrameSnapshot;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatStartRequest {
    #[serde(default)]
    pub client_request_id: Option<String>,
    pub message: String,
    #[serde(default)]
    pub input_modality: AssistantInputModality,
    #[serde(default)]
    pub audio_response: AssistantAudioResponsePolicy,
    #[serde(default)]
    pub deep_search: AssistantDeepSearchOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantDeepSearchOptions {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub seed_urls: Vec<String>,
    #[serde(default)]
    pub enable_web_discovery: Option<bool>,
    #[serde(default)]
    pub search_providers: Vec<String>,
    #[serde(default)]
    pub include_general_web: Option<bool>,
    #[serde(default)]
    pub include_academic_sources: Option<bool>,
    #[serde(default)]
    pub document_ingestion: Option<bool>,
    #[serde(default)]
    pub prefer_academic_landing_pages: Option<bool>,
    #[serde(default)]
    pub enable_pdf_text_extraction: Option<bool>,
    #[serde(default)]
    pub max_discovery_results_per_provider: Option<usize>,
    #[serde(default)]
    pub max_discovered_sources: Option<usize>,
    #[serde(default)]
    pub initial_query_count: Option<usize>,
    #[serde(default)]
    pub autonomous_loop: Option<bool>,
    #[serde(default)]
    pub max_research_passes: Option<usize>,
    #[serde(default)]
    pub min_research_passes: Option<usize>,
    #[serde(default)]
    pub max_sources_per_pass: Option<usize>,
    #[serde(default)]
    pub min_new_information_gain: Option<f32>,
    #[serde(default)]
    pub min_coverage_score: Option<f32>,
    #[serde(default)]
    pub min_supported_claim_ratio: Option<f32>,
    #[serde(default)]
    pub enable_claim_graph: Option<bool>,
    #[serde(default)]
    pub min_independent_sources_for_claim: Option<usize>,
    #[serde(default)]
    pub enable_contradiction_detection: Option<bool>,
    #[serde(default)]
    pub enable_memory_promotion_policy: Option<bool>,
    #[serde(default)]
    pub auto_promote_supported_claims: Option<bool>,
    #[serde(default)]
    pub require_user_confirmation_for_system_verified: Option<bool>,
    #[serde(default)]
    pub min_promotion_confidence: Option<f32>,
    #[serde(default)]
    pub min_promotion_independent_sources: Option<usize>,
    #[serde(default)]
    pub enable_source_reliability_scoring: Option<bool>,
    #[serde(default)]
    pub min_reliable_source_score_for_promotion: Option<f32>,
    #[serde(default)]
    pub allowed_domains: Vec<String>,
    #[serde(default)]
    pub blocked_domains: Vec<String>,
    #[serde(default)]
    pub max_sources: Option<usize>,
    #[serde(default)]
    pub require_cross_source_verification: bool,
}

impl Default for AssistantDeepSearchOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            seed_urls: Vec::new(),
            enable_web_discovery: Some(true),
            search_providers: Vec::new(),
            include_general_web: Some(true),
            include_academic_sources: Some(true),
            document_ingestion: Some(true),
            prefer_academic_landing_pages: Some(true),
            enable_pdf_text_extraction: Some(true),
            max_discovery_results_per_provider: Some(10),
            max_discovered_sources: Some(192),
            initial_query_count: Some(6),
            autonomous_loop: Some(true),
            max_research_passes: Some(5),
            min_research_passes: Some(2),
            max_sources_per_pass: Some(8),
            min_new_information_gain: Some(0.08),
            min_coverage_score: Some(0.66),
            min_supported_claim_ratio: Some(0.55),
            enable_claim_graph: Some(true),
            min_independent_sources_for_claim: Some(2),
            enable_contradiction_detection: Some(true),
            enable_memory_promotion_policy: Some(true),
            auto_promote_supported_claims: Some(true),
            require_user_confirmation_for_system_verified: Some(true),
            min_promotion_confidence: Some(0.62),
            min_promotion_independent_sources: Some(2),
            enable_source_reliability_scoring: Some(true),
            min_reliable_source_score_for_promotion: Some(0.50),
            allowed_domains: Vec::new(),
            blocked_domains: Vec::new(),
            max_sources: None,
            require_cross_source_verification: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartChatResponse {
    pub request_id: String,
    pub model: String,
    pub audio_response_enabled: bool,
    pub deep_search_enabled: bool,
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
    pub deep_search_enabled: bool,
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
        assert!(!request.deep_search.enabled);
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
            deep_search_enabled: false,
        })
        .expect("json");

        assert_eq!(serialized["audio_response_enabled"], false);
    }
}
