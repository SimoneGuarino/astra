//! Transcript-backed meeting intelligence.
//!
//! This module treats transcript entries as the source of truth. Model output,
//! local model output must pass through the same schema and evidence validation
//! before it can be stored.

use super::{action_item_tracker::ActionItemTracker, decision_log::DecisionLog, types::*};
use chrono::Utc;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    env,
    future::Future,
    pin::Pin,
    time::{Duration, Instant},
};

const MAX_SUMMARY_BULLETS: usize = 6;
const MAX_DECISIONS: usize = 24;
const MAX_ACTION_ITEMS: usize = 40;
const MAX_OPEN_QUESTIONS: usize = 40;
const MAX_RISKS: usize = 30;
const MAX_TIMELINE_ITEMS: usize = 40;
const MAX_TECHNICAL_ITEMS: usize = 12;
const DEFAULT_OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MEETING_LLM_CANDIDATES: &str = "gpt-oss:20b,qwen3:14b,qwen3:8b,llama3.1:8b";
const DEFAULT_PROMPT_MAX_CHARS_TOTAL: usize = 24_000;
const DEFAULT_PROMPT_MAX_CHARS_PER_SEGMENT: usize = 900;
const DEFAULT_MEETING_LLM_TIMEOUT_SECS: u64 = 45;
const ITALIAN_LANGUAGE_MARKERS: &[&str] = &[
    "che",
    "di",
    "del",
    "della",
    "delle",
    "il",
    "lo",
    "la",
    "gli",
    "le",
    "un",
    "una",
    "per",
    "con",
    "non",
    "sono",
    "abbiamo",
    "deve",
    "dobbiamo",
    "quindi",
    "perche",
    "perché",
    "anche",
    "questo",
    "questa",
    "grazie",
    "ciao",
    "buongiorno",
    "allora",
    "durante",
    "sessione",
    "riunione",
    "riepilogo",
    "italiano",
    "italiana",
    "bozza",
    "punti",
    "principali",
    "emersi",
    "decisione",
    "azioni",
    "rischi",
    "errore",
];
const ENGLISH_LANGUAGE_MARKERS: &[&str] = &[
    "the", "and", "that", "this", "with", "for", "not", "we", "you", "should", "need", "because",
    "about", "then", "also", "thanks", "hello", "meeting", "please", "summary", "decided",
    "follow", "up", "risk", "issue", "build",
];
const TECHNICAL_DEBUGGING_MARKERS: &[&str] = &[
    "error",
    "errore",
    "bug",
    "test",
    "cargo",
    "npm",
    "build",
    "compile",
    "compilazione",
    "stack",
    "trace",
    "endpoint",
    "module",
    "modulo",
    "file",
    "runtime",
    "config",
    "env",
    "gpu",
    "vram",
    "cuda",
    "tts",
    "stt",
    "ollama",
];
const PLANNING_MARKERS: &[&str] = &[
    "roadmap",
    "milestone",
    "fase",
    "phase",
    "next",
    "step",
    "priorita",
    "priorità",
    "plan",
    "piano",
    "planning",
    "sprint",
    "prossimo",
    "prossima",
];
const DECISION_REVIEW_MARKERS: &[&str] = &[
    "decided",
    "deciso",
    "decisione",
    "confirmed",
    "confermato",
    "approved",
    "approvato",
    "rejected",
    "respinto",
    "proceed",
    "procedere",
    "accept",
    "accettare",
    "validate",
    "validare",
];
const SUPPORT_CALL_MARKERS: &[&str] = &[
    "issue",
    "problema",
    "ticket",
    "customer",
    "cliente",
    "support",
    "assistenza",
    "user",
    "utente",
    "report",
    "segnalazione",
    "reproduce",
    "riprodurre",
];
const WORK_MEETING_MARKERS: &[&str] = &[
    "meeting",
    "riunione",
    "sessione",
    "agenda",
    "follow",
    "stakeholder",
    "azione",
    "azioni",
    "action",
    "allineamento",
    "work",
    "lavoro",
];

#[derive(Debug, Clone)]
pub struct MeetingIntelligenceInput {
    pub session_id: String,
    pub transcript_entries: Vec<TranscriptEntry>,
    pub speakers: Vec<SpeakerLabel>,
    pub screen_contexts: Vec<MeetingScreenContext>,
    pub generation_options: MeetingIntelligenceGenerationOptions,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MeetingLlmPromptStats {
    pub input_segment_count: usize,
    pub input_truncated: bool,
    pub input_char_count: usize,
    pub max_segments: usize,
    pub max_chars_total: usize,
    pub max_chars_per_segment: usize,
    pub included_segment_ids: Vec<String>,
    pub detected_language: MeetingLanguage,
    pub language_confidence: f32,
    pub language_source: MeetingLanguageSource,
    pub session_type: MeetingSessionType,
    pub session_type_confidence: f32,
    pub session_type_source: MeetingSessionTypeSource,
}

#[derive(Debug, Clone)]
pub struct MeetingLlmPromptSegment {
    pub segment_id: String,
    pub speaker_id: Option<String>,
    pub speaker_label: String,
    pub source: TranscriptSource,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct MeetingLlmPromptInput {
    pub session_id: String,
    pub prompt: String,
    pub segments: Vec<MeetingLlmPromptSegment>,
    pub stats: MeetingLlmPromptStats,
}

#[derive(Debug, Clone)]
pub struct MeetingLlmRawOutput {
    pub raw_json: String,
    pub provider: String,
    pub model: String,
    pub stats: MeetingLlmPromptStats,
    pub endpoint: Option<String>,
    pub llm_generation_duration_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeetingLlmErrorKind {
    Unavailable,
    Timeout,
    Http,
    InvalidResponse,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeetingLlmError {
    pub kind: MeetingLlmErrorKind,
    pub message: String,
    pub provider: String,
    pub model: Option<String>,
    pub stats: MeetingLlmPromptStats,
    pub endpoint: Option<String>,
    pub llm_generation_duration_ms: Option<u64>,
}

impl MeetingLlmError {
    pub fn unavailable(message: impl Into<String>, stats: MeetingLlmPromptStats) -> Self {
        Self {
            kind: MeetingLlmErrorKind::Unavailable,
            message: bounded_error_message(message.into()),
            provider: "ollama".to_string(),
            model: None,
            stats,
            endpoint: None,
            llm_generation_duration_ms: None,
        }
    }

    fn with_kind(
        kind: MeetingLlmErrorKind,
        message: impl Into<String>,
        model: Option<String>,
        stats: MeetingLlmPromptStats,
        endpoint: Option<String>,
        llm_generation_duration_ms: Option<u64>,
    ) -> Self {
        Self {
            kind,
            message: bounded_error_message(message.into()),
            provider: "ollama".to_string(),
            model,
            stats,
            endpoint,
            llm_generation_duration_ms,
        }
    }

    pub(crate) fn reason_code(&self) -> String {
        match self.kind {
            MeetingLlmErrorKind::Unavailable => "local_llm_unavailable",
            MeetingLlmErrorKind::Timeout => "local_llm_timeout",
            MeetingLlmErrorKind::Http => "local_llm_http_error",
            MeetingLlmErrorKind::InvalidResponse => "local_llm_invalid_response",
        }
        .to_string()
    }
}

pub type MeetingLlmFuture<'a> =
    Pin<Box<dyn Future<Output = Result<MeetingLlmRawOutput, MeetingLlmError>> + Send + 'a>>;

pub trait MeetingIntelligenceLlm: Send + Sync {
    fn generate_intelligence_json<'a>(
        &'a self,
        input: MeetingLlmPromptInput,
    ) -> MeetingLlmFuture<'a>;
}

#[derive(Clone)]
pub struct OllamaMeetingIntelligenceLlm {
    client: Client,
    timeout: Duration,
    base_url: String,
    sanitized_endpoint: String,
}

impl OllamaMeetingIntelligenceLlm {
    pub fn new() -> Self {
        let timeout_secs = env::var("ASTRA_MEETING_INTELLIGENCE_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| (1..=300).contains(value))
            .unwrap_or(DEFAULT_MEETING_LLM_TIMEOUT_SECS);
        let base_url = configured_ollama_base_url();
        let sanitized_endpoint = sanitize_ollama_endpoint(&base_url);
        Self {
            client: Client::new(),
            timeout: Duration::from_secs(timeout_secs),
            base_url,
            sanitized_endpoint,
        }
    }

    async fn select_model(&self) -> Result<String, String> {
        if let Ok(model) = env::var("ASTRA_MEETING_INTELLIGENCE_MODEL") {
            let model = model.trim();
            if !model.is_empty() {
                return Ok(model.to_string());
            }
        }

        let installed = self.fetch_installed_models().await.unwrap_or_default();
        let candidates = env::var("ASTRA_MEETING_INTELLIGENCE_MODEL_CANDIDATES")
            .unwrap_or_else(|_| DEFAULT_MEETING_LLM_CANDIDATES.to_string())
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        select_first_available(&candidates, &installed)
            .or_else(|| candidates.first().cloned())
            .ok_or_else(|| "no meeting intelligence model candidates configured".to_string())
    }

    async fn fetch_installed_models(&self) -> Result<Vec<String>, String> {
        let response = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .map_err(|error| format!("Ollama tags request failed: {error}"))?;
        if !response.status().is_success() {
            return Err(format!("Ollama tags HTTP error: {}", response.status()));
        }
        let body: Value = response
            .json()
            .await
            .map_err(|error| format!("Ollama tags parse failed: {error}"))?;
        Ok(body
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.get("name").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .collect())
    }

    async fn call_model(
        &self,
        model: &str,
        prompt: &MeetingLlmPromptInput,
    ) -> Result<String, String> {
        let payload = json!({
            "model": model,
            "stream": false,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "num_ctx": 16384
            },
            "messages": [
                {"role": "system", "content": meeting_llm_system_prompt()},
                {"role": "user", "content": prompt.prompt}
            ]
        });

        let response = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&payload)
            .send()
            .await
            .map_err(|error| format!("Ollama meeting intelligence request failed: {error}"))?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!(
                "Ollama meeting intelligence HTTP error {status}: {}",
                bounded_error_message(body)
            ));
        }
        let body: Value = response.json().await.map_err(|error| {
            format!("Ollama meeting intelligence response parse failed: {error}")
        })?;
        let content = body
            .get("message")
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "Ollama meeting intelligence returned an empty response".to_string())?;
        extract_json_object(content)
            .map(ToOwned::to_owned)
            .ok_or_else(|| {
                "meeting intelligence response did not contain a JSON object".to_string()
            })
    }
}

impl Default for OllamaMeetingIntelligenceLlm {
    fn default() -> Self {
        Self::new()
    }
}

impl MeetingIntelligenceLlm for OllamaMeetingIntelligenceLlm {
    fn generate_intelligence_json<'a>(
        &'a self,
        input: MeetingLlmPromptInput,
    ) -> MeetingLlmFuture<'a> {
        Box::pin(async move {
            let started_at = Instant::now();
            let stats = input.stats.clone();
            let model = match self.select_model().await {
                Ok(model) => model,
                Err(error) => {
                    let mut llm_error = MeetingLlmError::unavailable(error, stats);
                    llm_error.endpoint = Some(self.sanitized_endpoint.clone());
                    llm_error.llm_generation_duration_ms = Some(elapsed_ms(started_at));
                    return Err(llm_error);
                }
            };
            let call = self.call_model(&model, &input);
            match tokio::time::timeout(self.timeout, call).await {
                Ok(Ok(raw_json)) => Ok(MeetingLlmRawOutput {
                    raw_json,
                    provider: "ollama".to_string(),
                    model,
                    stats: input.stats,
                    endpoint: Some(self.sanitized_endpoint.clone()),
                    llm_generation_duration_ms: Some(elapsed_ms(started_at)),
                }),
                Ok(Err(error)) => {
                    let kind = if error.contains("empty response")
                        || error.contains("did not contain a JSON object")
                        || error.contains("response parse failed")
                    {
                        MeetingLlmErrorKind::InvalidResponse
                    } else {
                        MeetingLlmErrorKind::Http
                    };
                    Err(MeetingLlmError::with_kind(
                        kind,
                        error,
                        Some(model),
                        input.stats,
                        Some(self.sanitized_endpoint.clone()),
                        Some(elapsed_ms(started_at)),
                    ))
                }
                Err(_) => Err(MeetingLlmError::with_kind(
                    MeetingLlmErrorKind::Timeout,
                    "Ollama meeting intelligence request timed out",
                    Some(model),
                    input.stats,
                    Some(self.sanitized_endpoint.clone()),
                    Some(elapsed_ms(started_at)),
                )),
            }
        })
    }
}

#[derive(Debug, Default)]
struct ValidationStats {
    invalid_evidence_ids: usize,
    rejected_artifact_count: usize,
    warnings: Vec<String>,
}

pub struct MeetingIntelligenceEngine;

impl MeetingIntelligenceEngine {
    pub fn generate(
        input: MeetingIntelligenceInput,
    ) -> Result<MeetingIntelligenceResult, MeetingRuntimeError> {
        if input.transcript_entries.is_empty() {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "meeting intelligence requires at least one transcript segment"
                    .to_string(),
            });
        }

        Ok(Self::rule_based_result(
            &bounded_input(input),
            ArtifactGenerator::RuleBased,
            MeetingIntelligenceStatus::Generated,
            None,
            Vec::new(),
        ))
    }

    pub fn generate_with_llm_json_or_rule_based(
        input: MeetingIntelligenceInput,
        llm_json: Option<&str>,
        model_name: Option<&str>,
    ) -> Result<MeetingIntelligenceResult, MeetingRuntimeError> {
        let output = llm_json.map(|raw_json| MeetingLlmRawOutput {
            raw_json: raw_json.to_string(),
            provider: "ollama".to_string(),
            model: model_name.unwrap_or("local").to_string(),
            stats: prompt_stats_for_input(&input),
            endpoint: None,
            llm_generation_duration_ms: None,
        });
        Self::generate_with_llm_output_or_rule_based(input, output, None)
    }

    pub fn generate_with_llm_output_or_rule_based(
        input: MeetingIntelligenceInput,
        llm_output: Option<MeetingLlmRawOutput>,
        llm_error: Option<MeetingLlmError>,
    ) -> Result<MeetingIntelligenceResult, MeetingRuntimeError> {
        if input.transcript_entries.is_empty() {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "meeting intelligence requires at least one transcript segment"
                    .to_string(),
            });
        }
        let input = bounded_input(input);

        if let Some(error) = llm_error {
            let mut fallback = Self::rule_based_result(
                &input,
                ArtifactGenerator::RuleBased,
                MeetingIntelligenceStatus::Degraded,
                Some(error.reason_code()),
                vec![format!(
                    "Local model generation failed ({:?}); rule-based fallback was used",
                    error.kind
                )],
            );
            fallback.diagnostics.fallback_used = true;
            fallback.diagnostics.degraded_reason = Some(error.message);
            fallback.diagnostics.model_provider = Some(error.provider);
            fallback.diagnostics.model_name = error.model;
            fallback.diagnostics.llm_endpoint = error.endpoint;
            fallback.diagnostics.llm_generation_duration_ms = error.llm_generation_duration_ms;
            apply_prompt_stats(&mut fallback.diagnostics, &error.stats);
            refresh_output_language_diagnostics(&mut fallback);
            return Ok(fallback);
        }

        let Some(output) = llm_output else {
            let mut warnings = Vec::new();
            let unavailable_reason = if input.generation_options.use_local_llm {
                warnings.push(
                        "Local LLM generation was requested but no governed meeting LLM adapter is connected; rule-based fallback was used".to_string(),
                    );
                Some("meeting_local_llm_adapter_not_connected".to_string())
            } else {
                None
            };
            return Ok(Self::rule_based_result(
                &input,
                ArtifactGenerator::RuleBased,
                if warnings.is_empty() {
                    MeetingIntelligenceStatus::Generated
                } else {
                    MeetingIntelligenceStatus::Degraded
                },
                unavailable_reason,
                warnings,
            ));
        };

        let validation_input = input_for_prompt_stats(&input, &output.stats);
        match Self::validate_llm_json(&output.raw_json, &validation_input, &output.model) {
            Ok(mut result) => {
                result.diagnostics.generator = ArtifactGenerator::LocalLlm {
                    provider: output.provider.clone(),
                    model: output.model.clone(),
                };
                result.diagnostics.model_provider = Some(output.provider);
                result.diagnostics.model_name = Some(output.model);
                result.diagnostics.llm_endpoint = output.endpoint;
                result.diagnostics.llm_generation_duration_ms = output.llm_generation_duration_ms;
                result.diagnostics.llm_used = true;
                result.diagnostics.fallback_used = false;
                apply_prompt_stats(&mut result.diagnostics, &output.stats);
                refresh_output_language_diagnostics(&mut result);
                result.source_transcript_segment_count = output.stats.input_segment_count;
                Ok(result)
            }
            Err(mut diagnostics) => {
                diagnostics.warnings.push(
                    "Malformed local model output was not stored; rule-based fallback was used"
                        .to_string(),
                );
                let mut fallback = Self::rule_based_result(
                    &input,
                    ArtifactGenerator::RuleBased,
                    MeetingIntelligenceStatus::Degraded,
                    Some("llm_output_rejected".to_string()),
                    diagnostics.warnings,
                );
                fallback.diagnostics.json_parse_failed = diagnostics.json_parse_failed;
                fallback.diagnostics.invalid_evidence_ids = diagnostics.invalid_evidence_ids;
                fallback.diagnostics.rejected_artifact_count = diagnostics.rejected_artifact_count;
                fallback.diagnostics.fallback_used = true;
                fallback.diagnostics.degraded_reason =
                    Some("local model output failed schema or evidence validation".to_string());
                fallback.diagnostics.model_provider = Some(output.provider);
                fallback.diagnostics.model_name = Some(output.model);
                fallback.diagnostics.llm_endpoint = output.endpoint;
                fallback.diagnostics.llm_generation_duration_ms = output.llm_generation_duration_ms;
                apply_prompt_stats(&mut fallback.diagnostics, &output.stats);
                refresh_output_language_diagnostics(&mut fallback);
                Ok(fallback)
            }
        }
    }

    pub fn validate_llm_json(
        raw_json: &str,
        input: &MeetingIntelligenceInput,
        model_name: &str,
    ) -> Result<MeetingIntelligenceResult, MeetingIntelligenceDiagnostics> {
        let draft = match serde_json::from_str::<LlmMeetingIntelligenceDraft>(raw_json) {
            Ok(value) => value,
            Err(_) => {
                let mut diagnostics = base_diagnostics(
                    MeetingIntelligenceStatus::Failed,
                    ArtifactGenerator::LocalLlm {
                        provider: "ollama".to_string(),
                        model: model_name.to_string(),
                    },
                    Some("json_parse_failed".to_string()),
                    true,
                    0,
                    0,
                    false,
                    Vec::new(),
                );
                let language_detection = detect_meeting_language(&input.transcript_entries);
                apply_language_detection(&mut diagnostics, &language_detection);
                let session_type_detection = detect_meeting_session_type(&input.transcript_entries);
                apply_session_type_detection(&mut diagnostics, &session_type_detection);
                return Err(diagnostics);
            }
        };

        let mut stats = ValidationStats::default();
        let evidence_ids = evidence_id_set(&input.transcript_entries);
        let generator = ArtifactGenerator::LocalLlm {
            provider: "ollama".to_string(),
            model: model_name.to_string(),
        };
        let now = Utc::now();

        let summary = draft.summary.and_then(|summary| {
            let evidence =
                validated_evidence(&summary.evidence_segment_ids, &evidence_ids, &mut stats);
            if evidence.is_empty() || summary.text.trim().is_empty() {
                stats.rejected_artifact_count += 1;
                return None;
            }
            Some(MeetingSummary {
                id: new_meeting_artifact_id(),
                session_id: input.session_id.clone(),
                text: bounded_text(&summary.text, 2_000),
                bullets: summary
                    .bullets
                    .into_iter()
                    .map(|bullet| bounded_text(&bullet, 280))
                    .filter(|bullet| !bullet.trim().is_empty())
                    .take(MAX_SUMMARY_BULLETS)
                    .collect(),
                evidence_segment_ids: evidence,
                generated_at: now,
                generator: generator.clone(),
                confidence: clamp_confidence(summary.confidence),
            })
        });

        let decisions = draft
            .decisions
            .into_iter()
            .filter_map(|decision| {
                let evidence =
                    validated_evidence(&decision.evidence_segment_ids, &evidence_ids, &mut stats);
                if evidence.is_empty() || decision.decision.trim().is_empty() {
                    stats.rejected_artifact_count += 1;
                    return None;
                }
                Some(MeetingDecision {
                    id: new_meeting_artifact_id(),
                    session_id: input.session_id.clone(),
                    decision: bounded_text(&decision.decision, 800),
                    rationale: non_empty_bounded(decision.rationale.as_deref(), 800),
                    made_by_speaker_id: valid_speaker_id(
                        decision.made_by_speaker_id.as_deref(),
                        input,
                    ),
                    made_by_display_name: non_empty_bounded(
                        decision.made_by_display_name.as_deref(),
                        80,
                    ),
                    evidence_segment_ids: evidence,
                    confidence: clamp_confidence(decision.confidence),
                    generated_at: now,
                    generator: generator.clone(),
                })
            })
            .take(MAX_DECISIONS)
            .collect::<Vec<_>>();

        let action_items = draft
            .action_items
            .into_iter()
            .filter_map(|item| {
                let evidence =
                    validated_evidence(&item.evidence_segment_ids, &evidence_ids, &mut stats);
                if evidence.is_empty() || item.task.trim().is_empty() {
                    stats.rejected_artifact_count += 1;
                    return None;
                }
                Some(MeetingActionItem {
                    id: new_meeting_artifact_id(),
                    session_id: input.session_id.clone(),
                    task: bounded_text(&item.task, 800),
                    assignee_speaker_id: valid_speaker_id(
                        item.assignee_speaker_id.as_deref(),
                        input,
                    ),
                    assignee_display_name: non_empty_bounded(
                        item.assignee_display_name.as_deref(),
                        80,
                    ),
                    due_date: non_empty_bounded(item.due_date.as_deref(), 80),
                    evidence_segment_ids: evidence,
                    confidence: clamp_confidence(item.confidence),
                    status: ActionItemStatus::Open,
                    generated_at: now,
                    generator: generator.clone(),
                })
            })
            .take(MAX_ACTION_ITEMS)
            .collect::<Vec<_>>();

        let open_questions = draft
            .open_questions
            .into_iter()
            .filter_map(|question| {
                let evidence =
                    validated_evidence(&question.evidence_segment_ids, &evidence_ids, &mut stats);
                if evidence.is_empty() || question.question.trim().is_empty() {
                    stats.rejected_artifact_count += 1;
                    return None;
                }
                Some(MeetingOpenQuestion {
                    id: new_meeting_artifact_id(),
                    session_id: input.session_id.clone(),
                    question: bounded_text(&question.question, 800),
                    asked_by_speaker_id: valid_speaker_id(
                        question.asked_by_speaker_id.as_deref(),
                        input,
                    ),
                    asked_by_display_name: non_empty_bounded(
                        question.asked_by_display_name.as_deref(),
                        80,
                    ),
                    evidence_segment_ids: evidence,
                    confidence: clamp_confidence(question.confidence),
                    generated_at: now,
                    generator: generator.clone(),
                })
            })
            .take(MAX_OPEN_QUESTIONS)
            .collect::<Vec<_>>();

        let risks = draft
            .risks
            .into_iter()
            .filter_map(|risk| {
                let evidence =
                    validated_evidence(&risk.evidence_segment_ids, &evidence_ids, &mut stats);
                if evidence.is_empty() || risk.risk.trim().is_empty() {
                    stats.rejected_artifact_count += 1;
                    return None;
                }
                Some(MeetingRisk {
                    id: new_meeting_artifact_id(),
                    session_id: input.session_id.clone(),
                    risk: bounded_text(&risk.risk, 800),
                    severity: risk.severity.unwrap_or_default(),
                    evidence_segment_ids: evidence,
                    confidence: clamp_confidence(risk.confidence),
                    generated_at: now,
                    generator: generator.clone(),
                })
            })
            .take(MAX_RISKS)
            .collect::<Vec<_>>();

        let technical_recap = draft.technical_recap.and_then(|recap| {
            let evidence =
                validated_evidence(&recap.evidence_segment_ids, &evidence_ids, &mut stats);
            if evidence.is_empty() {
                stats.rejected_artifact_count += 1;
                return None;
            }
            Some(MeetingTechnicalRecap {
                id: new_meeting_artifact_id(),
                session_id: input.session_id.clone(),
                bullets: recap
                    .bullets
                    .into_iter()
                    .map(|bullet| bounded_text(&bullet, 280))
                    .filter(|bullet| !bullet.trim().is_empty())
                    .take(MAX_TECHNICAL_ITEMS)
                    .collect(),
                mentioned_files: bounded_string_list(recap.mentioned_files, 20, 120),
                mentioned_commands: bounded_string_list(recap.mentioned_commands, 20, 160),
                mentioned_errors: bounded_string_list(recap.mentioned_errors, 20, 160),
                evidence_segment_ids: evidence,
                confidence: clamp_confidence(recap.confidence),
                generated_at: now,
                generator: generator.clone(),
            })
        });

        let follow_up_draft = draft.follow_up_draft.and_then(|draft| {
            let evidence =
                validated_evidence(&draft.evidence_segment_ids, &evidence_ids, &mut stats);
            if evidence.is_empty()
                || draft.subject.trim().is_empty()
                || draft.body.trim().is_empty()
            {
                stats.rejected_artifact_count += 1;
                return None;
            }
            Some(MeetingFollowUpDraft {
                id: new_meeting_artifact_id(),
                session_id: input.session_id.clone(),
                subject: bounded_text(&draft.subject, 160),
                body: bounded_text(&draft.body, 4_000),
                tone: FollowUpTone::Professional,
                evidence_segment_ids: evidence,
                confidence: clamp_confidence(draft.confidence),
                generated_at: now,
                generator: generator.clone(),
            })
        });

        let timeline = if draft.timeline.is_empty() {
            rule_based_timeline(input)
        } else {
            draft
                .timeline
                .into_iter()
                .filter_map(|item| {
                    let evidence =
                        validated_evidence(&item.evidence_segment_ids, &evidence_ids, &mut stats);
                    if evidence.is_empty() || item.title.trim().is_empty() {
                        stats.rejected_artifact_count += 1;
                        return None;
                    }
                    Some(MeetingTimelineItem {
                        id: new_meeting_artifact_id(),
                        timestamp_ms: item.timestamp_ms,
                        speaker_id: item.speaker_id,
                        speaker_display_name: non_empty_bounded(
                            item.speaker_display_name.as_deref(),
                            80,
                        ),
                        title: bounded_text(&item.title, 160),
                        detail: bounded_text(&item.detail, 400),
                        evidence_segment_ids: evidence,
                    })
                })
                .take(MAX_TIMELINE_ITEMS)
                .collect()
        };

        let mut diagnostics = base_diagnostics(
            MeetingIntelligenceStatus::Generated,
            generator.clone(),
            None,
            false,
            stats.invalid_evidence_ids,
            stats.rejected_artifact_count,
            false,
            stats.warnings,
        );
        diagnostics.llm_used = true;
        let language_detection = detect_meeting_language(&input.transcript_entries);
        apply_language_detection(&mut diagnostics, &language_detection);
        let session_type_detection = detect_meeting_session_type(&input.transcript_entries);
        apply_session_type_detection(&mut diagnostics, &session_type_detection);

        let mut result = MeetingIntelligenceResult {
            session_id: input.session_id.clone(),
            status: MeetingIntelligenceStatus::Generated,
            summary,
            decisions,
            action_items,
            open_questions,
            risks,
            technical_recap,
            follow_up_draft,
            timeline,
            diagnostics,
            source_transcript_segment_count: input.transcript_entries.len(),
            generated_at: now,
        };
        refresh_output_language_diagnostics(&mut result);
        Ok(result)
    }

    fn rule_based_result(
        input: &MeetingIntelligenceInput,
        generator: ArtifactGenerator,
        status: MeetingIntelligenceStatus,
        unavailable_reason: Option<String>,
        warnings: Vec<String>,
    ) -> MeetingIntelligenceResult {
        let now = Utc::now();
        let summary = Some(rule_based_summary(input, generator.clone(), now));
        let decisions = rule_based_decisions(input, generator.clone(), now);
        let action_items = rule_based_action_items(input, generator.clone(), now);
        let open_questions = rule_based_open_questions(input, generator.clone(), now);
        let risks = rule_based_risks(input, generator.clone(), now);
        let technical_recap = Some(rule_based_technical_recap(input, generator.clone(), now));
        let follow_up_draft = Some(rule_based_follow_up_draft(
            input,
            generator.clone(),
            now,
            summary.as_ref(),
            &decisions,
            &action_items,
            &open_questions,
            &risks,
        ));
        let timeline = rule_based_timeline(input);

        let mut diagnostics = base_diagnostics(
            status.clone(),
            generator,
            unavailable_reason,
            false,
            0,
            0,
            status == MeetingIntelligenceStatus::Degraded,
            warnings,
        );
        let language_detection = detect_meeting_language(&input.transcript_entries);
        apply_language_detection(&mut diagnostics, &language_detection);
        let session_type_detection = detect_meeting_session_type(&input.transcript_entries);
        apply_session_type_detection(&mut diagnostics, &session_type_detection);

        let mut result = MeetingIntelligenceResult {
            session_id: input.session_id.clone(),
            status,
            summary,
            decisions,
            action_items,
            open_questions,
            risks,
            technical_recap,
            follow_up_draft,
            timeline,
            diagnostics,
            source_transcript_segment_count: input.transcript_entries.len(),
            generated_at: now,
        };
        refresh_output_language_diagnostics(&mut result);
        result
    }
}

fn bounded_input(mut input: MeetingIntelligenceInput) -> MeetingIntelligenceInput {
    let max_segments = input
        .generation_options
        .max_transcript_segments
        .clamp(1, 500);
    if input.transcript_entries.len() > max_segments {
        let start = input.transcript_entries.len() - max_segments;
        input.transcript_entries = input.transcript_entries[start..].to_vec();
    }
    input
}

pub fn build_meeting_llm_prompt_input(input: &MeetingIntelligenceInput) -> MeetingLlmPromptInput {
    let max_segments = input
        .generation_options
        .max_transcript_segments
        .clamp(1, 500);
    let max_chars_total = configured_usize(
        "ASTRA_MEETING_INTELLIGENCE_MAX_PROMPT_CHARS",
        DEFAULT_PROMPT_MAX_CHARS_TOTAL,
        2_000,
        120_000,
    );
    let max_chars_per_segment = configured_usize(
        "ASTRA_MEETING_INTELLIGENCE_MAX_CHARS_PER_SEGMENT",
        DEFAULT_PROMPT_MAX_CHARS_PER_SEGMENT,
        120,
        8_000,
    );

    let mut selected_rev = Vec::new();
    let mut input_char_count = 0usize;
    let mut input_truncated = input.transcript_entries.len() > max_segments;

    for entry in input.transcript_entries.iter().rev().take(max_segments) {
        let original_len = entry.text.trim().chars().count();
        if original_len == 0 {
            continue;
        }
        let remaining = max_chars_total.saturating_sub(input_char_count);
        if remaining == 0 {
            input_truncated = true;
            break;
        }
        let text_limit = remaining.min(max_chars_per_segment);
        let text = bounded_text(&entry.text, text_limit);
        let text_len = text.chars().count();
        if text_len < original_len {
            input_truncated = true;
        }
        input_char_count = input_char_count.saturating_add(text_len);
        selected_rev.push(MeetingLlmPromptSegment {
            segment_id: entry.segment_id.clone(),
            speaker_id: entry.speaker_id.clone(),
            speaker_label: entry.speaker_display_name().to_string(),
            source: entry.source,
            start_ms: entry.start_ms,
            end_ms: entry.end_ms,
            text,
        });
    }

    selected_rev.reverse();
    let language_detection = detect_prompt_language(&selected_rev);
    let session_type_detection = detect_prompt_session_type(&selected_rev);
    let included_segment_ids = selected_rev
        .iter()
        .map(|segment| segment.segment_id.clone())
        .collect::<Vec<_>>();
    let stats = MeetingLlmPromptStats {
        input_segment_count: selected_rev.len(),
        input_truncated,
        input_char_count,
        max_segments,
        max_chars_total,
        max_chars_per_segment,
        included_segment_ids,
        detected_language: language_detection.language,
        language_confidence: language_detection.confidence,
        language_source: language_detection.source,
        session_type: session_type_detection.session_type,
        session_type_confidence: session_type_detection.confidence,
        session_type_source: session_type_detection.source,
    };
    let prompt = meeting_llm_user_prompt(input, &selected_rev, &stats);
    MeetingLlmPromptInput {
        session_id: input.session_id.clone(),
        prompt,
        segments: selected_rev,
        stats,
    }
}

pub fn build_meeting_llm_language_retry_prompt_input(
    previous: &MeetingLlmPromptInput,
) -> MeetingLlmPromptInput {
    let mut retry = previous.clone();
    let language = meeting_language_label(previous.stats.detected_language);
    retry.prompt.push_str(&format!(
        r#"

LANGUAGE CORRECTION RETRY:
- The previous model output did not match the detected transcript language.
- Return the same strict JSON schema and the same evidence_segment_ids rules.
- Keep every JSON key exactly as specified in English.
- Rewrite every user-facing string value in {language}.
- For Italian, use natural professional Italian. Do not use English greetings such as "Hi" or sign-offs such as "Best".
- Do not add unsupported facts, names, due dates, commitments, or recipients.
- Return JSON only.
"#
    ));
    retry
}

fn prompt_stats_for_input(input: &MeetingIntelligenceInput) -> MeetingLlmPromptStats {
    let bounded = bounded_input(input.clone());
    let language_detection = detect_meeting_language(&bounded.transcript_entries);
    let session_type_detection = detect_meeting_session_type(&bounded.transcript_entries);
    MeetingLlmPromptStats {
        input_segment_count: bounded.transcript_entries.len(),
        input_truncated: input.transcript_entries.len() != bounded.transcript_entries.len(),
        input_char_count: bounded
            .transcript_entries
            .iter()
            .map(|entry| entry.text.chars().count())
            .sum(),
        max_segments: input
            .generation_options
            .max_transcript_segments
            .clamp(1, 500),
        max_chars_total: 0,
        max_chars_per_segment: 0,
        included_segment_ids: bounded
            .transcript_entries
            .iter()
            .map(|entry| entry.segment_id.clone())
            .collect(),
        detected_language: language_detection.language,
        language_confidence: language_detection.confidence,
        language_source: language_detection.source,
        session_type: session_type_detection.session_type,
        session_type_confidence: session_type_detection.confidence,
        session_type_source: session_type_detection.source,
    }
}

fn input_for_prompt_stats(
    input: &MeetingIntelligenceInput,
    stats: &MeetingLlmPromptStats,
) -> MeetingIntelligenceInput {
    if stats.included_segment_ids.is_empty() {
        return bounded_input(input.clone());
    }
    let included = stats
        .included_segment_ids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let mut filtered = input.clone();
    filtered.transcript_entries = input
        .transcript_entries
        .iter()
        .filter(|entry| included.contains(&entry.segment_id))
        .cloned()
        .collect();
    filtered
}

fn apply_prompt_stats(
    diagnostics: &mut MeetingIntelligenceDiagnostics,
    stats: &MeetingLlmPromptStats,
) {
    diagnostics.input_segment_count = stats.input_segment_count;
    diagnostics.input_truncated = stats.input_truncated;
    diagnostics.input_char_count = stats.input_char_count;
    diagnostics.max_segments = stats.max_segments;
    diagnostics.max_chars_total = stats.max_chars_total;
    diagnostics.max_chars_per_segment = stats.max_chars_per_segment;
    diagnostics.detected_language = stats.detected_language;
    diagnostics.language_confidence = stats.language_confidence;
    diagnostics.language_source = stats.language_source;
    diagnostics.session_type = stats.session_type;
    diagnostics.session_type_confidence = stats.session_type_confidence;
    diagnostics.session_type_source = stats.session_type_source;
    if stats.input_truncated
        && !diagnostics.warnings.iter().any(|warning| {
            warning == "Local model input was truncated to bounded transcript context"
        })
    {
        diagnostics
            .warnings
            .push("Local model input was truncated to bounded transcript context".to_string());
    }
}

fn base_diagnostics(
    status: MeetingIntelligenceStatus,
    generator: ArtifactGenerator,
    unavailable_reason: Option<String>,
    json_parse_failed: bool,
    invalid_evidence_ids: usize,
    rejected_artifact_count: usize,
    fallback_used: bool,
    warnings: Vec<String>,
) -> MeetingIntelligenceDiagnostics {
    let (provider, model) = match &generator {
        ArtifactGenerator::LocalLlm { provider, model } => {
            (Some(provider.clone()), Some(model.clone()))
        }
        _ => (None, None),
    };
    MeetingIntelligenceDiagnostics {
        status,
        generator,
        model_provider: provider,
        model_name: model,
        llm_endpoint: None,
        degraded_reason: unavailable_reason.clone(),
        model_unavailable_reason: unavailable_reason,
        llm_used: false,
        json_parse_failed,
        invalid_evidence_ids,
        rejected_artifact_count,
        fallback_used,
        input_segment_count: 0,
        input_truncated: false,
        input_char_count: 0,
        max_segments: 0,
        max_chars_total: 0,
        max_chars_per_segment: 0,
        detected_language: MeetingLanguage::Unknown,
        language_confidence: 0.0,
        language_source: MeetingLanguageSource::Unknown,
        output_language: MeetingLanguage::Unknown,
        output_language_mismatch: false,
        language_retry_attempted: false,
        language_retry_succeeded: false,
        session_type: MeetingSessionType::General,
        session_type_confidence: 0.0,
        session_type_source: MeetingSessionTypeSource::Unknown,
        llm_generation_duration_ms: None,
        total_generation_duration_ms: None,
        transcript_changed_during_generation: false,
        snapshot_transcript_segment_count: 0,
        transcript_text_logged: false,
        audit_redacted: true,
        warnings,
        generated_at: Utc::now(),
    }
}

fn meeting_llm_system_prompt() -> &'static str {
    "You are Astra Meeting Intelligence, a local-only model behind a Rust-governed desktop assistant. Return strict JSON only. Do not reveal chain-of-thought. Use only the provided transcript evidence. Do not fabricate speaker identities, people, commitments, due dates, files, commands, errors, or facts. If evidence is missing, omit the artifact."
}

fn meeting_llm_user_prompt(
    input: &MeetingIntelligenceInput,
    segments: &[MeetingLlmPromptSegment],
    stats: &MeetingLlmPromptStats,
) -> String {
    let speaker_registry = input
        .speakers
        .iter()
        .map(|speaker| {
            json!({
                "speaker_id": &speaker.speaker_id,
                "display_name": &speaker.display_name,
                "source": speaker.source,
                "attribution_method": speaker.attribution_method,
                "confidence": speaker.confidence,
            })
        })
        .collect::<Vec<_>>();
    let transcript = segments
        .iter()
        .map(|segment| {
            json!({
                "segment_id": &segment.segment_id,
                "speaker_id": &segment.speaker_id,
                "speaker_label": &segment.speaker_label,
                "source": segment.source,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "text": &segment.text,
            })
        })
        .collect::<Vec<_>>();
    let speaker_json =
        serde_json::to_string_pretty(&speaker_registry).unwrap_or_else(|_| "[]".to_string());
    let transcript_json =
        serde_json::to_string_pretty(&transcript).unwrap_or_else(|_| "[]".to_string());
    let screen_context = input
        .screen_contexts
        .iter()
        .rev()
        .take(12)
        .map(|context| {
            json!({
                "context_id": &context.context_id,
                "captured_at": context.captured_at.to_rfc3339(),
                "linked_transcript_segment_ids": &context.linked_transcript_segment_ids,
                "summary": bounded_text(&context.summary, 700),
                "screenshot_stored": context.screenshot_ref.is_some(),
                "redaction": context.redaction,
                "confidence": context.confidence,
            })
        })
        .collect::<Vec<_>>();
    let screen_context_json =
        serde_json::to_string_pretty(&screen_context).unwrap_or_else(|_| "[]".to_string());
    let language_instruction = meeting_language_instruction(stats.detected_language);
    let session_type_instruction = meeting_session_type_instruction(stats.session_type);

    format!(
        r#"Generate transcript-backed meeting intelligence for session "{session_id}".

Output language:
- Detected transcript language: {detected_language} (confidence {language_confidence}, source {language_source}).
{language_instruction}
- Keep JSON keys exactly as specified in English. Only user-facing string values should follow the output language.
- Do not switch to English unless the transcript is primarily English or a direct quote requires it.

Bounded input:
- input_segment_count: {input_segment_count}
- input_truncated: {input_truncated}
- input_char_count: {input_char_count}
- max_segments: {max_segments}
- max_chars_total: {max_chars_total}
- max_chars_per_segment: {max_chars_per_segment}

Session type:
- Detected session type: {session_type} (confidence {session_type_confidence}, source {session_type_source}).
{session_type_instruction}

Speaker registry JSON:
{speaker_json}

Transcript entries JSON:
{transcript_json}

Attached screen context JSON:
{screen_context_json}

Return JSON only, with this exact top-level shape:
{{
  "summary": {{
    "text": "string",
    "bullets": ["string"],
    "evidence_segment_ids": ["segment_id"],
    "confidence": 0.0
  }},
  "decisions": [
    {{
      "decision": "string",
      "rationale": "string or null",
      "made_by_speaker_id": "speaker_id or null",
      "made_by_display_name": "display name or null",
      "evidence_segment_ids": ["segment_id"],
      "confidence": 0.0
    }}
  ],
  "action_items": [
    {{
      "task": "string",
      "assignee_speaker_id": "speaker_id or null",
      "assignee_display_name": "display name or null",
      "due_date": "string or null",
      "evidence_segment_ids": ["segment_id"],
      "confidence": 0.0
    }}
  ],
  "open_questions": [
    {{
      "question": "string",
      "asked_by_speaker_id": "speaker_id or null",
      "asked_by_display_name": "display name or null",
      "evidence_segment_ids": ["segment_id"],
      "confidence": 0.0
    }}
  ],
  "risks": [
    {{
      "risk": "string",
      "severity": "low|medium|high",
      "evidence_segment_ids": ["segment_id"],
      "confidence": 0.0
    }}
  ],
  "technical_recap": {{
    "bullets": ["string"],
    "mentioned_files": ["string"],
    "mentioned_commands": ["string"],
    "mentioned_errors": ["string"],
    "evidence_segment_ids": ["segment_id"],
    "confidence": 0.0
  }},
  "follow_up_draft": {{
    "subject": "string",
    "body": "string",
    "evidence_segment_ids": ["segment_id"],
    "confidence": 0.0
  }},
  "timeline": [
    {{
      "timestamp_ms": 0,
      "speaker_id": "speaker_id or null",
      "speaker_display_name": "display name or null",
      "title": "string",
      "detail": "string",
      "evidence_segment_ids": ["segment_id"]
    }}
  ]
}}

Rules:
- Every meaningful artifact must include valid evidence_segment_ids from the transcript entries above.
- Attached screen context is supplemental evidence only. Use it to improve wording and technical recap when it clarifies what was visible, but do not replace transcript evidence or fabricate facts from it.
- Omit decisions/action_items/open_questions/risks that are not directly supported by evidence.
- Summary: write a natural professional recap of what happened, what was discussed, and why it matters. Do not write a mechanical segment-count summary unless forced into rule-based fallback.
- Decisions: include only real commitments, conclusions, confirmations, approvals, or rejections. Do not classify "we talked about X" as a decision.
- Action items: make tasks concrete and actionable. Do not invent assignees or dates; leave them null when absent.
- Open questions: include unresolved points only; do not list questions that the transcript later answers.
- Risks/blockers: include actual concerns grounded in transcript evidence, not generic project risks.
- Technical recap: extract files/modules, commands, errors, architecture choices, config/env vars, runtime behavior, tests, validation, and limitations when present.
- Follow-up draft: professional, concise, copy-only, no fabricated recipients, no sending implied.
- Timeline: compactly group meaningful events; do not duplicate every transcript line.
- Do not invent recipient names. The follow-up draft is draft-only and must not imply sending.
- Do not include markdown, comments, or explanatory text outside JSON.
"#,
        session_id = input.session_id,
        input_segment_count = stats.input_segment_count,
        input_truncated = stats.input_truncated,
        input_char_count = stats.input_char_count,
        max_segments = stats.max_segments,
        max_chars_total = stats.max_chars_total,
        max_chars_per_segment = stats.max_chars_per_segment,
        session_type = meeting_session_type_label(stats.session_type),
        session_type_confidence = format!("{:.2}", stats.session_type_confidence),
        session_type_source = meeting_session_type_source_label(stats.session_type_source),
        session_type_instruction = session_type_instruction,
        detected_language = meeting_language_label(stats.detected_language),
        language_confidence = format!("{:.2}", stats.language_confidence),
        language_source = meeting_language_source_label(stats.language_source),
        language_instruction = language_instruction,
        speaker_json = speaker_json,
        transcript_json = transcript_json,
        screen_context_json = screen_context_json
    )
}

fn configured_usize(key: &str, default: usize, min: usize, max: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
}

fn configured_ollama_base_url() -> String {
    let astra_url = env::var("ASTRA_OLLAMA_BASE_URL").ok();
    let ollama_host = env::var("OLLAMA_HOST").ok();
    configured_ollama_base_url_from(astra_url.as_deref(), ollama_host.as_deref())
}

fn configured_ollama_base_url_from(astra_url: Option<&str>, ollama_host: Option<&str>) -> String {
    astra_url
        .and_then(normalize_ollama_base_url)
        .or_else(|| ollama_host.and_then(normalize_ollama_base_url))
        .unwrap_or_else(|| DEFAULT_OLLAMA_BASE_URL.to_string())
}

fn normalize_ollama_base_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        Some(trimmed.to_string())
    } else {
        Some(format!("http://{trimmed}"))
    }
}

fn sanitize_ollama_endpoint(value: &str) -> String {
    let without_scheme = value
        .strip_prefix("http://")
        .or_else(|| value.strip_prefix("https://"))
        .unwrap_or(value);
    let host = without_scheme.split('/').next().unwrap_or("").trim();
    if host.is_empty() || host.contains('@') {
        return "configured endpoint".to_string();
    }
    if host.starts_with("127.0.0.1")
        || host.starts_with("localhost")
        || host.starts_with("0.0.0.0")
        || host.starts_with("[::1]")
    {
        host.to_string()
    } else {
        "configured endpoint".to_string()
    }
}

fn elapsed_ms(started_at: Instant) -> u64 {
    started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MeetingLanguageDetection {
    language: MeetingLanguage,
    confidence: f32,
    source: MeetingLanguageSource,
}

impl Default for MeetingLanguageDetection {
    fn default() -> Self {
        Self {
            language: MeetingLanguage::Unknown,
            confidence: 0.0,
            source: MeetingLanguageSource::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MeetingSessionTypeDetection {
    session_type: MeetingSessionType,
    confidence: f32,
    source: MeetingSessionTypeSource,
}

impl Default for MeetingSessionTypeDetection {
    fn default() -> Self {
        Self {
            session_type: MeetingSessionType::General,
            confidence: 0.0,
            source: MeetingSessionTypeSource::Unknown,
        }
    }
}

fn detect_meeting_language(entries: &[TranscriptEntry]) -> MeetingLanguageDetection {
    let mut total_italian = 0.0f32;
    let mut total_english = 0.0f32;
    let mut local_italian = 0.0f32;
    let mut local_english = 0.0f32;

    for entry in entries {
        let (italian, english) = score_language_markers(&entry.text);
        let is_local = entry.source == TranscriptSource::Microphone
            || entry.speaker_id.as_deref() == Some(LOCAL_USER_SPEAKER_ID);
        let weight = if is_local { 1.6 } else { 1.0 };
        total_italian += italian * weight;
        total_english += english * weight;
        if is_local {
            local_italian += italian * weight;
            local_english += english * weight;
        }
    }

    choose_language_detection(total_italian, total_english, local_italian, local_english)
}

fn detect_prompt_language(segments: &[MeetingLlmPromptSegment]) -> MeetingLanguageDetection {
    let mut total_italian = 0.0f32;
    let mut total_english = 0.0f32;
    let mut local_italian = 0.0f32;
    let mut local_english = 0.0f32;

    for segment in segments {
        let (italian, english) = score_language_markers(&segment.text);
        let is_local = segment.source == TranscriptSource::Microphone
            || segment.speaker_id.as_deref() == Some(LOCAL_USER_SPEAKER_ID);
        let weight = if is_local { 1.6 } else { 1.0 };
        total_italian += italian * weight;
        total_english += english * weight;
        if is_local {
            local_italian += italian * weight;
            local_english += english * weight;
        }
    }

    choose_language_detection(total_italian, total_english, local_italian, local_english)
}

fn detect_meeting_session_type(entries: &[TranscriptEntry]) -> MeetingSessionTypeDetection {
    let scores = entries
        .iter()
        .map(|entry| {
            let weight = if entry.source == TranscriptSource::Microphone
                || entry.speaker_id.as_deref() == Some(LOCAL_USER_SPEAKER_ID)
            {
                1.25
            } else {
                1.0
            };
            session_type_scores(&entry.text, weight)
        })
        .fold([0.0f32; 5], |mut totals, scores| {
            for (index, score) in scores.into_iter().enumerate() {
                totals[index] += score;
            }
            totals
        });
    choose_session_type(scores)
}

fn detect_prompt_session_type(segments: &[MeetingLlmPromptSegment]) -> MeetingSessionTypeDetection {
    let scores = segments
        .iter()
        .map(|segment| {
            let weight = if segment.source == TranscriptSource::Microphone
                || segment.speaker_id.as_deref() == Some(LOCAL_USER_SPEAKER_ID)
            {
                1.25
            } else {
                1.0
            };
            session_type_scores(&segment.text, weight)
        })
        .fold([0.0f32; 5], |mut totals, scores| {
            for (index, score) in scores.into_iter().enumerate() {
                totals[index] += score;
            }
            totals
        });
    choose_session_type(scores)
}

fn session_type_scores(text: &str, weight: f32) -> [f32; 5] {
    let mut scores = [0.0f32; 5];
    for token in text
        .to_lowercase()
        .split(|character: char| {
            !character.is_alphanumeric() && character != '_' && character != '-'
        })
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if TECHNICAL_DEBUGGING_MARKERS.contains(&token) {
            scores[0] += weight;
        }
        if PLANNING_MARKERS.contains(&token) {
            scores[1] += weight;
        }
        if DECISION_REVIEW_MARKERS.contains(&token) {
            scores[2] += weight;
        }
        if SUPPORT_CALL_MARKERS.contains(&token) {
            scores[3] += weight;
        }
        if WORK_MEETING_MARKERS.contains(&token) {
            scores[4] += weight;
        }
    }
    scores
}

fn choose_session_type(scores: [f32; 5]) -> MeetingSessionTypeDetection {
    let total = scores.iter().sum::<f32>();
    if total < 1.5 {
        return MeetingSessionTypeDetection::default();
    }

    let mut best_index = 0usize;
    let mut best_score = 0.0f32;
    for (index, score) in scores.iter().enumerate() {
        if *score > best_score {
            best_score = *score;
            best_index = index;
        }
    }

    if best_score < 1.5 {
        return MeetingSessionTypeDetection::default();
    }

    let session_type = match best_index {
        0 => MeetingSessionType::TechnicalDebugging,
        1 => MeetingSessionType::Planning,
        2 => MeetingSessionType::DecisionReview,
        3 => MeetingSessionType::SupportCall,
        4 => MeetingSessionType::WorkMeeting,
        _ => MeetingSessionType::General,
    };
    MeetingSessionTypeDetection {
        session_type,
        confidence: (best_score / total.max(1.0)).clamp(0.0, 1.0),
        source: MeetingSessionTypeSource::TranscriptHeuristic,
    }
}

fn choose_language_detection(
    total_italian: f32,
    total_english: f32,
    local_italian: f32,
    local_english: f32,
) -> MeetingLanguageDetection {
    let local_total = local_italian + local_english;
    if local_total >= 2.0 {
        if local_italian > 0.0 && local_italian >= local_english * 1.2 {
            return MeetingLanguageDetection {
                language: MeetingLanguage::Italian,
                confidence: language_confidence(local_italian, local_english),
                source: MeetingLanguageSource::UserSourceWeighted,
            };
        }
        if local_english > 0.0 && local_english >= local_italian * 1.2 {
            return MeetingLanguageDetection {
                language: MeetingLanguage::English,
                confidence: language_confidence(local_english, local_italian),
                source: MeetingLanguageSource::UserSourceWeighted,
            };
        }
    }

    let total = total_italian + total_english;
    if total < 2.0 {
        return MeetingLanguageDetection::default();
    }
    if total_italian > 0.0 && total_english > 0.0 {
        let larger = total_italian.max(total_english);
        let smaller = total_italian.min(total_english);
        if larger / smaller < 1.35 {
            return MeetingLanguageDetection {
                language: MeetingLanguage::Mixed,
                confidence: 1.0 - ((larger - smaller) / total),
                source: MeetingLanguageSource::TranscriptHeuristic,
            };
        }
    }

    if total_italian > total_english {
        MeetingLanguageDetection {
            language: MeetingLanguage::Italian,
            confidence: language_confidence(total_italian, total_english),
            source: MeetingLanguageSource::TranscriptHeuristic,
        }
    } else {
        MeetingLanguageDetection {
            language: MeetingLanguage::English,
            confidence: language_confidence(total_english, total_italian),
            source: MeetingLanguageSource::TranscriptHeuristic,
        }
    }
}

fn score_language_markers(text: &str) -> (f32, f32) {
    let mut italian = 0.0f32;
    let mut english = 0.0f32;
    for token in text
        .to_lowercase()
        .split(|character: char| !character.is_alphabetic() && character != '\'')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        if ITALIAN_LANGUAGE_MARKERS.contains(&token) {
            italian += 1.0;
        }
        if ENGLISH_LANGUAGE_MARKERS.contains(&token) {
            english += 1.0;
        }
    }
    (italian, english)
}

fn language_confidence(primary: f32, secondary: f32) -> f32 {
    let total = primary + secondary;
    if total <= f32::EPSILON {
        0.0
    } else {
        ((primary - secondary).abs() / total).clamp(0.0, 1.0)
    }
}

fn apply_language_detection(
    diagnostics: &mut MeetingIntelligenceDiagnostics,
    detection: &MeetingLanguageDetection,
) {
    diagnostics.detected_language = detection.language;
    diagnostics.language_confidence = detection.confidence;
    diagnostics.language_source = detection.source;
}

fn apply_session_type_detection(
    diagnostics: &mut MeetingIntelligenceDiagnostics,
    detection: &MeetingSessionTypeDetection,
) {
    diagnostics.session_type = detection.session_type;
    diagnostics.session_type_confidence = detection.confidence;
    diagnostics.session_type_source = detection.source;
}

fn refresh_output_language_diagnostics(result: &mut MeetingIntelligenceResult) {
    let detection = detect_result_output_language(result);
    result.diagnostics.output_language = detection.language;
    result.diagnostics.output_language_mismatch =
        output_language_mismatch(result.diagnostics.detected_language, detection.language);
    if result.diagnostics.output_language_mismatch
        && !result
            .diagnostics
            .warnings
            .iter()
            .any(|warning| warning == "Generated output language did not match transcript language")
    {
        result
            .diagnostics
            .warnings
            .push("Generated output language did not match transcript language".to_string());
    }
}

fn output_language_mismatch(expected: MeetingLanguage, actual: MeetingLanguage) -> bool {
    matches!(
        (expected, actual),
        (MeetingLanguage::Italian, MeetingLanguage::English)
            | (MeetingLanguage::English, MeetingLanguage::Italian)
    )
}

fn detect_result_output_language(result: &MeetingIntelligenceResult) -> MeetingLanguageDetection {
    let mut text = Vec::new();
    if let Some(summary) = &result.summary {
        text.push(summary.text.as_str());
        text.extend(summary.bullets.iter().map(String::as_str));
    }
    text.extend(
        result
            .decisions
            .iter()
            .flat_map(|decision| {
                [
                    Some(decision.decision.as_str()),
                    decision.rationale.as_deref(),
                    decision.made_by_display_name.as_deref(),
                ]
            })
            .flatten(),
    );
    text.extend(
        result
            .action_items
            .iter()
            .flat_map(|item| {
                [
                    Some(item.task.as_str()),
                    item.assignee_display_name.as_deref(),
                    item.due_date.as_deref(),
                ]
            })
            .flatten(),
    );
    text.extend(
        result
            .open_questions
            .iter()
            .flat_map(|question| {
                [
                    Some(question.question.as_str()),
                    question.asked_by_display_name.as_deref(),
                ]
            })
            .flatten(),
    );
    text.extend(result.risks.iter().map(|risk| risk.risk.as_str()));
    if let Some(recap) = &result.technical_recap {
        text.extend(recap.bullets.iter().map(String::as_str));
        text.extend(recap.mentioned_files.iter().map(String::as_str));
        text.extend(recap.mentioned_commands.iter().map(String::as_str));
        text.extend(recap.mentioned_errors.iter().map(String::as_str));
    }
    if let Some(draft) = &result.follow_up_draft {
        text.push(draft.subject.as_str());
        text.push(draft.body.as_str());
    }
    let joined = text.join(" ");
    let (italian, english) = score_language_markers(&joined);
    choose_language_detection(italian, english, 0.0, 0.0)
}

fn meeting_language_label(language: MeetingLanguage) -> &'static str {
    match language {
        MeetingLanguage::Italian => "Italian",
        MeetingLanguage::English => "English",
        MeetingLanguage::Mixed => "Mixed",
        MeetingLanguage::Unknown => "Unknown",
    }
}

fn meeting_language_source_label(source: MeetingLanguageSource) -> &'static str {
    match source {
        MeetingLanguageSource::TranscriptHeuristic => "transcript_heuristic",
        MeetingLanguageSource::UserSourceWeighted => "user_source_weighted",
        MeetingLanguageSource::Unknown => "unknown",
    }
}

fn meeting_session_type_label(session_type: MeetingSessionType) -> &'static str {
    match session_type {
        MeetingSessionType::TechnicalDebugging => "TechnicalDebugging",
        MeetingSessionType::WorkMeeting => "WorkMeeting",
        MeetingSessionType::Planning => "Planning",
        MeetingSessionType::DecisionReview => "DecisionReview",
        MeetingSessionType::SupportCall => "SupportCall",
        MeetingSessionType::General => "General",
    }
}

fn meeting_session_type_source_label(source: MeetingSessionTypeSource) -> &'static str {
    match source {
        MeetingSessionTypeSource::TranscriptHeuristic => "transcript_heuristic",
        MeetingSessionTypeSource::Unknown => "unknown",
    }
}

fn meeting_session_type_instruction(session_type: MeetingSessionType) -> &'static str {
    match session_type {
        MeetingSessionType::TechnicalDebugging => {
            "- This looks like a technical debugging session: emphasize concrete errors, files/modules, commands, configuration, runtime behavior, fixes, tests, and validation outcomes.\n- The technical_recap should be especially useful for engineering follow-up."
        }
        MeetingSessionType::Planning => {
            "- This looks like a planning session: emphasize roadmap, priorities, next steps, dependencies, owners, and sequencing.\n- Do not turn loose ideas into decisions unless the transcript confirms them."
        }
        MeetingSessionType::DecisionReview => {
            "- This looks like a decision review: emphasize real decisions, rationale, confirmations, rejected options, owners, and evidence.\n- Generic discussion is not a decision."
        }
        MeetingSessionType::SupportCall => {
            "- This looks like a support call: emphasize the reported issue, user/customer impact, reproduction details, blockers, next support steps, and validation."
        }
        MeetingSessionType::WorkMeeting => {
            "- This looks like a work meeting: emphasize agenda progress, decisions, action items, open questions, blockers, and follow-up."
        }
        MeetingSessionType::General => {
            "- Session type is general or unclear: keep artifacts concise and grounded, and omit categories without clear evidence."
        }
    }
}

fn meeting_language_instruction(language: MeetingLanguage) -> &'static str {
    match language {
        MeetingLanguage::Italian => {
            "- Generate every user-facing string value in natural professional Italian.\n- The follow-up subject and body must be Italian; do not use \"Hi\" or \"Best\"."
        }
        MeetingLanguage::English => {
            "- Generate every user-facing string value in professional English.\n- The follow-up subject and body must be English."
        }
        MeetingLanguage::Mixed => {
            "- Prefer the local user/microphone dominant language when clear.\n- If a specific item is only supported by another language in the transcript, preserve that item language instead of inventing a translation."
        }
        MeetingLanguage::Unknown => {
            "- Preserve the transcript's natural language where detectable from the provided entries.\n- If language remains unclear, use concise professional language without inventing context."
        }
    }
}

fn select_first_available(candidates: &[String], installed_models: &[String]) -> Option<String> {
    let installed_lower = installed_models
        .iter()
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();
    candidates.iter().find_map(|candidate| {
        let exact = candidate.to_ascii_lowercase();
        if installed_lower.iter().any(|installed| installed == &exact) {
            return Some(candidate.clone());
        }
        let base = exact.split(':').next().unwrap_or(&exact).to_string();
        installed_models.iter().find_map(|installed| {
            let installed_lower = installed.to_ascii_lowercase();
            (installed_lower == base || installed_lower.starts_with(&(base.clone() + ":")))
                .then(|| installed.clone())
        })
    })
}

fn extract_json_object(content: &str) -> Option<&str> {
    let trimmed = content.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Some(trimmed);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    (end > start).then_some(trimmed[start..=end].trim())
}

fn bounded_error_message(message: String) -> String {
    message.trim().chars().take(240).collect()
}

fn rule_based_summary(
    input: &MeetingIntelligenceInput,
    generator: ArtifactGenerator,
    now: chrono::DateTime<Utc>,
) -> MeetingSummary {
    let evidence = all_evidence(input);
    let speakers = speaker_names(input);
    let topics = extract_topics(&input.transcript_entries);
    let language = detect_meeting_language(&input.transcript_entries).language;
    let italian = matches!(language, MeetingLanguage::Italian);
    let session_type = detect_meeting_session_type(&input.transcript_entries).session_type;
    let mut bullets = Vec::new();
    let speaker_phrase = if speakers.is_empty() {
        if italian {
            "speaker non identificati".to_string()
        } else {
            "unidentified speakers".to_string()
        }
    } else {
        speakers.join(", ")
    };
    let topic_phrase = if topics.is_empty() {
        if italian {
            "i punti principali della sessione".to_string()
        } else {
            "the main session points".to_string()
        }
    } else {
        topics.join(", ")
    };
    bullets.push(if italian {
        format!("Durante la sessione, {speaker_phrase} hanno discusso {topic_phrase}.")
    } else {
        format!("During the session, {speaker_phrase} discussed {topic_phrase}.")
    });
    bullets.push(match (italian, session_type) {
        (true, MeetingSessionType::TechnicalDebugging) => {
            "Il recap privilegia errori, comandi, moduli, configurazione e verifiche citate nella trascrizione.".to_string()
        }
        (false, MeetingSessionType::TechnicalDebugging) => {
            "The recap emphasizes errors, commands, modules, configuration, and validation mentioned in the transcript.".to_string()
        }
        (true, MeetingSessionType::Planning) => {
            "Il recap mette in evidenza priorita, prossimi passi, dipendenze e sequenza operativa.".to_string()
        }
        (false, MeetingSessionType::Planning) => {
            "The recap emphasizes priorities, next steps, dependencies, and sequencing.".to_string()
        }
        (true, MeetingSessionType::DecisionReview) => {
            "Il recap distingue le decisioni confermate dalla semplice discussione.".to_string()
        }
        (false, MeetingSessionType::DecisionReview) => {
            "The recap separates confirmed decisions from general discussion.".to_string()
        }
        (true, MeetingSessionType::SupportCall) => {
            "Il recap mette in evidenza problema segnalato, impatto, riproduzione e prossimi passi di supporto.".to_string()
        }
        (false, MeetingSessionType::SupportCall) => {
            "The recap emphasizes reported issue, impact, reproduction, and support next steps.".to_string()
        }
        (true, _) => {
            "Gli elementi derivati sono stati sintetizzati solo quando supportati dalla trascrizione.".to_string()
        }
        (false, _) => {
            "Derived items were summarized only when backed by transcript evidence.".to_string()
        }
    });
    if !topics.is_empty() {
        bullets.push(format!(
            "{}: {}.",
            if italian {
                "Argomenti ricorrenti"
            } else {
                "Recurring topics"
            },
            topics.join(", ")
        ));
    }
    bullets.push(format!(
        "{} {} {}.",
        if italian {
            "Gli artifact derivati restano collegati a"
        } else {
            "Derived artifacts are evidence-linked to"
        },
        evidence.len(),
        if italian {
            "segmenti della trascrizione"
        } else {
            "transcript segment(s)"
        }
    ));

    MeetingSummary {
        id: new_meeting_artifact_id(),
        session_id: input.session_id.clone(),
        text: bullets
            .iter()
            .take(2)
            .cloned()
            .collect::<Vec<_>>()
            .join(" "),
        bullets,
        evidence_segment_ids: evidence,
        generated_at: now,
        generator,
        confidence: 0.72,
    }
}

fn rule_based_decisions(
    input: &MeetingIntelligenceInput,
    generator: ArtifactGenerator,
    now: chrono::DateTime<Utc>,
) -> Vec<MeetingDecision> {
    let mut tracker = DecisionLog::new();
    tracker
        .track(&input.transcript_entries)
        .into_iter()
        .filter(|decision| !decision.evidence_segment_ids.is_empty())
        .map(|decision| MeetingDecision {
            id: new_meeting_artifact_id(),
            session_id: input.session_id.clone(),
            decision: bounded_text(&decision.decision, 800),
            rationale: non_empty_bounded(Some(decision.rationale.as_str()), 800),
            made_by_speaker_id: decision
                .made_by
                .as_ref()
                .and_then(|participant| participant.speaker_id.clone()),
            made_by_display_name: decision.made_by.map(|participant| participant.name),
            evidence_segment_ids: decision.evidence_segment_ids,
            confidence: 0.7,
            generated_at: now,
            generator: generator.clone(),
        })
        .collect()
}

fn rule_based_action_items(
    input: &MeetingIntelligenceInput,
    generator: ArtifactGenerator,
    now: chrono::DateTime<Utc>,
) -> Vec<MeetingActionItem> {
    let mut tracker = ActionItemTracker::new();
    tracker
        .track(&input.transcript_entries)
        .into_iter()
        .filter(|item| !item.evidence_segment_ids.is_empty())
        .map(|item| MeetingActionItem {
            id: new_meeting_artifact_id(),
            session_id: input.session_id.clone(),
            task: bounded_text(
                if item.title.trim().is_empty() {
                    &item.description
                } else {
                    &item.title
                },
                800,
            ),
            assignee_speaker_id: item
                .assignee
                .as_ref()
                .and_then(|participant| participant.speaker_id.clone()),
            assignee_display_name: item.assignee.map(|participant| participant.name),
            due_date: item
                .deadline
                .map(|deadline| deadline.format("%Y-%m-%d").to_string()),
            evidence_segment_ids: item.evidence_segment_ids,
            confidence: 0.68,
            status: item.status,
            generated_at: now,
            generator: generator.clone(),
        })
        .collect()
}

fn rule_based_open_questions(
    input: &MeetingIntelligenceInput,
    generator: ArtifactGenerator,
    now: chrono::DateTime<Utc>,
) -> Vec<MeetingOpenQuestion> {
    input
        .transcript_entries
        .iter()
        .filter(|entry| looks_like_question(&entry.text))
        .map(|entry| MeetingOpenQuestion {
            id: new_meeting_artifact_id(),
            session_id: input.session_id.clone(),
            question: bounded_text(&entry.text, 800),
            asked_by_speaker_id: entry.speaker_id.clone(),
            asked_by_display_name: Some(entry.speaker_display_name().to_string()),
            evidence_segment_ids: vec![entry.segment_id.clone()],
            confidence: 0.72,
            generated_at: now,
            generator: generator.clone(),
        })
        .collect()
}

fn rule_based_risks(
    input: &MeetingIntelligenceInput,
    generator: ArtifactGenerator,
    now: chrono::DateTime<Utc>,
) -> Vec<MeetingRisk> {
    input
        .transcript_entries
        .iter()
        .filter(|entry| looks_like_risk(&entry.text))
        .map(|entry| MeetingRisk {
            id: new_meeting_artifact_id(),
            session_id: input.session_id.clone(),
            risk: bounded_text(&entry.text, 800),
            severity: risk_severity(&entry.text),
            evidence_segment_ids: vec![entry.segment_id.clone()],
            confidence: 0.66,
            generated_at: now,
            generator: generator.clone(),
        })
        .collect()
}

fn rule_based_technical_recap(
    input: &MeetingIntelligenceInput,
    generator: ArtifactGenerator,
    now: chrono::DateTime<Utc>,
) -> MeetingTechnicalRecap {
    let mut bullets = Vec::new();
    let mut files = Vec::new();
    let mut commands = Vec::new();
    let mut errors = Vec::new();
    let mut evidence = Vec::new();

    for entry in &input.transcript_entries {
        let technical = extract_technical_tokens(&entry.text);
        if technical.has_content() {
            push_unique(&mut evidence, entry.segment_id.clone());
            for file in technical.files {
                push_unique_limited(&mut files, file, 20);
            }
            for command in technical.commands {
                push_unique_limited(&mut commands, command, 20);
            }
            for error in technical.errors {
                push_unique_limited(&mut errors, error, 20);
            }
            push_unique_limited(
                &mut bullets,
                format!(
                    "[{}] {}",
                    entry.speaker_display_name(),
                    bounded_text(&entry.text, 220)
                ),
                MAX_TECHNICAL_ITEMS,
            );
        }
    }

    MeetingTechnicalRecap {
        id: new_meeting_artifact_id(),
        session_id: input.session_id.clone(),
        bullets,
        mentioned_files: files,
        mentioned_commands: commands,
        mentioned_errors: errors,
        evidence_segment_ids: evidence,
        confidence: 0.64,
        generated_at: now,
        generator,
    }
}

fn rule_based_follow_up_draft(
    input: &MeetingIntelligenceInput,
    generator: ArtifactGenerator,
    now: chrono::DateTime<Utc>,
    summary: Option<&MeetingSummary>,
    decisions: &[MeetingDecision],
    action_items: &[MeetingActionItem],
    questions: &[MeetingOpenQuestion],
    risks: &[MeetingRisk],
) -> MeetingFollowUpDraft {
    let language = detect_meeting_language(&input.transcript_entries).language;
    let italian = matches!(language, MeetingLanguage::Italian);
    let mut body = if italian {
        String::from(
            "Buongiorno,\n\n\
             di seguito il riepilogo dei punti principali emersi dalla trascrizione della sessione.\n\n",
        )
    } else {
        String::from("Hi,\n\nHere is the transcript-backed recap from the work session.\n\n")
    };
    if let Some(summary) = summary {
        body.push_str(if italian {
            "Riepilogo:\n"
        } else {
            "Summary:\n"
        });
        for bullet in &summary.bullets {
            body.push_str("- ");
            body.push_str(bullet);
            body.push('\n');
        }
        body.push('\n');
    }
    if !decisions.is_empty() {
        body.push_str(if italian {
            "Decisioni:\n"
        } else {
            "Decisions:\n"
        });
        for decision in decisions.iter().take(6) {
            body.push_str("- ");
            body.push_str(&decision.decision);
            body.push('\n');
        }
        body.push('\n');
    }
    if !action_items.is_empty() {
        body.push_str(if italian {
            "Azioni da completare:\n"
        } else {
            "Action items:\n"
        });
        for item in action_items.iter().take(8) {
            body.push_str("- ");
            body.push_str(&item.task);
            if let Some(assignee) = &item.assignee_display_name {
                body.push_str(" (");
                body.push_str(assignee);
                body.push(')');
            }
            body.push('\n');
        }
        body.push('\n');
    }
    if !questions.is_empty() {
        body.push_str(if italian {
            "Domande aperte:\n"
        } else {
            "Open questions:\n"
        });
        for question in questions.iter().take(6) {
            body.push_str("- ");
            body.push_str(&question.question);
            body.push('\n');
        }
        body.push('\n');
    }
    if !risks.is_empty() {
        body.push_str(if italian {
            "Rischi / blocchi:\n"
        } else {
            "Risks / blockers:\n"
        });
        for risk in risks.iter().take(6) {
            body.push_str("- ");
            body.push_str(&risk.risk);
            body.push('\n');
        }
        body.push('\n');
    }
    body.push_str(if italian { "Saluti\n" } else { "Best,\n" });

    MeetingFollowUpDraft {
        id: new_meeting_artifact_id(),
        session_id: input.session_id.clone(),
        subject: if italian {
            "Riepilogo sessione".to_string()
        } else {
            "Follow-up: meeting recap".to_string()
        },
        body,
        tone: FollowUpTone::Professional,
        evidence_segment_ids: all_evidence(input),
        confidence: 0.66,
        generated_at: now,
        generator,
    }
}

fn rule_based_timeline(input: &MeetingIntelligenceInput) -> Vec<MeetingTimelineItem> {
    input
        .transcript_entries
        .iter()
        .filter(|entry| entry.text.trim().chars().count() >= 18)
        .take(12.min(MAX_TIMELINE_ITEMS))
        .map(|entry| MeetingTimelineItem {
            id: new_meeting_artifact_id(),
            timestamp_ms: entry.start_ms,
            speaker_id: entry.speaker_id.clone(),
            speaker_display_name: Some(entry.speaker_display_name().to_string()),
            title: format!(
                "{} - {}",
                entry.speaker_display_name(),
                bounded_text(&entry.text, 72)
            ),
            detail: bounded_text(&entry.text, 260),
            evidence_segment_ids: vec![entry.segment_id.clone()],
        })
        .collect()
}

fn evidence_id_set(entries: &[TranscriptEntry]) -> HashSet<String> {
    entries
        .iter()
        .map(|entry| entry.segment_id.clone())
        .collect::<HashSet<_>>()
}

fn validated_evidence(
    requested: &[String],
    valid_ids: &HashSet<String>,
    stats: &mut ValidationStats,
) -> Vec<String> {
    let mut evidence = Vec::new();
    for id in requested {
        if valid_ids.contains(id) {
            push_unique(&mut evidence, id.clone());
        } else {
            stats.invalid_evidence_ids += 1;
        }
    }
    evidence
}

fn all_evidence(input: &MeetingIntelligenceInput) -> Vec<String> {
    input
        .transcript_entries
        .iter()
        .map(|entry| entry.segment_id.clone())
        .collect()
}

fn speaker_names(input: &MeetingIntelligenceInput) -> Vec<String> {
    let mut names = Vec::new();
    for entry in &input.transcript_entries {
        push_unique(&mut names, entry.speaker_display_name().to_string());
    }
    names
}

fn valid_speaker_id(id: Option<&str>, input: &MeetingIntelligenceInput) -> Option<String> {
    let id = id?.trim();
    if id.is_empty() {
        return None;
    }
    input
        .speakers
        .iter()
        .any(|speaker| speaker.speaker_id == id)
        .then(|| id.to_string())
}

fn extract_topics(entries: &[TranscriptEntry]) -> Vec<String> {
    let stop_words = [
        "the", "and", "for", "with", "that", "this", "are", "you", "from", "have", "will", "per",
        "che", "con", "una", "del", "della", "sono", "come", "abbiamo", "meeting",
    ];
    let mut counts = HashMap::<String, usize>::new();
    for entry in entries {
        for word in entry
            .text
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .map(str::trim)
            .filter(|word| word.len() >= 4)
        {
            let normalized = word.to_ascii_lowercase();
            if !stop_words.contains(&normalized.as_str()) {
                *counts.entry(normalized).or_insert(0) += 1;
            }
        }
    }
    let mut topics = counts.into_iter().collect::<Vec<_>>();
    topics.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    topics.into_iter().take(6).map(|(word, _)| word).collect()
}

fn looks_like_question(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    lower.contains('?')
        || [
            "what ", "why ", "how ", "when ", "who ", "can ", "should ", "could ", "cosa ",
            "come ", "quando ", "perche ",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
}

fn looks_like_risk(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    [
        "risk",
        "blocker",
        "blocked",
        "issue",
        "problem",
        "concern",
        "failure",
        "failed",
        "error",
        "bug",
        "regression",
        "unsafe",
        "privacy",
        "rischio",
        "blocco",
        "errore",
        "problema",
    ]
    .iter()
    .any(|keyword| lower.contains(keyword))
}

fn risk_severity(text: &str) -> RiskSeverity {
    let lower = text.to_ascii_lowercase();
    if [
        "critical", "blocker", "blocked", "security", "privacy", "unsafe",
    ]
    .iter()
    .any(|keyword| lower.contains(keyword))
    {
        RiskSeverity::High
    } else if ["minor", "low"]
        .iter()
        .any(|keyword| lower.contains(keyword))
    {
        RiskSeverity::Low
    } else {
        RiskSeverity::Medium
    }
}

#[derive(Default)]
struct TechnicalTokens {
    files: Vec<String>,
    commands: Vec<String>,
    errors: Vec<String>,
}

impl TechnicalTokens {
    fn has_content(&self) -> bool {
        !self.files.is_empty() || !self.commands.is_empty() || !self.errors.is_empty()
    }
}

fn extract_technical_tokens(text: &str) -> TechnicalTokens {
    let mut tokens = TechnicalTokens::default();
    let lower = text.to_ascii_lowercase();
    for raw in text.split_whitespace() {
        let token = raw.trim_matches(|character: char| {
            matches!(
                character,
                ',' | ';' | ':' | '"' | '\'' | ')' | '(' | '[' | ']'
            )
        });
        if looks_like_file_reference(token) {
            push_unique_limited(&mut tokens.files, token.to_string(), 20);
        }
        if looks_like_command(token) {
            push_unique_limited(&mut tokens.commands, token.to_string(), 20);
        }
    }

    for keyword in [
        "error",
        "failed",
        "panic",
        "exception",
        "bug",
        "regression",
        "timeout",
    ] {
        if lower.contains(keyword) {
            push_unique_limited(&mut tokens.errors, keyword.to_string(), 20);
        }
    }
    tokens
}

fn looks_like_file_reference(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    [
        ".rs", ".tsx", ".ts", ".py", ".json", ".md", ".toml", ".yaml", ".yml", ".css", ".html",
    ]
    .iter()
    .any(|extension| lower.ends_with(extension))
        || lower.contains("src/")
        || lower.contains("src\\")
}

fn looks_like_command(token: &str) -> bool {
    matches!(
        token,
        "cargo" | "npm" | "pnpm" | "python" | "pytest" | "git" | "rustc" | "node" | "tauri"
    )
}

fn clamp_confidence(confidence: Option<f32>) -> f32 {
    confidence.unwrap_or(0.65).clamp(0.0, 1.0)
}

fn bounded_text(text: &str, max_chars: usize) -> String {
    text.trim().chars().take(max_chars).collect::<String>()
}

fn non_empty_bounded(value: Option<&str>, max_chars: usize) -> Option<String> {
    value
        .map(|text| bounded_text(text, max_chars))
        .filter(|text| !text.trim().is_empty())
}

fn bounded_string_list(values: Vec<String>, max_items: usize, max_chars: usize) -> Vec<String> {
    let mut result = Vec::new();
    for value in values {
        push_unique_limited(&mut result, bounded_text(&value, max_chars), max_items);
    }
    result
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !value.trim().is_empty() && !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn push_unique_limited(values: &mut Vec<String>, value: String, limit: usize) {
    if values.len() < limit {
        push_unique(values, value);
    }
}

#[derive(Debug, Deserialize)]
struct LlmMeetingIntelligenceDraft {
    #[serde(default)]
    summary: Option<LlmSummaryDraft>,
    #[serde(default)]
    decisions: Vec<LlmDecisionDraft>,
    #[serde(default)]
    action_items: Vec<LlmActionItemDraft>,
    #[serde(default)]
    open_questions: Vec<LlmOpenQuestionDraft>,
    #[serde(default)]
    risks: Vec<LlmRiskDraft>,
    #[serde(default)]
    technical_recap: Option<LlmTechnicalRecapDraft>,
    #[serde(default)]
    follow_up_draft: Option<LlmFollowUpDraft>,
    #[serde(default)]
    timeline: Vec<LlmTimelineItemDraft>,
}

#[derive(Debug, Deserialize)]
struct LlmSummaryDraft {
    text: String,
    #[serde(default)]
    bullets: Vec<String>,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LlmDecisionDraft {
    decision: String,
    #[serde(default)]
    rationale: Option<String>,
    #[serde(default)]
    made_by_speaker_id: Option<String>,
    #[serde(default)]
    made_by_display_name: Option<String>,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LlmActionItemDraft {
    task: String,
    #[serde(default)]
    assignee_speaker_id: Option<String>,
    #[serde(default)]
    assignee_display_name: Option<String>,
    #[serde(default)]
    due_date: Option<String>,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LlmOpenQuestionDraft {
    question: String,
    #[serde(default)]
    asked_by_speaker_id: Option<String>,
    #[serde(default)]
    asked_by_display_name: Option<String>,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LlmRiskDraft {
    risk: String,
    #[serde(default)]
    severity: Option<RiskSeverity>,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LlmTechnicalRecapDraft {
    #[serde(default)]
    bullets: Vec<String>,
    #[serde(default)]
    mentioned_files: Vec<String>,
    #[serde(default)]
    mentioned_commands: Vec<String>,
    #[serde(default)]
    mentioned_errors: Vec<String>,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LlmFollowUpDraft {
    subject: String,
    body: String,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
    #[serde(default)]
    confidence: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct LlmTimelineItemDraft {
    #[serde(default)]
    timestamp_ms: Option<u64>,
    #[serde(default)]
    speaker_id: Option<String>,
    #[serde(default)]
    speaker_display_name: Option<String>,
    title: String,
    #[serde(default)]
    detail: String,
    #[serde(default)]
    evidence_segment_ids: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> MeetingIntelligenceInput {
        let mut first = TranscriptEntry::sourced(
            "session",
            TranscriptSource::Microphone,
            "You",
            "We decided to ship the meeting intelligence module. Please update src-tauri/src/meeting/runtime.rs by tomorrow.",
            0.9,
        );
        first.segment_id = "seg-1".to_string();
        first.speaker_id = Some(LOCAL_USER_SPEAKER_ID.to_string());
        first.speaker_label = Some("You".to_string());
        first.start_ms = Some(0);

        let mut second = TranscriptEntry::sourced(
            "session",
            TranscriptSource::SystemAudio,
            "Speaker 1",
            "What is the blocker if cargo test fails with a semantic_frame error?",
            0.86,
        );
        second.segment_id = "seg-2".to_string();
        second.speaker_id = Some(REMOTE_SPEAKER_1_ID.to_string());
        second.speaker_label = Some("Speaker 1".to_string());
        second.start_ms = Some(1_000);

        MeetingIntelligenceInput {
            session_id: "session".to_string(),
            transcript_entries: vec![first, second],
            speakers: vec![
                SpeakerLabel::source_default(TranscriptSource::Microphone),
                SpeakerLabel::source_default(TranscriptSource::SystemAudio),
            ],
            screen_contexts: Vec::new(),
            generation_options: MeetingIntelligenceGenerationOptions {
                use_local_llm: true,
                max_transcript_segments: 20,
            },
        }
    }

    fn italian_input() -> MeetingIntelligenceInput {
        let mut input = input();
        input.transcript_entries.clear();
        let mut first = TranscriptEntry::sourced(
            "session",
            TranscriptSource::Microphone,
            "You",
            "Ciao, dobbiamo chiudere questa milestone perché il riepilogo deve essere in italiano e non in inglese.",
            0.94,
        );
        first.segment_id = "seg-it-1".to_string();
        first.speaker_id = Some(LOCAL_USER_SPEAKER_ID.to_string());
        first.speaker_label = Some("You".to_string());
        first.start_ms = Some(0);
        let mut second = TranscriptEntry::sourced(
            "session",
            TranscriptSource::SystemAudio,
            "Speaker 1",
            "Abbiamo deciso che anche la bozza di follow-up deve usare un tono professionale italiano.",
            0.9,
        );
        second.segment_id = "seg-it-2".to_string();
        second.speaker_id = Some(REMOTE_SPEAKER_1_ID.to_string());
        second.speaker_label = Some("Speaker 1".to_string());
        second.start_ms = Some(1_000);
        input.transcript_entries = vec![first, second];
        input
    }

    #[test]
    fn prompt_includes_attached_screen_context_as_supplemental_evidence() {
        let baseline_prompt = build_meeting_llm_prompt_input(&input()).prompt;
        assert!(!baseline_prompt.contains("screen-context-1"));

        let mut with_context = input();
        with_context.screen_contexts.push(MeetingScreenContext {
            context_id: "screen-context-1".to_string(),
            session_id: "session".to_string(),
            captured_at: Utc::now(),
            source: ScreenContextSource::ManualCapture,
            attachment_mode: ScreenContextAttachmentMode::CurrentMoment,
            linked_transcript_segment_ids: vec!["seg-1".to_string()],
            linked_time_window: None,
            summary: "Visible screen shows a Cargo test failure in semantic_frame.rs".to_string(),
            structured_observation: None,
            screenshot_ref: None,
            redaction: ScreenContextRedaction::ScreenshotNotStored,
            confidence: 0.7,
            diagnostics: Vec::new(),
        });

        let prompt = build_meeting_llm_prompt_input(&with_context).prompt;
        assert!(prompt.contains("Attached screen context JSON"));
        assert!(prompt.contains("screen-context-1"));
        assert!(prompt.contains("semantic_frame.rs"));
        assert!(prompt.contains("supplemental evidence only"));
    }

    #[test]
    fn invalid_llm_json_falls_back_with_diagnostic() {
        let result = MeetingIntelligenceEngine::generate_with_llm_json_or_rule_based(
            input(),
            Some("{not valid json"),
            Some("test-model"),
        )
        .expect("fallback result");

        assert_eq!(result.status, MeetingIntelligenceStatus::Degraded);
        assert!(result.diagnostics.json_parse_failed);
        assert!(result.diagnostics.fallback_used);
        assert!(result.summary.is_some());
    }

    #[test]
    fn invalid_evidence_ids_are_sanitized_and_empty_artifacts_rejected() {
        let raw = r#"{
            "summary": {"text":"Grounded summary","bullets":["One"],"evidence_segment_ids":["seg-1","missing"],"confidence":0.8},
            "decisions": [{"decision":"Use the module","evidence_segment_ids":["missing"],"confidence":0.9}],
            "action_items": [{"task":"Update runtime","evidence_segment_ids":["seg-1"],"confidence":0.9}]
        }"#;

        let result = MeetingIntelligenceEngine::validate_llm_json(raw, &input(), "test-model")
            .expect("validated llm result");

        assert_eq!(result.diagnostics.invalid_evidence_ids, 2);
        assert_eq!(result.diagnostics.rejected_artifact_count, 1);
        assert!(result.decisions.is_empty());
        assert_eq!(result.action_items.len(), 1);
        assert_eq!(
            result.summary.unwrap().evidence_segment_ids,
            vec!["seg-1".to_string()]
        );
    }

    #[test]
    fn valid_llm_output_can_store_italian_followup() {
        let raw = r#"{
            "summary": {"text":"Sintesi fondata","bullets":["Decisione confermata"],"evidence_segment_ids":["seg-1"],"confidence":0.8},
            "follow_up_draft": {"subject":"Riepilogo incontro","body":"Ciao,\nabbiamo confermato la milestone e i prossimi passi.","evidence_segment_ids":["seg-1"],"confidence":0.82}
        }"#;

        let output = MeetingLlmRawOutput {
            raw_json: raw.to_string(),
            provider: "ollama".to_string(),
            model: "test-model".to_string(),
            stats: build_meeting_llm_prompt_input(&input()).stats,
            endpoint: Some("127.0.0.1:11434".to_string()),
            llm_generation_duration_ms: Some(12),
        };
        let result = MeetingIntelligenceEngine::generate_with_llm_output_or_rule_based(
            input(),
            Some(output),
            None,
        )
        .expect("llm result");

        assert_eq!(result.status, MeetingIntelligenceStatus::Generated);
        assert!(result.diagnostics.llm_used);
        assert!(!result.diagnostics.fallback_used);
        assert_eq!(result.diagnostics.model_provider.as_deref(), Some("ollama"));
        assert_eq!(
            result.diagnostics.llm_endpoint.as_deref(),
            Some("127.0.0.1:11434")
        );
        assert_eq!(
            result.diagnostics.detected_language,
            MeetingLanguage::English
        );
        assert!(result
            .follow_up_draft
            .as_ref()
            .is_some_and(|draft| draft.body.starts_with("Ciao")));
    }

    #[test]
    fn prompt_snapshot_is_latest_bounded_and_chronological() {
        let mut input = input();
        input.generation_options.max_transcript_segments = 2;
        for index in 3..=5 {
            let mut entry = TranscriptEntry::sourced(
                "session",
                TranscriptSource::Manual,
                "Simone",
                format!("Transcript segment {index}"),
                0.9,
            );
            entry.segment_id = format!("seg-{index}");
            entry.start_ms = Some((index as u64) * 1_000);
            input.transcript_entries.push(entry);
        }

        let prompt = build_meeting_llm_prompt_input(&input);

        assert_eq!(
            prompt
                .segments
                .iter()
                .map(|segment| segment.segment_id.as_str())
                .collect::<Vec<_>>(),
            vec!["seg-4", "seg-5"]
        );
        assert!(prompt.stats.input_truncated);
        assert!(prompt.prompt.contains("Output language:"));
    }

    #[test]
    fn detects_italian_transcript_language() {
        let detection = detect_meeting_language(&italian_input().transcript_entries);

        assert_eq!(detection.language, MeetingLanguage::Italian);
        assert!(detection.confidence > 0.5);
    }

    #[test]
    fn detects_english_transcript_language() {
        let detection = detect_meeting_language(&input().transcript_entries);

        assert_eq!(detection.language, MeetingLanguage::English);
    }

    #[test]
    fn mixed_transcript_prefers_local_user_language() {
        let mut input = input();
        input.transcript_entries[0].text =
            "Ciao, quindi dobbiamo fare questa modifica perché la sintesi deve essere italiana."
                .to_string();
        input.transcript_entries[0].source = TranscriptSource::Microphone;
        input.transcript_entries[0].speaker_id = Some(LOCAL_USER_SPEAKER_ID.to_string());
        input.transcript_entries[1].text =
            "The remote system audio contains English words and should not override the local user."
                .to_string();

        let detection = detect_meeting_language(&input.transcript_entries);

        assert_eq!(detection.language, MeetingLanguage::Italian);
        assert_eq!(detection.source, MeetingLanguageSource::UserSourceWeighted);
    }

    fn session_type_input(text: &str) -> Vec<TranscriptEntry> {
        let mut entry =
            TranscriptEntry::sourced("session", TranscriptSource::Microphone, "You", text, 0.9);
        entry.segment_id = "seg-session-type".to_string();
        entry.speaker_id = Some(LOCAL_USER_SPEAKER_ID.to_string());
        vec![entry]
    }

    #[test]
    fn detects_technical_debugging_session_type() {
        let entries = session_type_input(
            "The build has an error in src-tauri/src/meeting/runtime.rs and cargo test fails with a stack trace.",
        );
        let detection = detect_meeting_session_type(&entries);

        assert_eq!(
            detection.session_type,
            MeetingSessionType::TechnicalDebugging
        );
        assert!(detection.confidence > 0.4);
    }

    #[test]
    fn detects_planning_session_type() {
        let entries = session_type_input(
            "We need the roadmap, milestone, next step, sprint priority, and planning sequence.",
        );
        let detection = detect_meeting_session_type(&entries);

        assert_eq!(detection.session_type, MeetingSessionType::Planning);
    }

    #[test]
    fn detects_decision_review_session_type() {
        let entries = session_type_input(
            "We decided and confirmed the approved path, rejected the alternative, and will proceed after validation.",
        );
        let detection = detect_meeting_session_type(&entries);

        assert_eq!(detection.session_type, MeetingSessionType::DecisionReview);
    }

    #[test]
    fn detects_support_call_session_type() {
        let entries = session_type_input(
            "The customer reported an issue in the support ticket and the user problem must be reproduced.",
        );
        let detection = detect_meeting_session_type(&entries);

        assert_eq!(detection.session_type, MeetingSessionType::SupportCall);
    }

    #[test]
    fn detects_general_session_type_fallback() {
        let entries = session_type_input("Hello and thanks for the short conversation.");
        let detection = detect_meeting_session_type(&entries);

        assert_eq!(detection.session_type, MeetingSessionType::General);
    }

    #[test]
    fn prompt_includes_session_type_specific_instruction() {
        let prompt = build_meeting_llm_prompt_input(&input());

        assert_eq!(
            prompt.stats.session_type,
            MeetingSessionType::TechnicalDebugging
        );
        assert!(prompt
            .prompt
            .contains("Detected session type: TechnicalDebugging"));
        assert!(prompt.prompt.contains("technical debugging session"));
        assert!(prompt
            .prompt
            .contains("Summary: write a natural professional recap"));
    }

    #[test]
    fn prompt_instructs_language_without_changing_json_keys() {
        let prompt = build_meeting_llm_prompt_input(&italian_input());

        assert_eq!(prompt.stats.detected_language, MeetingLanguage::Italian);
        assert!(prompt
            .prompt
            .contains("Generate every user-facing string value in natural professional Italian"));
        assert!(prompt.prompt.contains("\"summary\""));
        assert!(prompt.prompt.contains("\"follow_up_draft\""));
    }

    #[test]
    fn detects_output_language_mismatch_for_italian_expected_english_output() {
        let raw = r#"{
            "summary": {"text":"Here is the meeting summary and follow up from the session.","bullets":["The team decided the next step."],"evidence_segment_ids":["seg-it-1"],"confidence":0.8}
        }"#;
        let result = MeetingIntelligenceEngine::validate_llm_json(raw, &italian_input(), "test")
            .expect("validated");

        assert_eq!(
            result.diagnostics.detected_language,
            MeetingLanguage::Italian
        );
        assert_eq!(result.diagnostics.output_language, MeetingLanguage::English);
        assert!(result.diagnostics.output_language_mismatch);
    }

    #[test]
    fn accepts_italian_output_for_italian_transcript() {
        let raw = r#"{
            "summary": {"text":"Durante la sessione abbiamo confermato il riepilogo italiano.","bullets":["Sintesi professionale in italiano"],"evidence_segment_ids":["seg-it-1"],"confidence":0.8}
        }"#;
        let result = MeetingIntelligenceEngine::validate_llm_json(raw, &italian_input(), "test")
            .expect("validated");

        assert_eq!(result.diagnostics.output_language, MeetingLanguage::Italian);
        assert!(!result.diagnostics.output_language_mismatch);
    }

    #[test]
    fn accepts_english_output_for_english_transcript() {
        let raw = r#"{
            "summary": {"text":"The team decided the meeting intelligence module is ready.","bullets":["The next step is validation."],"evidence_segment_ids":["seg-1"],"confidence":0.8}
        }"#;
        let result =
            MeetingIntelligenceEngine::validate_llm_json(raw, &input(), "test").expect("validated");

        assert_eq!(result.diagnostics.output_language, MeetingLanguage::English);
        assert!(!result.diagnostics.output_language_mismatch);
    }

    #[test]
    fn rule_based_followup_uses_italian_template_for_italian_transcript() {
        let result = MeetingIntelligenceEngine::generate(italian_input()).expect("rule based");
        let draft = result.follow_up_draft.expect("draft");

        assert_eq!(
            result.diagnostics.detected_language,
            MeetingLanguage::Italian
        );
        assert_eq!(draft.subject, "Riepilogo sessione");
        assert!(draft.body.starts_with("Buongiorno,"));
        assert!(draft.body.contains("Riepilogo:"));
        assert!(!draft.body.contains("Hi,"));
        assert!(!draft.body.contains("Best,"));
    }

    #[test]
    fn ollama_base_url_prefers_astra_env_value() {
        assert_eq!(
            configured_ollama_base_url_from(
                Some(" http://localhost:11435/ "),
                Some("http://localhost:11436")
            ),
            "http://localhost:11435"
        );
    }

    #[test]
    fn ollama_base_url_uses_ollama_host_when_astra_absent() {
        assert_eq!(
            configured_ollama_base_url_from(None, Some("localhost:11436/")),
            "http://localhost:11436"
        );
    }

    #[test]
    fn ollama_base_url_falls_back_to_local_default() {
        assert_eq!(
            configured_ollama_base_url_from(Some(" "), None),
            DEFAULT_OLLAMA_BASE_URL
        );
    }

    #[test]
    fn rule_based_intelligence_is_evidence_linked_and_technical() {
        let result = MeetingIntelligenceEngine::generate(input()).expect("rule based");

        assert_eq!(result.status, MeetingIntelligenceStatus::Generated);
        assert!(result
            .summary
            .as_ref()
            .is_some_and(|summary| !summary.evidence_segment_ids.is_empty()));
        assert!(result
            .action_items
            .iter()
            .all(|item| !item.evidence_segment_ids.is_empty()));
        assert!(result.technical_recap.as_ref().is_some_and(|recap| recap
            .mentioned_files
            .iter()
            .any(|file| file.ends_with(".rs"))));
        assert_eq!(result.timeline.len(), 2);
        assert_eq!(result.timeline[0].timestamp_ms, Some(0));
    }
}
