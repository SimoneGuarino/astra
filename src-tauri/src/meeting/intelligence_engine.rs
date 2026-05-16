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
    time::Duration,
};

const MAX_SUMMARY_BULLETS: usize = 6;
const MAX_DECISIONS: usize = 24;
const MAX_ACTION_ITEMS: usize = 40;
const MAX_OPEN_QUESTIONS: usize = 40;
const MAX_RISKS: usize = 30;
const MAX_TIMELINE_ITEMS: usize = 40;
const MAX_TECHNICAL_ITEMS: usize = 12;
const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";
const DEFAULT_MEETING_LLM_CANDIDATES: &str = "gpt-oss:20b,qwen3:14b,qwen3:8b,llama3.1:8b";
const DEFAULT_PROMPT_MAX_CHARS_TOTAL: usize = 24_000;
const DEFAULT_PROMPT_MAX_CHARS_PER_SEGMENT: usize = 900;
const DEFAULT_MEETING_LLM_TIMEOUT_SECS: u64 = 45;

#[derive(Debug, Clone)]
pub struct MeetingIntelligenceInput {
    pub session_id: String,
    pub transcript_entries: Vec<TranscriptEntry>,
    pub speakers: Vec<SpeakerLabel>,
    pub generation_options: MeetingIntelligenceGenerationOptions,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeetingLlmPromptStats {
    pub input_segment_count: usize,
    pub input_truncated: bool,
    pub input_char_count: usize,
    pub max_segments: usize,
    pub max_chars_total: usize,
    pub max_chars_per_segment: usize,
    pub included_segment_ids: Vec<String>,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeetingLlmErrorKind {
    Unavailable,
    Timeout,
    Http,
    InvalidResponse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeetingLlmError {
    pub kind: MeetingLlmErrorKind,
    pub message: String,
    pub provider: String,
    pub model: Option<String>,
    pub stats: MeetingLlmPromptStats,
}

impl MeetingLlmError {
    pub fn unavailable(message: impl Into<String>, stats: MeetingLlmPromptStats) -> Self {
        Self {
            kind: MeetingLlmErrorKind::Unavailable,
            message: bounded_error_message(message.into()),
            provider: "ollama".to_string(),
            model: None,
            stats,
        }
    }

    fn with_kind(
        kind: MeetingLlmErrorKind,
        message: impl Into<String>,
        model: Option<String>,
        stats: MeetingLlmPromptStats,
    ) -> Self {
        Self {
            kind,
            message: bounded_error_message(message.into()),
            provider: "ollama".to_string(),
            model,
            stats,
        }
    }

    fn reason_code(&self) -> String {
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
}

impl OllamaMeetingIntelligenceLlm {
    pub fn new() -> Self {
        let timeout_secs = env::var("ASTRA_MEETING_INTELLIGENCE_TIMEOUT_SECS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| (1..=300).contains(value))
            .unwrap_or(DEFAULT_MEETING_LLM_TIMEOUT_SECS);
        Self {
            client: Client::new(),
            timeout: Duration::from_secs(timeout_secs),
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
            .get(format!("{OLLAMA_BASE_URL}/api/tags"))
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
            .post(format!("{OLLAMA_BASE_URL}/api/chat"))
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
            let stats = input.stats.clone();
            let model = match self.select_model().await {
                Ok(model) => model,
                Err(error) => {
                    return Err(MeetingLlmError::unavailable(error, stats));
                }
            };
            let call = self.call_model(&model, &input);
            match tokio::time::timeout(self.timeout, call).await {
                Ok(Ok(raw_json)) => Ok(MeetingLlmRawOutput {
                    raw_json,
                    provider: "ollama".to_string(),
                    model,
                    stats: input.stats,
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
                    ))
                }
                Err(_) => Err(MeetingLlmError::with_kind(
                    MeetingLlmErrorKind::Timeout,
                    "Ollama meeting intelligence request timed out",
                    Some(model),
                    input.stats,
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
            apply_prompt_stats(&mut fallback.diagnostics, &error.stats);
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
                result.diagnostics.llm_used = true;
                result.diagnostics.fallback_used = false;
                apply_prompt_stats(&mut result.diagnostics, &output.stats);
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
                apply_prompt_stats(&mut fallback.diagnostics, &output.stats);
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
                return Err(base_diagnostics(
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
                ));
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

        Ok(MeetingIntelligenceResult {
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
        })
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

        let diagnostics = base_diagnostics(
            status.clone(),
            generator,
            unavailable_reason,
            false,
            0,
            0,
            status == MeetingIntelligenceStatus::Degraded,
            warnings,
        );

        MeetingIntelligenceResult {
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
        }
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
    };
    let prompt = meeting_llm_user_prompt(input, &selected_rev, &stats);
    MeetingLlmPromptInput {
        session_id: input.session_id.clone(),
        prompt,
        segments: selected_rev,
        stats,
    }
}

fn prompt_stats_for_input(input: &MeetingIntelligenceInput) -> MeetingLlmPromptStats {
    let bounded = bounded_input(input.clone());
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

    format!(
        r#"Generate transcript-backed meeting intelligence for session "{session_id}".

Language:
- Generate artifacts in the dominant language of the transcript.
- If mixed, prefer the user's/local microphone language when clear.

Bounded input:
- input_segment_count: {input_segment_count}
- input_truncated: {input_truncated}
- input_char_count: {input_char_count}
- max_segments: {max_segments}
- max_chars_total: {max_chars_total}
- max_chars_per_segment: {max_chars_per_segment}

Speaker registry JSON:
{speaker_json}

Transcript entries JSON:
{transcript_json}

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
- Omit decisions/action_items/open_questions/risks that are not directly supported by evidence.
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
        speaker_json = speaker_json,
        transcript_json = transcript_json
    )
}

fn configured_usize(key: &str, default: usize, min: usize, max: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .map(|value| value.clamp(min, max))
        .unwrap_or(default)
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
    let mut bullets = Vec::new();
    bullets.push(format!(
        "Transcript contains {} grounded segments from {}.",
        input.transcript_entries.len(),
        if speakers.is_empty() {
            "unknown speakers".to_string()
        } else {
            speakers.join(", ")
        }
    ));
    if !topics.is_empty() {
        bullets.push(format!("Main repeated terms: {}.", topics.join(", ")));
    }
    bullets.push(format!(
        "Derived artifacts are evidence-linked to {} transcript segment(s).",
        evidence.len()
    ));

    MeetingSummary {
        id: new_meeting_artifact_id(),
        session_id: input.session_id.clone(),
        text: bullets.join(" "),
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
    let mut body =
        String::from("Hi,\n\nHere is the transcript-backed recap from the work session.\n\n");
    if let Some(summary) = summary {
        body.push_str("Summary:\n");
        for bullet in &summary.bullets {
            body.push_str("- ");
            body.push_str(bullet);
            body.push('\n');
        }
        body.push('\n');
    }
    if !decisions.is_empty() {
        body.push_str("Decisions:\n");
        for decision in decisions.iter().take(6) {
            body.push_str("- ");
            body.push_str(&decision.decision);
            body.push('\n');
        }
        body.push('\n');
    }
    if !action_items.is_empty() {
        body.push_str("Action items:\n");
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
        body.push_str("Open questions:\n");
        for question in questions.iter().take(6) {
            body.push_str("- ");
            body.push_str(&question.question);
            body.push('\n');
        }
        body.push('\n');
    }
    if !risks.is_empty() {
        body.push_str("Risks / blockers:\n");
        for risk in risks.iter().take(6) {
            body.push_str("- ");
            body.push_str(&risk.risk);
            body.push('\n');
        }
        body.push('\n');
    }
    body.push_str("Best,\n");

    MeetingFollowUpDraft {
        id: new_meeting_artifact_id(),
        session_id: input.session_id.clone(),
        subject: "Follow-up: meeting recap".to_string(),
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
        .take(MAX_TIMELINE_ITEMS)
        .map(|entry| MeetingTimelineItem {
            id: new_meeting_artifact_id(),
            timestamp_ms: entry.start_ms,
            speaker_id: entry.speaker_id.clone(),
            speaker_display_name: Some(entry.speaker_display_name().to_string()),
            title: format!(
                "{}: {}",
                entry.speaker_display_name(),
                bounded_text(&entry.text, 80)
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
            generation_options: MeetingIntelligenceGenerationOptions {
                use_local_llm: true,
                max_transcript_segments: 20,
            },
        }
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
        assert!(prompt
            .prompt
            .contains("Generate artifacts in the dominant language of the transcript"));
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
