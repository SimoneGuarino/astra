//! Transcript-backed meeting intelligence.
//!
//! This module treats transcript entries as the source of truth. Model output,
//! when connected in a future milestone, must pass through the same schema and
//! evidence validation before it can be stored.

use super::{action_item_tracker::ActionItemTracker, decision_log::DecisionLog, types::*};
use chrono::Utc;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

const MAX_SUMMARY_BULLETS: usize = 6;
const MAX_TIMELINE_ITEMS: usize = 40;
const MAX_TECHNICAL_ITEMS: usize = 12;

#[derive(Debug, Clone)]
pub struct MeetingIntelligenceInput {
    pub session_id: String,
    pub transcript_entries: Vec<TranscriptEntry>,
    pub speakers: Vec<SpeakerLabel>,
    pub generation_options: MeetingIntelligenceGenerationOptions,
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
        if input.transcript_entries.is_empty() {
            return Err(MeetingRuntimeError::InvalidConfig {
                message: "meeting intelligence requires at least one transcript segment"
                    .to_string(),
            });
        }
        let input = bounded_input(input);

        let Some(raw_json) = llm_json else {
            let mut warnings = vec![
                "Local LLM generation was requested but no governed meeting LLM adapter is connected; rule-based fallback was used".to_string(),
            ];
            if !input.generation_options.use_local_llm {
                warnings.clear();
            }
            return Ok(Self::rule_based_result(
                &input,
                ArtifactGenerator::RuleBased,
                if warnings.is_empty() {
                    MeetingIntelligenceStatus::Generated
                } else {
                    MeetingIntelligenceStatus::Degraded
                },
                Some("meeting_local_llm_adapter_not_connected".to_string())
                    .filter(|_| input.generation_options.use_local_llm),
                warnings,
            ));
        };

        match Self::validate_llm_json(raw_json, &input, model_name.unwrap_or("local")) {
            Ok(result) => Ok(result),
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

        let diagnostics = base_diagnostics(
            MeetingIntelligenceStatus::Generated,
            generator.clone(),
            None,
            false,
            stats.invalid_evidence_ids,
            stats.rejected_artifact_count,
            false,
            stats.warnings,
        );

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
        model_unavailable_reason: unavailable_reason,
        json_parse_failed,
        invalid_evidence_ids,
        rejected_artifact_count,
        fallback_used,
        transcript_text_logged: false,
        audit_redacted: true,
        warnings,
        generated_at: Utc::now(),
    }
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
