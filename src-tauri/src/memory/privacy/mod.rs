//! Privacy, redaction, and retention guardrails for AstraOS cognitive memory.
//!
//! This module intentionally stays deterministic and Rust-owned: the LLM can
//! propose memories, but memory persistence must pass through these local
//! guards before anything is written to the Memory Graph. The current guard is
//! conservative, dependency-free, and focused on preventing accidental storage
//! of common secrets/credentials while preserving enough context for governed
//! RAG.

use crate::memory::consolidation::{
    ConversationDecision, ConversationEntity, ConversationImportantPoint, ConversationMemoryBundle,
    ConversationPreference, ConversationProcedure, ConversationSemanticAtom,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const MAX_REDACTION_SAMPLES: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPrivacyDecision {
    Allowed,
    AllowedRedacted,
    RequiresUserReview,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRetentionClass {
    Transient,
    Episodic,
    LongTerm,
    UserProfile,
    SensitiveReview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPrivacyReport {
    pub schema_version: u32,
    pub decision: MemoryPrivacyDecision,
    pub retention_class: MemoryRetentionClass,
    pub redacted: bool,
    pub redaction_count: usize,
    pub sensitive_hint_count: usize,
    #[serde(default)]
    pub redaction_samples: Vec<String>,
    #[serde(default)]
    pub reasons: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

impl MemoryPrivacyReport {
    pub fn metadata(&self) -> Value {
        json!({
            "schema_version": self.schema_version,
            "decision": &self.decision,
            "retention_class": &self.retention_class,
            "redacted": self.redacted,
            "redaction_count": self.redaction_count,
            "sensitive_hint_count": self.sensitive_hint_count,
            "redaction_samples": &self.redaction_samples,
            "reasons": &self.reasons,
            "metadata": &self.metadata,
        })
    }
}

#[derive(Debug, Clone)]
struct RedactionOutcome {
    text: String,
    redactions: usize,
    samples: Vec<String>,
    sensitive_hints: usize,
}

/// Sanitizes an LLM-proposed conversation bundle before persistence.
///
/// The function does not authorize storage by itself; it returns a sanitized
/// bundle plus a report that the caller must persist as audit metadata. This
/// keeps the memory pipeline Rust-governed and makes future stronger policy
/// engines pluggable without changing the graph store contract.
pub fn sanitize_conversation_bundle(
    bundle: ConversationMemoryBundle,
) -> (ConversationMemoryBundle, MemoryPrivacyReport) {
    let durable_signal_count = durable_signal_count(&bundle);
    let mut total_redactions = 0usize;
    let mut total_sensitive_hints = 0usize;
    let mut samples = Vec::new();
    let mut reasons = Vec::new();

    let user = redact_sensitive_text(bundle.user_message.as_str());
    accumulate_outcome(&user, &mut total_redactions, &mut total_sensitive_hints, &mut samples);
    let assistant = redact_sensitive_text(bundle.assistant_answer.as_str());
    accumulate_outcome(&assistant, &mut total_redactions, &mut total_sensitive_hints, &mut samples);

    let topic = bundle.topic.map(|value| redact_string(value, &mut total_redactions, &mut total_sensitive_hints, &mut samples));
    let summary = bundle.summary.map(|value| redact_string(value, &mut total_redactions, &mut total_sensitive_hints, &mut samples));

    let semantic_atoms = bundle
        .semantic_atoms
        .into_iter()
        .map(|atom| sanitize_semantic_atom(atom, &mut total_redactions, &mut total_sensitive_hints, &mut samples))
        .collect::<Vec<_>>();
    let important_points = bundle
        .important_points
        .into_iter()
        .map(|point| sanitize_important_point(point, &mut total_redactions, &mut total_sensitive_hints, &mut samples))
        .collect::<Vec<_>>();
    let entities = bundle
        .entities
        .into_iter()
        .map(|entity| sanitize_entity(entity, &mut total_redactions, &mut total_sensitive_hints, &mut samples))
        .collect::<Vec<_>>();
    let preferences = bundle
        .preferences
        .into_iter()
        .map(|preference| sanitize_preference(preference, &mut total_redactions, &mut total_sensitive_hints, &mut samples))
        .collect::<Vec<_>>();
    let procedures = bundle
        .procedures
        .into_iter()
        .map(|procedure| sanitize_procedure(procedure, &mut total_redactions, &mut total_sensitive_hints, &mut samples))
        .collect::<Vec<_>>();
    let decisions = bundle
        .decisions
        .into_iter()
        .map(|decision| sanitize_decision(decision, &mut total_redactions, &mut total_sensitive_hints, &mut samples))
        .collect::<Vec<_>>();

    if total_redactions > 0 {
        reasons.push("sensitive-looking tokens or credential-like fields were redacted before memory persistence".into());
    }
    if durable_signal_count == 0 {
        reasons.push("conversation has no extracted durable semantic signal; store only as low-salience episodic context".into());
    }

    let retention_class = if total_sensitive_hints > 0 {
        MemoryRetentionClass::SensitiveReview
    } else if has_user_profile_signal(&semantic_atoms, &preferences) {
        MemoryRetentionClass::UserProfile
    } else if durable_signal_count > 0 {
        MemoryRetentionClass::LongTerm
    } else {
        MemoryRetentionClass::Episodic
    };

    let decision = if total_sensitive_hints > 0 {
        MemoryPrivacyDecision::AllowedRedacted
    } else {
        MemoryPrivacyDecision::Allowed
    };

    let mut metadata = bundle.metadata;
    if let Some(object) = metadata.as_object_mut() {
        object.insert("privacy_guard_applied".into(), Value::Bool(true));
    } else {
        metadata = json!({"privacy_guard_applied": true, "original_metadata": metadata});
    }

    let sanitized = ConversationMemoryBundle {
        request_id: bundle.request_id,
        source: bundle.source,
        user_message: user.text,
        assistant_answer: assistant.text,
        topic,
        summary,
        importance: bundle.importance,
        confidence: bundle.confidence,
        tags: bundle.tags,
        semantic_atoms,
        important_points,
        entities,
        preferences,
        procedures,
        decisions,
        metadata,
    };

    let report = MemoryPrivacyReport {
        schema_version: 1,
        decision,
        retention_class,
        redacted: total_redactions > 0,
        redaction_count: total_redactions,
        sensitive_hint_count: total_sensitive_hints,
        redaction_samples: samples.into_iter().take(MAX_REDACTION_SAMPLES).collect(),
        reasons,
        metadata: json!({
            "source": "rust_memory_privacy_guard",
            "durable_signal_count": durable_signal_count,
            "llm_cannot_bypass": true,
            "metadata_only": true,
        }),
    };

    (sanitized, report)
}

fn durable_signal_count(bundle: &ConversationMemoryBundle) -> usize {
    bundle.semantic_atoms.len()
        + bundle.important_points.len()
        + bundle.preferences.len()
        + bundle.procedures.len()
        + bundle.decisions.len()
}

fn has_user_profile_signal(atoms: &[ConversationSemanticAtom], preferences: &[ConversationPreference]) -> bool {
    !preferences.is_empty()
        || atoms.iter().any(|atom| {
            atom.tags.iter().any(|tag| {
                matches!(
                    normalize_key(tag).as_str(),
                    "user_profile" | "profile_fact" | "identity" | "preferred_name" | "name" | "preference"
                )
            }) || matches!(
                atom.kind.as_deref().map(normalize_key).unwrap_or_default().as_str(),
                "profile_fact" | "identity" | "name" | "preference" | "user_preference"
            )
        })
}

fn sanitize_semantic_atom(
    atom: ConversationSemanticAtom,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) -> ConversationSemanticAtom {
    ConversationSemanticAtom {
        title: atom.title.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        summary: atom.summary.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        subject: atom.subject.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        predicate: atom.predicate.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        object: atom.object.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        evidence: atom.evidence.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        kind: atom.kind,
        confidence: atom.confidence,
        tags: atom.tags,
        metadata: atom.metadata,
    }
}

fn sanitize_important_point(
    point: ConversationImportantPoint,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) -> ConversationImportantPoint {
    ConversationImportantPoint {
        title: redact_string(point.title, total_redactions, total_sensitive_hints, samples),
        summary: redact_string(point.summary, total_redactions, total_sensitive_hints, samples),
        kind: point.kind,
        confidence: point.confidence,
        tags: point.tags,
        metadata: point.metadata,
    }
}

fn sanitize_entity(
    entity: ConversationEntity,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) -> ConversationEntity {
    ConversationEntity {
        name: redact_string(entity.name, total_redactions, total_sensitive_hints, samples),
        entity_type: entity.entity_type,
        summary: entity.summary.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        confidence: entity.confidence,
        metadata: entity.metadata,
    }
}

fn sanitize_preference(
    preference: ConversationPreference,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) -> ConversationPreference {
    ConversationPreference {
        preference: redact_string(preference.preference, total_redactions, total_sensitive_hints, samples),
        rationale: preference.rationale.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        confidence: preference.confidence,
        metadata: preference.metadata,
    }
}

fn sanitize_procedure(
    procedure: ConversationProcedure,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) -> ConversationProcedure {
    ConversationProcedure {
        title: redact_string(procedure.title, total_redactions, total_sensitive_hints, samples),
        steps: procedure
            .steps
            .into_iter()
            .map(|step| redact_string(step, total_redactions, total_sensitive_hints, samples))
            .collect(),
        rationale: procedure.rationale.map(|value| redact_string(value, total_redactions, total_sensitive_hints, samples)),
        confidence: procedure.confidence,
        metadata: procedure.metadata,
    }
}

fn sanitize_decision(
    decision: ConversationDecision,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) -> ConversationDecision {
    ConversationDecision {
        title: redact_string(decision.title, total_redactions, total_sensitive_hints, samples),
        summary: redact_string(decision.summary, total_redactions, total_sensitive_hints, samples),
        confidence: decision.confidence,
        metadata: decision.metadata,
    }
}

fn redact_string(
    value: String,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) -> String {
    let outcome = redact_sensitive_text(&value);
    accumulate_outcome(&outcome, total_redactions, total_sensitive_hints, samples);
    outcome.text
}

fn accumulate_outcome(
    outcome: &RedactionOutcome,
    total_redactions: &mut usize,
    total_sensitive_hints: &mut usize,
    samples: &mut Vec<String>,
) {
    *total_redactions += outcome.redactions;
    *total_sensitive_hints += outcome.sensitive_hints;
    for sample in &outcome.samples {
        if samples.len() >= MAX_REDACTION_SAMPLES {
            break;
        }
        samples.push(sample.clone());
    }
}

fn redact_sensitive_text(input: &str) -> RedactionOutcome {
    let mut redactions = 0usize;
    let mut sensitive_hints = 0usize;
    let mut samples = Vec::new();
    let mut output_lines = Vec::new();

    for line in input.lines() {
        let lower = line.to_ascii_lowercase();
        if looks_like_sensitive_assignment(&lower) {
            redactions += 1;
            sensitive_hints += 1;
            push_sample(&mut samples, line);
            output_lines.push(redact_assignment_line(line));
        } else {
            let (line_out, count, hints, line_samples) = redact_inline_tokens(line);
            redactions += count;
            sensitive_hints += hints;
            for sample in line_samples {
                push_sample(&mut samples, &sample);
            }
            output_lines.push(line_out);
        }
    }

    RedactionOutcome {
        text: output_lines.join("\n"),
        redactions,
        samples,
        sensitive_hints,
    }
}

fn looks_like_sensitive_assignment(lower: &str) -> bool {
    let has_key = [
        "password", "passwd", "pwd", "secret", "api_key", "apikey", "access_token", "refresh_token",
        "private_key", "client_secret", "authorization", "bearer ", "ssh-rsa", "BEGIN PRIVATE KEY",
    ]
    .iter()
    .any(|needle| lower.contains(&needle.to_ascii_lowercase()));
    has_key && (lower.contains('=') || lower.contains(':') || lower.contains("bearer "))
}

fn redact_assignment_line(line: &str) -> String {
    if let Some((left, _)) = line.split_once('=') {
        return format!("{}=[REDACTED_BY_ASTRA_MEMORY_PRIVACY_GUARD]", left.trim_end());
    }
    if let Some((left, _)) = line.split_once(':') {
        return format!("{}: [REDACTED_BY_ASTRA_MEMORY_PRIVACY_GUARD]", left.trim_end());
    }
    "[REDACTED_BY_ASTRA_MEMORY_PRIVACY_GUARD]".into()
}

fn redact_inline_tokens(line: &str) -> (String, usize, usize, Vec<String>) {
    let mut count = 0usize;
    let mut hints = 0usize;
    let mut samples = Vec::new();
    let tokens = line
        .split_whitespace()
        .map(|token| {
            let trimmed = token.trim_matches(|c: char| matches!(c, ',' | ';' | ')' | '(' | ']' | '[' | '"' | '\''));
            if looks_like_secret_token(trimmed) {
                count += 1;
                hints += 1;
                samples.push(trimmed.chars().take(24).collect::<String>());
                token.replace(trimmed, "[REDACTED_BY_ASTRA_MEMORY_PRIVACY_GUARD]")
            } else {
                token.to_string()
            }
        })
        .collect::<Vec<_>>();
    (tokens.join(" "), count, hints, samples)
}

fn looks_like_secret_token(token: &str) -> bool {
    let lower = token.to_ascii_lowercase();
    lower.starts_with("sk-")
        || lower.starts_with("xoxb-")
        || lower.starts_with("ghp_")
        || lower.starts_with("github_pat_")
        || lower.starts_with("bearer_")
        || (token.len() >= 32 && token.chars().any(|c| c.is_ascii_digit()) && token.chars().any(|c| c.is_ascii_alphabetic()) && mostly_token_chars(token))
}

fn mostly_token_chars(token: &str) -> bool {
    let allowed = token
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | '+'))
        .count();
    allowed.saturating_mul(100) / token.len().max(1) >= 92
}

fn push_sample(samples: &mut Vec<String>, sample: &str) {
    if samples.len() >= MAX_REDACTION_SAMPLES {
        return;
    }
    let clipped = sample.trim().chars().take(80).collect::<String>();
    if !clipped.is_empty() {
        samples.push(clipped);
    }
}

fn normalize_key(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(|c: char| c == ' ' || c == '-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_secret_assignments() {
        let outcome = redact_sensitive_text("api_key=sk-test-secret\nnormal line");
        assert!(outcome.redactions >= 1);
        assert!(outcome.text.contains("[REDACTED_BY_ASTRA_MEMORY_PRIVACY_GUARD]"));
        assert!(outcome.text.contains("normal line"));
    }
}
