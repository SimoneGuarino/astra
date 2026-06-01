use crate::memory::{
    errors::{MemoryError, MemoryResult},
    privacy::{sanitize_conversation_bundle, MemoryPrivacyDecision, MemoryRetentionClass},
    store::MemoryGraphStore,
    types::{
        CreateMemoryEdgeRequest, CreateMemoryNodeRequest, MemoryActivation,
        MemoryActivationRequest, MemoryNode, MemoryNodeKind, MemoryRelationKind,
        MemoryVerificationStatus,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const MAX_IMPORTANT_POINTS: usize = 18;
const MAX_SEMANTIC_ATOMS: usize = 24;
const MAX_ENTITIES: usize = 24;
const MAX_PREFERENCES: usize = 12;
const MAX_PROCEDURES: usize = 10;
const MAX_DECISIONS: usize = 12;
const MAX_TITLE_CHARS: usize = 180;
const MAX_SUMMARY_CHARS: usize = 3_000;
const MAX_CONTENT_CHARS: usize = 20_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMemoryBundle {
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    pub user_message: String,
    pub assistant_answer: String,
    #[serde(default)]
    pub topic: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub semantic_atoms: Vec<ConversationSemanticAtom>,
    #[serde(default)]
    pub important_points: Vec<ConversationImportantPoint>,
    #[serde(default)]
    pub entities: Vec<ConversationEntity>,
    #[serde(default)]
    pub preferences: Vec<ConversationPreference>,
    #[serde(default)]
    pub procedures: Vec<ConversationProcedure>,
    #[serde(default)]
    pub decisions: Vec<ConversationDecision>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSemanticAtom {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub predicate: Option<String>,
    #[serde(default)]
    pub object: Option<String>,
    #[serde(default)]
    pub evidence: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationImportantPoint {
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEntity {
    pub name: String,
    #[serde(default)]
    pub entity_type: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationPreference {
    pub preference: String,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationProcedure {
    pub title: String,
    #[serde(default)]
    pub steps: Vec<String>,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationDecision {
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMemoryConsolidationReceipt {
    pub accepted: bool,
    pub reason: String,
    pub turn_node: MemoryNode,
    pub created_node_ids: Vec<String>,
    pub created_edge_ids: Vec<String>,
    #[serde(default)]
    pub activation: Option<MemoryActivation>,
    pub summary: Value,
}

pub fn consolidate_conversation_bundle(
    store: &MemoryGraphStore,
    bundle: ConversationMemoryBundle,
) -> MemoryResult<ConversationMemoryConsolidationReceipt> {
    validate_bundle(&bundle)?;
    let (bundle, privacy_report) = sanitize_conversation_bundle(bundle);
    if matches!(&privacy_report.decision, MemoryPrivacyDecision::Blocked) {
        return Err(MemoryError::Validation(
            "conversation memory was blocked by Rust privacy guard".into(),
        ));
    }

    let bundle_hash = stable_bundle_hash(&bundle)?;
    let topic_raw = bundle
        .topic
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| fallback_topic(&bundle.user_message));
    let topic = cap_text(topic_raw, MAX_TITLE_CHARS);
    let confidence = clamp01(bundle.confidence.unwrap_or(0.72));
    let importance = clamp01(bundle.importance.unwrap_or(0.5));
    let mut base_defaults = vec!["conversation", "episodic_memory"];
    if matches!(&privacy_report.retention_class, MemoryRetentionClass::SensitiveReview) {
        base_defaults.push("sensitive_review");
    }
    if privacy_report.redacted {
        base_defaults.push("privacy_redacted");
    }
    if durable_memory_signal_count(&bundle) == 0 {
        base_defaults.push("low_signal_episode");
        base_defaults.push("episode_only");
    }
    let base_tags = normalize_tags(bundle.tags.clone(), &base_defaults);
    let turn_summary = bundle
        .summary
        .clone()
        .unwrap_or_else(|| fallback_summary(&bundle.user_message, &bundle.assistant_answer));

    let turn_node = store.create_node_once_by_source(CreateMemoryNodeRequest {
        kind: MemoryNodeKind::ConversationTurn,
        title: topic.clone(),
        summary: cap_text(turn_summary, MAX_SUMMARY_CHARS),
        content: Some(cap_text(render_turn_content(&bundle), MAX_CONTENT_CHARS)),
        tags: base_tags.clone(),
        source: Some(format!(
            "conversation_turn:{}",
            bundle
                .request_id
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(&bundle_hash)
        )),
        confidence,
        verification_status: MemoryVerificationStatus::LlmInferred,
        salience: conversation_turn_salience(importance, durable_memory_signal_count(&bundle), &privacy_report),
        metadata: json!({
            "ingestion_source": "conversation_memory_consolidation",
            "schema_version": 1,
            "request_id": bundle.request_id.clone(),
            "source": bundle.source.clone(),
            "bundle_hash": bundle_hash,
            "importance": importance,
            "semantic_atom_count": bundle.semantic_atoms.len(),
            "important_point_count": bundle.important_points.len(),
            "entity_count": bundle.entities.len(),
            "preference_count": bundle.preferences.len(),
            "procedure_count": bundle.procedures.len(),
            "decision_count": bundle.decisions.len(),
            "durable_signal_count": durable_memory_signal_count(&bundle),
            "privacy": privacy_report.metadata(),
            "retention_class": &privacy_report.retention_class,
            "governance_note": "conversation memory is advisory context only; actions still require governed tools and policy validation",
        }),
    })?;

    let mut created_node_ids = vec![turn_node.id.clone()];
    let mut created_edge_ids = Vec::new();

    for (index, atom) in bundle.semantic_atoms.iter().take(MAX_SEMANTIC_ATOMS).enumerate() {
        let title = semantic_atom_title(atom);
        let summary = semantic_atom_summary(atom);
        if title.trim().is_empty() || summary.trim().is_empty() {
            continue;
        }
        let atom_hash = short_hash(&format!(
            "{}:{}:{}:{}:{}",
            bundle_hash,
            index,
            atom.subject.as_deref().unwrap_or_default(),
            atom.predicate.as_deref().unwrap_or_default(),
            atom.object.as_deref().unwrap_or_default()
        ));
        let tags = normalize_tags(atom.tags.clone(), &["conversation", "semantic_atom", "long_term_memory"]);
        let is_profile_like = tags.iter().any(|tag| matches!(tag.as_str(), "user_profile" | "identity" | "name" | "profile_fact"));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: semantic_atom_kind(atom.kind.as_deref(), &tags),
            title: cap_text(title, MAX_TITLE_CHARS),
            summary: cap_text(summary, MAX_SUMMARY_CHARS),
            content: Some(cap_text(render_semantic_atom_content(atom, &bundle), MAX_CONTENT_CHARS)),
            tags,
            source: Some(format!("conversation_semantic_atom:{}:{}", bundle_hash, atom_hash)),
            confidence: clamp01(atom.confidence.unwrap_or(confidence).max(if is_profile_like { 0.78 } else { 0.58 })),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: if is_profile_like { 0.94 } else { 0.76 },
            metadata: json!({
                "ingestion_source": "conversation_memory_consolidation",
                "conversation_turn_id": turn_node.id.clone(),
                "atom_kind": atom.kind.clone(),
                "subject": atom.subject.clone(),
                "predicate": atom.predicate.clone(),
                "object_present": atom.object.as_ref().is_some_and(|value| !value.trim().is_empty()),
                "evidence_present": atom.evidence.as_ref().is_some_and(|value| !value.trim().is_empty()),
                "llm_semantic_distillation": true,
                "requires_user_control": is_profile_like,
                "metadata": atom.metadata.clone(),
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: turn_node.id.clone(),
            relation: MemoryRelationKind::DerivedFrom,
            weight: if is_profile_like { 0.92 } else { 0.82 },
            confidence: 0.82,
            metadata: json!({"ingestion_source": "conversation_semantic_atom_derived_from_turn"}),
        })?;
        created_node_ids.push(node.id);
        created_edge_ids.push(edge.id);
    }

    for (index, point) in bundle.important_points.iter().take(MAX_IMPORTANT_POINTS).enumerate() {
        if point.title.trim().is_empty() || point.summary.trim().is_empty() {
            continue;
        }
        let point_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, point.title));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: point_kind(point.kind.as_deref()),
            title: cap_text(point.title.trim(), MAX_TITLE_CHARS),
            summary: cap_text(point.summary.trim(), MAX_SUMMARY_CHARS),
            content: None,
            tags: normalize_tags(point.tags.clone(), &["conversation", "important_point"]),
            source: Some(format!("conversation_point:{}:{}", bundle_hash, point_hash)),
            confidence: clamp01(point.confidence.unwrap_or(confidence)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.68,
            metadata: json!({
                "ingestion_source": "conversation_memory_consolidation",
                "conversation_turn_id": turn_node.id.clone(),
                "point_kind": point.kind.clone(),
                "metadata": point.metadata.clone(),
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: turn_node.id.clone(),
            relation: MemoryRelationKind::DerivedFrom,
            weight: 0.78,
            confidence: 0.76,
            metadata: json!({"ingestion_source": "conversation_point_derived_from_turn"}),
        })?;
        created_node_ids.push(node.id);
        created_edge_ids.push(edge.id);
    }

    for (index, entity) in bundle.entities.iter().take(MAX_ENTITIES).enumerate() {
        if entity.name.trim().is_empty() {
            continue;
        }
        let entity_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, entity.name));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::Entity,
            title: cap_text(entity.name.trim(), MAX_TITLE_CHARS),
            summary: cap_text(
                entity
                    .summary
                    .clone()
                    .unwrap_or_else(|| format!("Entity mentioned in conversation: {}", entity.name.trim())),
                MAX_SUMMARY_CHARS,
            ),
            content: None,
            tags: normalize_tags(
                vec![entity.entity_type.clone().unwrap_or_else(|| "entity".into())],
                &["conversation_entity"],
            ),
            source: Some(format!("conversation_entity:{}:{}", bundle_hash, entity_hash)),
            confidence: clamp01(entity.confidence.unwrap_or(0.68)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.54,
            metadata: json!({
                "ingestion_source": "conversation_memory_consolidation",
                "conversation_turn_id": turn_node.id.clone(),
                "entity_type": entity.entity_type.clone(),
                "metadata": entity.metadata.clone(),
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: turn_node.id.clone(),
            to_node_id: node.id.clone(),
            relation: MemoryRelationKind::Mentions,
            weight: 0.58,
            confidence: 0.7,
            metadata: json!({"ingestion_source": "conversation_turn_mentions_entity"}),
        })?;
        created_node_ids.push(node.id);
        created_edge_ids.push(edge.id);
    }

    for (index, preference) in bundle.preferences.iter().take(MAX_PREFERENCES).enumerate() {
        if preference.preference.trim().is_empty() {
            continue;
        }
        let preference_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, preference.preference));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::UserPreference,
            title: cap_text(preference.preference.trim(), MAX_TITLE_CHARS),
            summary: cap_text(
                preference
                    .rationale
                    .clone()
                    .unwrap_or_else(|| preference.preference.trim().to_string()),
                MAX_SUMMARY_CHARS,
            ),
            content: preference.rationale.clone().map(|value| cap_text(value, MAX_CONTENT_CHARS)),
            tags: normalize_tags(vec!["preference".into(), "conversation".into()], &[]),
            source: Some(format!("conversation_preference:{}:{}", bundle_hash, preference_hash)),
            confidence: clamp01(preference.confidence.unwrap_or(0.74)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.82,
            metadata: json!({
                "ingestion_source": "conversation_memory_consolidation",
                "conversation_turn_id": turn_node.id.clone(),
                "metadata": preference.metadata.clone(),
                "requires_user_control": true,
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: turn_node.id.clone(),
            relation: MemoryRelationKind::PreferredByUser,
            weight: 0.88,
            confidence: 0.78,
            metadata: json!({"ingestion_source": "conversation_preference_from_turn"}),
        })?;
        created_node_ids.push(node.id);
        created_edge_ids.push(edge.id);
    }

    for (index, procedure) in bundle.procedures.iter().take(MAX_PROCEDURES).enumerate() {
        if procedure.title.trim().is_empty() {
            continue;
        }
        let steps = procedure
            .steps
            .iter()
            .map(|step| step.trim())
            .filter(|step| !step.is_empty())
            .collect::<Vec<_>>();
        let procedure_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, procedure.title));
        let summary = if steps.is_empty() {
            procedure.rationale.clone().unwrap_or_else(|| procedure.title.clone())
        } else {
            format!("{} steps: {}", steps.len(), steps.iter().take(6).cloned().collect::<Vec<_>>().join(" -> "))
        };
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::Procedure,
            title: cap_text(procedure.title.trim(), MAX_TITLE_CHARS),
            summary: cap_text(summary, MAX_SUMMARY_CHARS),
            content: Some(cap_text(render_procedure_content(procedure), MAX_CONTENT_CHARS)),
            tags: normalize_tags(vec!["procedure".into(), "procedural_memory".into(), "conversation".into()], &[]),
            source: Some(format!("conversation_procedure:{}:{}", bundle_hash, procedure_hash)),
            confidence: clamp01(procedure.confidence.unwrap_or(0.68)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.78,
            metadata: json!({
                "ingestion_source": "conversation_memory_consolidation",
                "conversation_turn_id": turn_node.id.clone(),
                "step_count": steps.len(),
                "metadata": procedure.metadata.clone(),
                "governance_note": "procedural memory is advisory context only; no autonomous action is authorized",
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: turn_node.id.clone(),
            relation: MemoryRelationKind::LearnedFrom,
            weight: 0.82,
            confidence: 0.78,
            metadata: json!({"ingestion_source": "conversation_procedure_learned_from_turn"}),
        })?;
        created_node_ids.push(node.id);
        created_edge_ids.push(edge.id);
    }

    for (index, decision) in bundle.decisions.iter().take(MAX_DECISIONS).enumerate() {
        if decision.title.trim().is_empty() || decision.summary.trim().is_empty() {
            continue;
        }
        let decision_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, decision.title));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::Decision,
            title: cap_text(decision.title.trim(), MAX_TITLE_CHARS),
            summary: cap_text(decision.summary.trim(), MAX_SUMMARY_CHARS),
            content: None,
            tags: normalize_tags(vec!["decision".into(), "conversation".into()], &[]),
            source: Some(format!("conversation_decision:{}:{}", bundle_hash, decision_hash)),
            confidence: clamp01(decision.confidence.unwrap_or(0.7)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.74,
            metadata: json!({
                "ingestion_source": "conversation_memory_consolidation",
                "conversation_turn_id": turn_node.id.clone(),
                "metadata": decision.metadata.clone(),
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: turn_node.id.clone(),
            relation: MemoryRelationKind::DerivedFrom,
            weight: 0.78,
            confidence: 0.76,
            metadata: json!({"ingestion_source": "conversation_decision_derived_from_turn"}),
        })?;
        created_node_ids.push(node.id);
        created_edge_ids.push(edge.id);
    }

    created_node_ids.sort();
    created_node_ids.dedup();
    created_edge_ids.sort();
    created_edge_ids.dedup();

    let activation = store
        .activate(MemoryActivationRequest {
            request_id: bundle.request_id.clone(),
            root_query: bundle.user_message.clone(),
            seed_node_ids: vec![turn_node.id.clone()],
            max_depth: 2,
            max_nodes: 32,
            metadata: json!({
                "activation_source": "conversation_memory_consolidation",
                "ui_hint": "electricity_reached_conversation_nodes",
                "topic": topic,
                "metadata_only": true,
            }),
        })
        .ok();

    Ok(ConversationMemoryConsolidationReceipt {
        accepted: true,
        reason: "conversation memory bundle consolidated into typed Memory Graph nodes and relations".into(),
        turn_node,
        created_node_ids: created_node_ids.clone(),
        created_edge_ids: created_edge_ids.clone(),
        activation,
        summary: json!({
            "schema_version": 1,
            "created_nodes": created_node_ids.len(),
            "created_edges": created_edge_ids.len(),
            "semantic_atoms": bundle.semantic_atoms.len().min(MAX_SEMANTIC_ATOMS),
            "important_points": bundle.important_points.len().min(MAX_IMPORTANT_POINTS),
            "preferences": bundle.preferences.len().min(MAX_PREFERENCES),
            "procedures": bundle.procedures.len().min(MAX_PROCEDURES),
            "privacy_redacted": privacy_report.redacted,
            "retention_class": &privacy_report.retention_class,
            "durable_signal_count": durable_memory_signal_count(&bundle),
            "conversation_memory_bypasses_governance": false,
        }),
    })
}

fn durable_memory_signal_count(bundle: &ConversationMemoryBundle) -> usize {
    bundle.semantic_atoms.len()
        + bundle.important_points.len()
        + bundle.preferences.len()
        + bundle.procedures.len()
        + bundle.decisions.len()
}

fn conversation_turn_salience(
    importance: f32,
    durable_signal_count: usize,
    privacy_report: &crate::memory::privacy::MemoryPrivacyReport,
) -> f32 {
    let base = if durable_signal_count == 0 { 0.24 } else { 0.48 };
    let privacy_penalty = if matches!(
        &privacy_report.retention_class,
        crate::memory::privacy::MemoryRetentionClass::SensitiveReview
    ) {
        0.08
    } else {
        0.0
    };
    (base + importance * 0.42 - privacy_penalty).clamp(0.05, 1.0)
}

fn semantic_atom_kind(kind: Option<&str>, tags: &[String]) -> MemoryNodeKind {
    let normalized = kind.unwrap_or_default().trim().to_ascii_lowercase();
    if tags.iter().any(|tag| matches!(tag.as_str(), "user_preference" | "preference")) {
        return MemoryNodeKind::UserPreference;
    }
    match normalized.as_str() {
        "profile_fact" | "fact" | "claim" | "identity" | "name" => MemoryNodeKind::Claim,
        "entity" | "person" | "organization" | "project" => MemoryNodeKind::Entity,
        "procedure" | "workflow" => MemoryNodeKind::Procedure,
        "decision" => MemoryNodeKind::Decision,
        "preference" | "user_preference" => MemoryNodeKind::UserPreference,
        "concept" | "topic" => MemoryNodeKind::Concept,
        _ => MemoryNodeKind::Claim,
    }
}

fn semantic_atom_title(atom: &ConversationSemanticAtom) -> String {
    if let Some(title) = atom.title.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
        return title.to_string();
    }
    let subject = atom.subject.as_deref().map(str::trim).filter(|value| !value.is_empty()).unwrap_or("memory fact");
    let predicate = atom.predicate.as_deref().map(str::trim).filter(|value| !value.is_empty()).unwrap_or("relates_to");
    let object = atom.object.as_deref().map(str::trim).filter(|value| !value.is_empty()).unwrap_or("context");
    format!("{subject} {predicate} {object}")
}

fn semantic_atom_summary(atom: &ConversationSemanticAtom) -> String {
    if let Some(summary) = atom.summary.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
        return summary.to_string();
    }
    let subject = atom.subject.as_deref().map(str::trim).filter(|value| !value.is_empty()).unwrap_or("The conversation");
    let predicate = atom.predicate.as_deref().map(str::trim).filter(|value| !value.is_empty()).unwrap_or("indicates");
    let object = atom.object.as_deref().map(str::trim).filter(|value| !value.is_empty()).unwrap_or("a durable memory fact");
    format!("{subject} {predicate} {object}.")
}

fn render_semantic_atom_content(atom: &ConversationSemanticAtom, bundle: &ConversationMemoryBundle) -> String {
    let mut content = String::new();
    content.push_str("Semantic memory atom distilled from conversation.\n");
    if let Some(subject) = atom.subject.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
        content.push_str(&format!("Subject: {subject}\n"));
    }
    if let Some(predicate) = atom.predicate.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
        content.push_str(&format!("Predicate: {predicate}\n"));
    }
    if let Some(object) = atom.object.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
        content.push_str(&format!("Object: {object}\n"));
    }
    if let Some(evidence) = atom.evidence.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
        content.push_str(&format!("Evidence: {evidence}\n"));
    }
    content.push_str("\nOriginal user message:\n");
    content.push_str(&cap_text(&bundle.user_message, 4_000));
    content.push_str("\n\nAssistant answer:\n");
    content.push_str(&cap_text(&bundle.assistant_answer, 4_000));
    content
}

fn validate_bundle(bundle: &ConversationMemoryBundle) -> MemoryResult<()> {
    if bundle.user_message.trim().is_empty() {
        return Err(MemoryError::Validation("conversation user message cannot be empty".into()));
    }
    if bundle.assistant_answer.trim().is_empty() {
        return Err(MemoryError::Validation("conversation assistant answer cannot be empty".into()));
    }
    Ok(())
}

fn point_kind(kind: Option<&str>) -> MemoryNodeKind {
    match kind.unwrap_or_default().trim().to_ascii_lowercase().as_str() {
        "task" | "todo" | "action" => MemoryNodeKind::Task,
        "error" | "bug" | "failure" => MemoryNodeKind::Error,
        "fix" | "solution" => MemoryNodeKind::Fix,
        "concept" | "topic" => MemoryNodeKind::Concept,
        "workflow" => MemoryNodeKind::Workflow,
        "code_pattern" | "pattern" => MemoryNodeKind::CodePattern,
        "decision" => MemoryNodeKind::Decision,
        "claim" => MemoryNodeKind::Claim,
        _ => MemoryNodeKind::Concept,
    }
}

fn stable_bundle_hash(bundle: &ConversationMemoryBundle) -> MemoryResult<String> {
    let value = serde_json::to_string(bundle)?;
    Ok(short_hash(&value))
}

fn short_hash(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize()).chars().take(16).collect()
}

fn fallback_topic(user_message: &str) -> String {
    let mut value = user_message.trim().chars().take(90).collect::<String>();
    if value.is_empty() {
        value = "Conversation memory".into();
    }
    value
}

fn fallback_summary(user_message: &str, assistant_answer: &str) -> String {
    format!(
        "Conversation between the user and AstraOS. User: {} Assistant: {}",
        cap_text(user_message, 420),
        cap_text(assistant_answer, 720)
    )
}

fn render_turn_content(bundle: &ConversationMemoryBundle) -> String {
    format!(
        "User message:\n{}\n\nAssistant answer:\n{}",
        cap_text(&bundle.user_message, 8_000),
        cap_text(&bundle.assistant_answer, 12_000)
    )
}

fn render_procedure_content(procedure: &ConversationProcedure) -> String {
    let mut content = format!("Procedure: {}", procedure.title);
    if let Some(rationale) = procedure.rationale.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
        content.push_str(&format!("\nRationale: {rationale}"));
    }
    if !procedure.steps.is_empty() {
        content.push_str("\nSteps:");
        for (index, step) in procedure.steps.iter().take(24).enumerate() {
            content.push_str(&format!("\n{}. {}", index + 1, step));
        }
    }
    content
}

fn normalize_tags(mut tags: Vec<String>, defaults: &[&str]) -> Vec<String> {
    for default in defaults {
        tags.push((*default).to_string());
    }
    let mut normalized = tags
        .into_iter()
        .map(|tag| tag.trim().to_ascii_lowercase().replace(' ', "_"))
        .filter(|tag| !tag.is_empty())
        .map(|tag| cap_text(tag, 64))
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized.truncate(32);
    normalized
}

fn clamp01(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.5;
    }
    value.clamp(0.0, 1.0)
}

fn cap_text(value: impl AsRef<str>, max_chars: usize) -> String {
    let value = value.as_ref().trim();
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut capped = value.chars().take(max_chars).collect::<String>();
    capped.push('…');
    capped
}
