use crate::memory::{
    errors::{MemoryError, MemoryResult},
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
use std::collections::HashMap;

const MAX_FINDINGS: usize = 24;
const MAX_CLAIMS: usize = 32;
const MAX_SOURCES: usize = 24;
const MAX_PROCEDURES: usize = 12;
const MAX_RECOMMENDATIONS: usize = 16;
const MAX_TITLE_CHARS: usize = 180;
const MAX_SUMMARY_CHARS: usize = 3_000;
const MAX_CONTENT_CHARS: usize = 16_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchMemoryBundle {
    pub topic: String,
    #[serde(default)]
    pub objective: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub verification_status: Option<MemoryVerificationStatus>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub sources: Vec<ResearchSource>,
    #[serde(default)]
    pub findings: Vec<ResearchFinding>,
    #[serde(default)]
    pub claims: Vec<ResearchClaim>,
    #[serde(default)]
    pub procedures: Vec<ResearchProcedure>,
    #[serde(default)]
    pub recommendations: Vec<ResearchRecommendation>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchSource {
    pub title: String,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchFinding {
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub evidence: Vec<String>,
    #[serde(default)]
    pub source_refs: Vec<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchClaim {
    pub claim: String,
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub source_refs: Vec<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub verification_status: Option<MemoryVerificationStatus>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchProcedure {
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
pub struct ResearchRecommendation {
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub actionability: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchMemoryConsolidationReceipt {
    pub accepted: bool,
    pub reason: String,
    pub topic_node: MemoryNode,
    pub created_node_ids: Vec<String>,
    pub created_edge_ids: Vec<String>,
    #[serde(default)]
    pub activation: Option<MemoryActivation>,
    pub summary: Value,
}

pub fn consolidate_research_bundle(
    store: &MemoryGraphStore,
    bundle: ResearchMemoryBundle,
) -> MemoryResult<ResearchMemoryConsolidationReceipt> {
    validate_bundle(&bundle)?;

    let topic = cap_text(bundle.topic.trim(), MAX_TITLE_CHARS);
    let bundle_hash = stable_bundle_hash(&bundle)?;
    let confidence = clamp01(bundle.confidence.unwrap_or(0.72));
    let verification_status = bundle
        .verification_status
        .clone()
        .unwrap_or(MemoryVerificationStatus::LlmInferred);
    let base_tags = normalize_tags(bundle.tags.clone(), &["research", "deep_research"]);

    let topic_summary = bundle
        .summary
        .clone()
        .or_else(|| bundle.objective.clone())
        .unwrap_or_else(|| format!("Deep research topic consolidated by AstraOS: {topic}"));

    let topic_node = store.create_node_once_by_source(CreateMemoryNodeRequest {
        kind: MemoryNodeKind::ResearchTopic,
        title: topic.clone(),
        summary: cap_text(topic_summary, MAX_SUMMARY_CHARS),
        content: Some(cap_text(render_research_bundle_content(&bundle), MAX_CONTENT_CHARS)),
        tags: base_tags.clone(),
        source: Some(format!("research_bundle:{bundle_hash}")),
        confidence,
        verification_status: verification_status.clone(),
        salience: 0.82,
        metadata: json!({
            "ingestion_source": "deep_research_memory_consolidation",
            "schema_version": 1,
            "objective": bundle.objective.clone(),
            "query": bundle.query.clone(),
            "bundle_hash": bundle_hash,
            "source_count": bundle.sources.len(),
            "finding_count": bundle.findings.len(),
            "claim_count": bundle.claims.len(),
            "procedure_count": bundle.procedures.len(),
            "recommendation_count": bundle.recommendations.len(),
            "metadata_only": false,
            "user_visible": true,
        }),
    })?;

    let mut created_node_ids = vec![topic_node.id.clone()];
    let mut created_edge_ids = Vec::new();
    let mut source_nodes_by_ref = HashMap::<String, MemoryNode>::new();

    for (index, source) in bundle.sources.iter().take(MAX_SOURCES).enumerate() {
        if source.title.trim().is_empty() {
            continue;
        }
        let source_ref = source_ref(source, index);
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::SourceDocument,
            title: cap_text(source.title.trim(), MAX_TITLE_CHARS),
            summary: cap_text(
                source
                    .summary
                    .clone()
                    .unwrap_or_else(|| source.uri.clone().unwrap_or_else(|| "Research source".into())),
                MAX_SUMMARY_CHARS,
            ),
            content: source.uri.clone(),
            tags: normalize_tags(vec!["research_source".into()], &[]),
            source: Some(format!("research_source:{}:{}", bundle_hash, source_ref)),
            confidence: clamp01(source.confidence.unwrap_or(0.72)),
            verification_status: MemoryVerificationStatus::Unverified,
            salience: 0.58,
            metadata: json!({
                "ingestion_source": "deep_research_memory_consolidation",
                "research_topic_id": topic_node.id,
                "source_ref": source_ref,
                "uri": source.uri.clone(),
                "source_type": source.source_type.clone(),
                "metadata": source.metadata.clone(),
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: topic_node.id.clone(),
            to_node_id: node.id.clone(),
            relation: MemoryRelationKind::DerivedFrom,
            weight: 0.62,
            confidence: 0.78,
            metadata: json!({"ingestion_source": "research_topic_derived_from_source"}),
        })?;
        created_node_ids.push(node.id.clone());
        created_edge_ids.push(edge.id);
        source_nodes_by_ref.insert(source_ref, node);
    }

    for (index, finding) in bundle.findings.iter().take(MAX_FINDINGS).enumerate() {
        if finding.title.trim().is_empty() || finding.summary.trim().is_empty() {
            continue;
        }
        let node_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, finding.title));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::ResearchFinding,
            title: cap_text(finding.title.trim(), MAX_TITLE_CHARS),
            summary: cap_text(finding.summary.trim(), MAX_SUMMARY_CHARS),
            content: (!finding.evidence.is_empty()).then(|| cap_text(finding.evidence.join("\n"), MAX_CONTENT_CHARS)),
            tags: normalize_tags(finding.tags.clone(), &["research", "finding"]),
            source: Some(format!("research_finding:{}:{}", bundle_hash, node_hash)),
            confidence: clamp01(finding.confidence.unwrap_or(confidence)),
            verification_status: verification_status.clone(),
            salience: 0.76,
            metadata: json!({
                "ingestion_source": "deep_research_memory_consolidation",
                "research_topic_id": topic_node.id,
                "source_refs": finding.source_refs.clone(),
                "evidence_count": finding.evidence.len(),
                "metadata": finding.metadata.clone(),
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: topic_node.id.clone(),
            relation: MemoryRelationKind::About,
            weight: 0.84,
            confidence: 0.84,
            metadata: json!({"ingestion_source": "research_finding_about_topic"}),
        })?;
        created_node_ids.push(node.id.clone());
        created_edge_ids.push(edge.id);
        link_sources(store, &node.id, &finding.source_refs, &source_nodes_by_ref, &mut created_edge_ids);
    }

    for (index, claim) in bundle.claims.iter().take(MAX_CLAIMS).enumerate() {
        if claim.claim.trim().is_empty() {
            continue;
        }
        let node_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, claim.claim));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::Claim,
            title: cap_text(claim.claim.trim(), MAX_TITLE_CHARS),
            summary: cap_text(
                claim
                    .rationale
                    .clone()
                    .unwrap_or_else(|| claim.claim.trim().to_string()),
                MAX_SUMMARY_CHARS,
            ),
            content: claim.rationale.clone().map(|value| cap_text(value, MAX_CONTENT_CHARS)),
            tags: normalize_tags(vec!["research".into(), "claim".into()], &[]),
            source: Some(format!("research_claim:{}:{}", bundle_hash, node_hash)),
            confidence: clamp01(claim.confidence.unwrap_or(confidence)),
            verification_status: claim
                .verification_status
                .clone()
                .unwrap_or(verification_status.clone()),
            salience: 0.72,
            metadata: json!({
                "ingestion_source": "deep_research_memory_consolidation",
                "research_topic_id": topic_node.id,
                "source_refs": claim.source_refs.clone(),
                "metadata": claim.metadata.clone(),
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: topic_node.id.clone(),
            relation: MemoryRelationKind::Supports,
            weight: 0.72,
            confidence: 0.72,
            metadata: json!({"ingestion_source": "research_claim_supports_topic"}),
        })?;
        created_node_ids.push(node.id.clone());
        created_edge_ids.push(edge.id);
        link_sources(store, &node.id, &claim.source_refs, &source_nodes_by_ref, &mut created_edge_ids);
    }

    for (index, procedure) in bundle.procedures.iter().take(MAX_PROCEDURES).enumerate() {
        if procedure.title.trim().is_empty() {
            continue;
        }
        let steps = procedure.steps.iter().map(|step| step.trim()).filter(|step| !step.is_empty()).collect::<Vec<_>>();
        let summary = if steps.is_empty() {
            procedure.rationale.clone().unwrap_or_else(|| procedure.title.clone())
        } else {
            format!("{} steps: {}", steps.len(), steps.iter().take(6).cloned().collect::<Vec<_>>().join(" -> "))
        };
        let node_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, procedure.title));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::Procedure,
            title: cap_text(procedure.title.trim(), MAX_TITLE_CHARS),
            summary: cap_text(summary, MAX_SUMMARY_CHARS),
            content: Some(cap_text(render_procedure_content(procedure), MAX_CONTENT_CHARS)),
            tags: normalize_tags(vec!["research".into(), "procedure".into(), "procedural_memory".into()], &[]),
            source: Some(format!("research_procedure:{}:{}", bundle_hash, node_hash)),
            confidence: clamp01(procedure.confidence.unwrap_or(0.68)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.8,
            metadata: json!({
                "ingestion_source": "deep_research_memory_consolidation",
                "research_topic_id": topic_node.id,
                "step_count": steps.len(),
                "metadata": procedure.metadata.clone(),
                "governance_note": "procedural memory is advisory context only; actions still require governed tools and policy validation",
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: topic_node.id.clone(),
            relation: MemoryRelationKind::LearnedFrom,
            weight: 0.82,
            confidence: 0.78,
            metadata: json!({"ingestion_source": "research_procedure_learned_from_topic"}),
        })?;
        created_node_ids.push(node.id.clone());
        created_edge_ids.push(edge.id);
    }

    for (index, recommendation) in bundle.recommendations.iter().take(MAX_RECOMMENDATIONS).enumerate() {
        if recommendation.title.trim().is_empty() || recommendation.summary.trim().is_empty() {
            continue;
        }
        let node_hash = short_hash(&format!("{}:{}:{}", bundle_hash, index, recommendation.title));
        let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::Decision,
            title: cap_text(recommendation.title.trim(), MAX_TITLE_CHARS),
            summary: cap_text(recommendation.summary.trim(), MAX_SUMMARY_CHARS),
            content: recommendation.actionability.clone().map(|value| cap_text(value, MAX_CONTENT_CHARS)),
            tags: normalize_tags(vec!["research".into(), "recommendation".into()], &[]),
            source: Some(format!("research_recommendation:{}:{}", bundle_hash, node_hash)),
            confidence: clamp01(recommendation.confidence.unwrap_or(0.62)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.66,
            metadata: json!({
                "ingestion_source": "deep_research_memory_consolidation",
                "research_topic_id": topic_node.id,
                "actionability": recommendation.actionability.clone(),
                "metadata": recommendation.metadata.clone(),
                "governance_note": "recommendations are memory context only; they do not authorize autonomous action",
            }),
        })?;
        let edge = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node.id.clone(),
            to_node_id: topic_node.id.clone(),
            relation: MemoryRelationKind::RelatedTo,
            weight: 0.7,
            confidence: 0.72,
            metadata: json!({"ingestion_source": "research_recommendation_related_to_topic"}),
        })?;
        created_node_ids.push(node.id.clone());
        created_edge_ids.push(edge.id);
    }

    created_node_ids.sort();
    created_node_ids.dedup();
    created_edge_ids.sort();
    created_edge_ids.dedup();

    let activation = store
        .activate(MemoryActivationRequest {
            request_id: None,
            root_query: bundle.query.clone().unwrap_or(topic.clone()),
            seed_node_ids: vec![topic_node.id.clone()],
            max_depth: 2,
            max_nodes: 32,
            metadata: json!({
                "activation_source": "deep_research_memory_consolidation",
                "ui_hint": "electricity_reached_research_nodes",
                "topic": topic,
                "metadata_only": true,
            }),
        })
        .ok();

    Ok(ResearchMemoryConsolidationReceipt {
        accepted: true,
        reason: "research memory bundle consolidated into typed Memory Graph nodes and relations".into(),
        topic_node,
        created_node_ids: created_node_ids.clone(),
        created_edge_ids: created_edge_ids.clone(),
        activation,
        summary: json!({
            "schema_version": 1,
            "created_nodes": created_node_ids.len(),
            "created_edges": created_edge_ids.len(),
            "max_findings": MAX_FINDINGS,
            "max_claims": MAX_CLAIMS,
            "max_sources": MAX_SOURCES,
            "llm_first_rust_governed": true,
            "memory_bypasses_governance": false,
        }),
    })
}

fn validate_bundle(bundle: &ResearchMemoryBundle) -> MemoryResult<()> {
    if bundle.topic.trim().is_empty() {
        return Err(MemoryError::Validation("research topic cannot be empty".into()));
    }
    if bundle.findings.is_empty()
        && bundle.claims.is_empty()
        && bundle.procedures.is_empty()
        && bundle.recommendations.is_empty()
        && bundle.summary.as_deref().map(str::trim).unwrap_or_default().is_empty()
    {
        return Err(MemoryError::Validation(
            "research bundle must include a summary, finding, claim, procedure, or recommendation".into(),
        ));
    }
    Ok(())
}

fn link_sources(
    store: &MemoryGraphStore,
    node_id: &str,
    refs: &[String],
    sources: &HashMap<String, MemoryNode>,
    edge_ids: &mut Vec<String>,
) {
    for source_ref in refs.iter().take(8) {
        let Some(source) = sources.get(source_ref) else { continue };
        if let Ok(edge) = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: node_id.to_string(),
            to_node_id: source.id.clone(),
            relation: MemoryRelationKind::DerivedFrom,
            weight: 0.72,
            confidence: 0.74,
            metadata: json!({
                "ingestion_source": "research_item_derived_from_source",
                "source_ref": source_ref,
            }),
        }) {
            edge_ids.push(edge.id);
        }
    }
}

fn source_ref(source: &ResearchSource, index: usize) -> String {
    if let Some(explicit_ref) = source.metadata.get("source_ref").and_then(Value::as_str) {
        let trimmed = explicit_ref.trim();
        if !trimmed.is_empty() {
            return cap_text(trimmed, 96);
        }
    }
    source
        .uri
        .as_deref()
        .or(Some(source.title.as_str()))
        .map(short_hash)
        .unwrap_or_else(|| format!("source_{index}"))
}

fn stable_bundle_hash(bundle: &ResearchMemoryBundle) -> MemoryResult<String> {
    let value = serde_json::to_string(bundle)?;
    Ok(short_hash(&value))
}

fn short_hash(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
        .chars()
        .take(16)
        .collect()
}

fn render_research_bundle_content(bundle: &ResearchMemoryBundle) -> String {
    let mut parts = Vec::new();
    if let Some(objective) = bundle.objective.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        parts.push(format!("Objective:\n{objective}"));
    }
    if let Some(summary) = bundle.summary.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        parts.push(format!("Summary:\n{summary}"));
    }
    if !bundle.findings.is_empty() {
        parts.push(format!(
            "Findings:\n{}",
            bundle
                .findings
                .iter()
                .take(MAX_FINDINGS)
                .map(|finding| format!("- {}: {}", finding.title, finding.summary))
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    if !bundle.procedures.is_empty() {
        parts.push(format!(
            "Procedures:\n{}",
            bundle
                .procedures
                .iter()
                .take(MAX_PROCEDURES)
                .map(render_procedure_content)
                .collect::<Vec<_>>()
                .join("\n\n")
        ));
    }
    parts.join("\n\n")
}

fn render_procedure_content(procedure: &ResearchProcedure) -> String {
    let mut content = format!("Procedure: {}", procedure.title);
    if let Some(rationale) = procedure.rationale.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
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
    let value = value.as_ref();
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut capped = value.chars().take(max_chars).collect::<String>();
    capped.push('…');
    capped
}
