use crate::memory::{
    errors::MemoryResult,
    store::MemoryGraphStore,
    types::{
        CreateMemoryEdgeRequest, CreateMemoryNodeRequest, MemoryRelationKind, MemoryNodeKind,
        MemoryVerificationStatus,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReflectionBundle {
    #[serde(default)]
    pub request_id: Option<String>,
    pub source: String,
    pub user_message: String,
    pub assistant_answer: String,
    #[serde(default)]
    pub memory_query: Option<String>,
    #[serde(default)]
    pub evaluated_node_ids: Vec<String>,
    #[serde(default)]
    pub used_node_ids: Vec<String>,
    #[serde(default)]
    pub ignored_relevant_node_ids: Vec<String>,
    #[serde(default)]
    pub corrected_or_contradicted_node_ids: Vec<String>,
    #[serde(default)]
    pub memory_use_quality: Option<String>,
    #[serde(default)]
    pub coverage_score: Option<f32>,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub lessons: Vec<MemoryReflectionLesson>,
    #[serde(default)]
    pub recommendations: Vec<MemoryReflectionRecommendation>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReflectionLesson {
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReflectionRecommendation {
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub action: Option<String>,
    #[serde(default)]
    pub target_node_id: Option<String>,
    #[serde(default)]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReflectionConsolidationReceipt {
    pub accepted: bool,
    pub reason: String,
    pub created_node_ids: Vec<String>,
    pub created_edge_ids: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
}

pub fn consolidate_memory_reflection_bundle(
    store: &MemoryGraphStore,
    bundle: MemoryReflectionBundle,
) -> MemoryResult<MemoryReflectionConsolidationReceipt> {
    let mut created_node_ids = Vec::new();
    let mut created_edge_ids = Vec::new();
    let coverage = clamp01(bundle.coverage_score.unwrap_or(0.5));
    let confidence = clamp01(bundle.confidence.unwrap_or(0.55));
    let should_persist = should_persist_reflection(&bundle, coverage);

    if !should_persist {
        let _ = store.append_memory_note(
            "memory_reflection_skipped",
            json!({
                "request_id": bundle.request_id,
                "source": bundle.source,
                "reason": "reflection_not_salient_enough",
                "coverage_score": coverage,
                "evaluated_node_count": bundle.evaluated_node_ids.len(),
                "used_node_count": bundle.used_node_ids.len(),
                "ignored_relevant_node_count": bundle.ignored_relevant_node_ids.len(),
                "metadata_only": true,
            }),
        );
        return Ok(MemoryReflectionConsolidationReceipt {
            accepted: true,
            reason: "reflection evaluated but not persisted because no durable lesson was found".into(),
            created_node_ids,
            created_edge_ids,
            metadata: json!({"coverage_score": coverage, "metadata_only": true}),
        });
    }

    let title = format!(
        "Memory reflection: {}",
        compact_title(&bundle.user_message, 96)
    );
    let summary = build_reflection_summary(&bundle, coverage);
    let source = reflection_source(&bundle);
    let reflection_node = store.create_node_once_by_source(CreateMemoryNodeRequest {
        kind: MemoryNodeKind::Decision,
        title,
        summary,
        content: Some(build_reflection_content(&bundle)),
        tags: vec![
            "memory_reflection".into(),
            "memory_quality".into(),
            "cognitive_loop".into(),
            bundle.source.clone(),
        ],
        source: Some(source),
        confidence,
        verification_status: MemoryVerificationStatus::LlmInferred,
        salience: if coverage < 0.55 { 0.78 } else { 0.58 },
        metadata: json!({
            "request_id": bundle.request_id,
            "memory_query": bundle.memory_query,
            "memory_use_quality": bundle.memory_use_quality,
            "coverage_score": coverage,
            "evaluated_node_ids": bundle.evaluated_node_ids,
            "used_node_ids": bundle.used_node_ids,
            "ignored_relevant_node_ids": bundle.ignored_relevant_node_ids,
            "corrected_or_contradicted_node_ids": bundle.corrected_or_contradicted_node_ids,
            "reflection_metadata": bundle.metadata,
            "metadata_only": false,
        }),
    })?;
    created_node_ids.push(reflection_node.id.clone());

    for node_id in bundle.used_node_ids.iter().take(24) {
        if let Ok(edge) = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: reflection_node.id.clone(),
            to_node_id: node_id.clone(),
            relation: MemoryRelationKind::VerifiedBy,
            weight: 0.62,
            confidence,
            metadata: json!({
                "edge_source": "memory_reflection_used_node",
                "request_id": bundle.request_id,
                "metadata_only": true,
            }),
        }) {
            created_edge_ids.push(edge.id);
        }
    }

    for node_id in bundle.ignored_relevant_node_ids.iter().take(24) {
        if let Ok(edge) = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: reflection_node.id.clone(),
            to_node_id: node_id.clone(),
            relation: MemoryRelationKind::RelatedTo,
            weight: 0.74,
            confidence,
            metadata: json!({
                "edge_source": "memory_reflection_ignored_relevant_node",
                "request_id": bundle.request_id,
                "recommended_action": "consider_in_future_answer",
                "metadata_only": true,
            }),
        }) {
            created_edge_ids.push(edge.id);
        }
    }

    for node_id in bundle.corrected_or_contradicted_node_ids.iter().take(24) {
        if let Ok(edge) = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: reflection_node.id.clone(),
            to_node_id: node_id.clone(),
            relation: MemoryRelationKind::Contradicts,
            weight: 0.78,
            confidence,
            metadata: json!({
                "edge_source": "memory_reflection_correction_or_contradiction",
                "request_id": bundle.request_id,
                "requires_user_governance": true,
                "metadata_only": true,
            }),
        }) {
            created_edge_ids.push(edge.id);
        }
    }

    for (index, lesson) in bundle.lessons.iter().take(8).enumerate() {
        let lesson_node = store.create_node_once_by_source(CreateMemoryNodeRequest {
            kind: MemoryNodeKind::Procedure,
            title: compact_title(&lesson.title, 160),
            summary: cap_text(&lesson.summary, 2048),
            content: Some(cap_text(&lesson.summary, 4096)),
            tags: normalized_tags(
                &lesson.tags,
                &["memory_reflection", "procedural_learning", "advisory_only"],
            ),
            source: Some(format!(
                "memory_reflection_lesson:{}:{}",
                bundle.request_id.clone().unwrap_or_else(|| "no_request".into()),
                index
            )),
            confidence: clamp01(lesson.confidence.unwrap_or(confidence)),
            verification_status: MemoryVerificationStatus::LlmInferred,
            salience: 0.64,
            metadata: json!({
                "request_id": bundle.request_id,
                "parent_reflection_node_id": reflection_node.id,
                "metadata_only": false,
            }),
        })?;
        created_node_ids.push(lesson_node.id.clone());
        if let Ok(edge) = store.create_edge(CreateMemoryEdgeRequest {
            from_node_id: lesson_node.id.clone(),
            to_node_id: reflection_node.id.clone(),
            relation: MemoryRelationKind::LearnedFrom,
            weight: 0.72,
            confidence: lesson_node.confidence,
            metadata: json!({
                "edge_source": "memory_reflection_lesson",
                "metadata_only": true,
            }),
        }) {
            created_edge_ids.push(edge.id);
        }
    }

    let _ = store.activate(crate::memory::types::MemoryActivationRequest {
        request_id: bundle.request_id.clone(),
        root_query: bundle
            .memory_query
            .clone()
            .unwrap_or_else(|| bundle.user_message.clone()),
        seed_node_ids: created_node_ids.iter().take(12).cloned().collect(),
        max_depth: 1,
        max_nodes: 24,
        metadata: json!({
            "activation_source": "memory_reflection_consolidation",
            "ui_hint": "reflection_loop_reached_nodes",
            "metadata_only": true,
        }),
    });

    Ok(MemoryReflectionConsolidationReceipt {
        accepted: true,
        reason: "memory reflection consolidated as advisory cognitive quality signal".into(),
        created_node_ids,
        created_edge_ids,
        metadata: json!({
            "coverage_score": coverage,
            "confidence": confidence,
            "metadata_only": true,
        }),
    })
}

fn should_persist_reflection(bundle: &MemoryReflectionBundle, coverage: f32) -> bool {
    coverage < 0.82
        || !bundle.ignored_relevant_node_ids.is_empty()
        || !bundle.corrected_or_contradicted_node_ids.is_empty()
        || !bundle.lessons.is_empty()
        || !bundle.recommendations.is_empty()
}

fn build_reflection_summary(bundle: &MemoryReflectionBundle, coverage: f32) -> String {
    let quality = bundle
        .memory_use_quality
        .clone()
        .unwrap_or_else(|| "unknown".into());
    let mut parts = vec![format!(
        "LLM reflection evaluated how Astra used Memory Graph context for this response. Memory use quality: {quality}. Coverage score: {:.2}.",
        coverage
    )];
    if !bundle.ignored_relevant_node_ids.is_empty() {
        parts.push(format!(
            "{} relevant memory nodes may have been underused.",
            bundle.ignored_relevant_node_ids.len()
        ));
    }
    if !bundle.corrected_or_contradicted_node_ids.is_empty() {
        parts.push(format!(
            "{} memory nodes may require correction or user governance.",
            bundle.corrected_or_contradicted_node_ids.len()
        ));
    }
    if !bundle.lessons.is_empty() {
        parts.push(format!("{} procedural memory lessons were proposed.", bundle.lessons.len()));
    }
    cap_text(&parts.join(" "), 2048)
}

fn build_reflection_content(bundle: &MemoryReflectionBundle) -> String {
    serde_json::to_string_pretty(bundle).unwrap_or_else(|_| "{}".into())
}

fn reflection_source(bundle: &MemoryReflectionBundle) -> String {
    format!(
        "memory_reflection:{}:{}",
        bundle.request_id.clone().unwrap_or_else(|| "no_request".into()),
        stable_hash(&format!("{}\n{}", bundle.user_message, bundle.assistant_answer))
    )
}

fn compact_title(value: &str, max_chars: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    cap_text(&compact, max_chars)
}

fn cap_text(value: &str, max_chars: usize) -> String {
    let mut out = value.chars().take(max_chars).collect::<String>();
    if value.chars().count() > max_chars {
        out.push('…');
    }
    out
}

fn normalized_tags(tags: &[String], defaults: &[&str]) -> Vec<String> {
    let mut output = defaults.iter().map(|value| value.to_string()).collect::<Vec<_>>();
    for tag in tags.iter().map(|tag| tag.trim()).filter(|tag| !tag.is_empty()) {
        let value = cap_text(tag, 48);
        if !output.iter().any(|existing| existing.eq_ignore_ascii_case(&value)) {
            output.push(value);
        }
    }
    output.truncate(24);
    output
}

fn clamp01(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn stable_hash(value: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
        .chars()
        .take(16)
        .collect()
}
