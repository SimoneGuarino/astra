use crate::memory::{
    errors::{MemoryError, MemoryResult},
    store::MemoryGraphStore,
    types::{
        now_ms, MemoryActivation, MemoryActivationRequest, MemoryNode, MemorySkillCandidate,
        MemorySkillCandidateExtractionReceipt, MemorySkillCandidateStatus,
        MemorySkillCandidateUpdateReceipt, MemorySkillCandidateUpdateRequest,
    },
};
use serde_json::json;

pub fn extract_skill_candidates(
    store: &MemoryGraphStore,
    limit: Option<usize>,
) -> MemoryResult<MemorySkillCandidateExtractionReceipt> {
    let max_candidates = limit.unwrap_or(80).clamp(1, 250);
    let candidates = store.extract_skill_candidates_from_memory(max_candidates)?;
    let activation = if candidates.is_empty() {
        None
    } else {
        store
            .activate(MemoryActivationRequest {
                request_id: None,
                root_query: "procedural_skill_candidate_extraction".into(),
                seed_node_ids: candidates
                    .iter()
                    .filter_map(|candidate| candidate.source_node_id.clone())
                    .take(24)
                    .collect(),
                max_depth: 1,
                max_nodes: 32,
                metadata: json!({
                    "activation_source": "procedural_skill_candidate_extraction",
                    "candidate_count": candidates.len(),
                    "metadata_only": true,
                }),
            })
            .ok()
    };

    Ok(MemorySkillCandidateExtractionReceipt {
        accepted: true,
        reason: "procedural skill candidates extracted from verified memory graph procedures/workflows; candidates are advisory and never execute without governed approval".into(),
        candidates,
        activation,
        metadata: json!({
            "source_of_truth": "sqlite_memory_graph",
            "execution_enabled": false,
            "approval_required_before_execution": true,
            "metadata_only": true,
        }),
    })
}

pub fn list_skill_candidates(
    store: &MemoryGraphStore,
    include_disabled: bool,
    limit: Option<usize>,
) -> MemoryResult<Vec<MemorySkillCandidate>> {
    store.list_skill_candidates(include_disabled, limit.unwrap_or(80).clamp(1, 250))
}

pub fn update_skill_candidate(
    store: &MemoryGraphStore,
    request: MemorySkillCandidateUpdateRequest,
) -> MemoryResult<MemorySkillCandidateUpdateReceipt> {
    if request.candidate_id.trim().is_empty() {
        return Err(MemoryError::Validation("candidate_id is required".into()));
    }

    let mut normalized = request;
    if let Some(status) = normalized.status.clone() {
        if matches!(status, MemorySkillCandidateStatus::Approved) && normalized.approved_by.is_none() {
            normalized.approved_by = Some("user".into());
        }
    }
    if normalized.metadata.is_null() {
        normalized.metadata = json!({});
    }
    store.update_skill_candidate(normalized)
}

pub fn activate_skill_candidate(
    store: &MemoryGraphStore,
    candidate_id: &str,
) -> MemoryResult<Option<MemoryActivation>> {
    let candidates = store.list_skill_candidates(true, 500)?;
    let Some(candidate) = candidates.into_iter().find(|candidate| candidate.id == candidate_id) else {
        return Err(MemoryError::Validation(format!(
            "skill candidate not found: {candidate_id}"
        )));
    };
    let Some(source_node_id) = candidate.source_node_id else {
        return Ok(None);
    };
    let activation = store.activate(MemoryActivationRequest {
        request_id: None,
        root_query: format!("skill_candidate:{}", candidate.title),
        seed_node_ids: vec![source_node_id],
        max_depth: 2,
        max_nodes: 24,
        metadata: json!({
            "activation_source": "skill_candidate_inspection",
            "candidate_id": candidate.id,
            "skill_status": candidate.status,
            "metadata_only": true,
        }),
    })?;
    Ok(Some(activation))
}

pub fn candidate_from_node(node: &MemoryNode, existing_count: usize) -> MemorySkillCandidate {
    let now = now_ms();
    MemorySkillCandidate {
        id: format!("skill_{}", uuid::Uuid::new_v4().simple()),
        title: node.title.clone(),
        summary: node.summary.clone(),
        source_node_id: Some(node.id.clone()),
        status: MemorySkillCandidateStatus::Candidate,
        confidence: node.confidence.clamp(0.0, 1.0),
        salience: node.salience.clamp(0.0, 1.0),
        trigger_hints: build_trigger_hints(node),
        required_tools: extract_required_tools(node),
        risk_level: infer_risk_level(node),
        created_at: now.saturating_add(existing_count as i64),
        updated_at: now.saturating_add(existing_count as i64),
        approved_by: None,
        approved_at: None,
        metadata: json!({
            "source_kind": node.kind.as_str(),
            "source_verification_status": node.verification_status.as_str(),
            "source_tags": node.tags.clone(),
            "governance_note": "candidate can be used as procedural context only until explicitly approved and bound to governed tools",
            "metadata_only": false,
        }),
    }
}

fn build_trigger_hints(node: &MemoryNode) -> Vec<String> {
    let mut hints = Vec::new();
    hints.push(node.title.clone());
    for tag in &node.tags {
        if hints.len() >= 8 {
            break;
        }
        if !tag.trim().is_empty() && !hints.iter().any(|existing| existing.eq_ignore_ascii_case(tag)) {
            hints.push(tag.clone());
        }
    }
    if let Some(content) = node.content.as_ref() {
        for line in content.lines().map(str::trim).filter(|line| !line.is_empty()) {
            if hints.len() >= 8 {
                break;
            }
            if line.chars().count() <= 120 {
                hints.push(line.to_string());
            }
        }
    }
    hints.into_iter().take(8).collect()
}

fn extract_required_tools(node: &MemoryNode) -> Vec<String> {
    let mut tools = Vec::new();
    let haystack = format!(
        "{}\n{}\n{}",
        node.title,
        node.summary,
        node.content.clone().unwrap_or_default()
    )
    .to_ascii_lowercase();
    for (needle, tool) in [
        ("browser", "browser"),
        ("work session", "work_session"),
        ("transcript", "work_session"),
        ("file", "filesystem"),
        ("terminal", "terminal"),
        ("screen", "screen_context"),
        ("meeting", "meeting"),
    ] {
        if haystack.contains(needle) && !tools.iter().any(|existing| existing == tool) {
            tools.push(tool.to_string());
        }
    }
    tools.truncate(8);
    tools
}

fn infer_risk_level(node: &MemoryNode) -> String {
    let haystack = format!(
        "{} {} {}",
        node.title,
        node.summary,
        node.content.clone().unwrap_or_default()
    )
    .to_ascii_lowercase();
    if haystack.contains("delete")
        || haystack.contains("write")
        || haystack.contains("terminal")
        || haystack.contains("send")
        || haystack.contains("approval")
    {
        "approval_gated".into()
    } else {
        "read_only_or_contextual".into()
    }
}
