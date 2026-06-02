use crate::memory::{
    errors::MemoryResult,
    store::MemoryGraphStore,
    types::{
        now_ms, CreateMemoryEdgeRequest, CreateMemoryNodeRequest, LegacyCanonicalMemoryCleanupItem,
        LegacyCanonicalMemoryCleanupReceipt, LegacyCanonicalMemoryCleanupRequest, MemoryMergeNodesRequest,
        MemoryNode, MemoryNodeGovernanceUpdateRequest, MemoryNodeKind, MemoryRelationKind,
        MemoryVerificationStatus,
    },
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone)]
struct CleanupCandidate {
    node: MemoryNode,
    canonical_source: String,
    title: String,
    summary: String,
    object: Option<String>,
    confidence: f32,
    reason: String,
}

pub fn run_legacy_canonical_memory_cleanup(
    store: &MemoryGraphStore,
    request: LegacyCanonicalMemoryCleanupRequest,
) -> MemoryResult<LegacyCanonicalMemoryCleanupReceipt> {
    let started_at = now_ms();
    let scan_limit = request.max_scan_nodes.unwrap_or(1200).clamp(50, 5000);
    let group_limit = request.max_groups.unwrap_or(24).clamp(1, 100);
    let mark_aliases_deprecated = request.mark_aliases_deprecated.unwrap_or(true);
    let reason = request
        .reason
        .clone()
        .unwrap_or_else(|| "memory_autopilot_legacy_canonical_cleanup".into());
    let nodes = store.list_nodes_for_maintenance(scan_limit, true)?;
    let scanned_nodes = nodes.len();

    let mut grouped: BTreeMap<String, Vec<CleanupCandidate>> = BTreeMap::new();
    for node in nodes {
        if node.verification_status == MemoryVerificationStatus::Deprecated
            || node.verification_status == MemoryVerificationStatus::Contradicted
        {
            continue;
        }
        if let Some(candidate) = detect_legacy_profile_candidate(node) {
            grouped.entry(candidate.canonical_source.clone()).or_default().push(candidate);
        }
    }

    let mut items = Vec::new();
    let mut canonical_nodes_created = 0usize;
    let mut canonical_nodes_existing = 0usize;
    let mut alias_nodes_merged = 0usize;
    let mut alias_nodes_deprecated = 0usize;
    let mut groups_processed = 0usize;
    let mut skipped_groups = 0usize;
    let mut warnings = Vec::new();

    for (canonical_source, mut candidates) in grouped.into_iter().take(group_limit) {
        candidates.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.node.updated_at.cmp(&left.node.updated_at))
        });
        candidates.dedup_by(|left, right| left.node.id == right.node.id);
        if candidates.is_empty() {
            continue;
        }

        let existing = store.find_node_by_source(&canonical_source)?;
        let canonical_existed = existing.is_some();
        let target = if let Some(node) = existing {
            node
        } else {
            if request.dry_run.unwrap_or(false) {
                skipped_groups += 1;
                items.push(LegacyCanonicalMemoryCleanupItem {
                    canonical_source: canonical_source.clone(),
                    target_node_id: None,
                    created_canonical_node: false,
                    merged_node_ids: Vec::new(),
                    deprecated_node_ids: Vec::new(),
                    linked_node_ids: candidates.iter().map(|candidate| candidate.node.id.clone()).collect(),
                    reason: "dry_run_would_create_canonical_node".into(),
                    metadata: json!({
                        "candidate_count": candidates.len(),
                        "sample_titles": candidates.iter().take(5).map(|candidate| candidate.node.title.clone()).collect::<Vec<_>>(),
                        "metadata_only": true,
                    }),
                });
                continue;
            }
            let best = &candidates[0];
            let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
                kind: MemoryNodeKind::Claim,
                title: best.title.clone(),
                summary: best.summary.clone(),
                content: Some(render_canonical_content(&candidates)),
                tags: vec![
                    "canonical_memory".into(),
                    "schema_first_memory".into(),
                    "semantic_fact".into(),
                    "user_profile".into(),
                    "profile_fact".into(),
                    "durable_fact".into(),
                    "long_term_memory".into(),
                    "auto_canonical_cleanup".into(),
                ],
                source: Some(canonical_source.clone()),
                confidence: best.confidence.max(0.82).min(0.95),
                verification_status: MemoryVerificationStatus::LlmInferred,
                salience: 0.97,
                metadata: json!({
                    "ingestion_source": "legacy_canonical_memory_cleanup",
                    "canonical_source": canonical_source.clone(),
                    "canonical_profile_slot": canonical_source.starts_with("astra://memory/profile/user/"),
                    "object": best.object.clone(),
                    "reason": reason.clone(),
                    "metadata_only": true,
                }),
            })?;
            canonical_nodes_created += 1;
            node
        };
        if canonical_existed {
            canonical_nodes_existing += 1;
        }

        let mut merged = Vec::new();
        let mut deprecated = Vec::new();
        let mut linked = Vec::new();
        let mut same_kind_sources = Vec::new();
        let mut different_kind_sources = Vec::new();
        let mut seen = HashSet::new();
        for candidate in candidates.iter() {
            if candidate.node.id == target.id || !seen.insert(candidate.node.id.clone()) {
                continue;
            }
            if candidate.node.source.as_deref() == Some(canonical_source.as_str()) {
                continue;
            }
            if candidate.node.kind == target.kind {
                same_kind_sources.push(candidate.node.id.clone());
            } else {
                different_kind_sources.push(candidate.node.id.clone());
            }
        }

        if !request.dry_run.unwrap_or(false) && !same_kind_sources.is_empty() {
            match store.merge_nodes(MemoryMergeNodesRequest {
                target_node_id: target.id.clone(),
                source_node_ids: same_kind_sources.clone(),
                mark_sources_deprecated: mark_aliases_deprecated,
                actor: Some("memory_autopilot".into()),
                reason: Some(reason.clone()),
                metadata: json!({
                    "source": "legacy_canonical_memory_cleanup",
                    "canonical_source": canonical_source.clone(),
                    "automatic": true,
                    "metadata_only": true,
                }),
            }) {
                Ok(receipt) => {
                    merged.extend(receipt.merged_node_ids.clone());
                    if mark_aliases_deprecated {
                        deprecated.extend(receipt.merged_node_ids);
                    }
                }
                Err(error) => {
                    warnings.push(format!("merge_failed:{canonical_source}:{error}"));
                    different_kind_sources.extend(same_kind_sources);
                }
            }
        } else if request.dry_run.unwrap_or(false) {
            linked.extend(same_kind_sources.clone());
        }

        for source_id in different_kind_sources {
            if request.dry_run.unwrap_or(false) {
                linked.push(source_id);
                continue;
            }
            match store.create_edge(CreateMemoryEdgeRequest {
                from_node_id: source_id.clone(),
                to_node_id: target.id.clone(),
                relation: MemoryRelationKind::SameTopicAs,
                weight: 0.91,
                confidence: 0.84,
                metadata: json!({
                    "semantic_relation": "legacy_alias_of_canonical_memory",
                    "source": "legacy_canonical_memory_cleanup",
                    "canonical_source": canonical_source.clone(),
                    "soft_link": true,
                    "metadata_only": true,
                }),
            }) {
                Ok(_) => linked.push(source_id.clone()),
                Err(error) => warnings.push(format!("link_failed:{source_id}:{error}")),
            }
            if mark_aliases_deprecated {
                match store.update_node_governance(MemoryNodeGovernanceUpdateRequest {
                    node_id: source_id.clone(),
                    verification_status: Some(MemoryVerificationStatus::Deprecated),
                    confidence: None,
                    salience: Some(0.18),
                    add_tags: vec!["legacy_alias".into(), "canonicalized".into(), "auto_deprecated_alias".into()],
                    remove_tags: Vec::new(),
                    reason: Some(reason.clone()),
                    actor: Some("memory_autopilot".into()),
                    metadata: json!({
                        "source": "legacy_canonical_memory_cleanup",
                        "canonical_source": canonical_source.clone(),
                        "target_node_id": target.id.clone(),
                        "metadata_only": true,
                    }),
                }) {
                    Ok(_) => deprecated.push(source_id.clone()),
                    Err(error) => warnings.push(format!("deprecate_failed:{source_id}:{error}")),
                }
            }
        }

        alias_nodes_merged += merged.len();
        alias_nodes_deprecated += deprecated.len();
        groups_processed += 1;
        items.push(LegacyCanonicalMemoryCleanupItem {
            canonical_source: canonical_source.clone(),
            target_node_id: Some(target.id.clone()),
            created_canonical_node: !canonical_existed,
            merged_node_ids: merged,
            deprecated_node_ids: deprecated,
            linked_node_ids: linked,
            reason: "canonical profile/fact aliases linked to schema-first memory".into(),
            metadata: json!({
                "target_title": target.title,
                "target_summary": target.summary,
                "dry_run": request.dry_run.unwrap_or(false),
                "metadata_only": true,
            }),
        });
    }

    Ok(LegacyCanonicalMemoryCleanupReceipt {
        accepted: true,
        reason: if request.dry_run.unwrap_or(false) {
            "legacy canonical memory cleanup dry run completed".into()
        } else {
            "legacy canonical memory cleanup completed".into()
        },
        started_at,
        completed_at: now_ms(),
        scanned_nodes,
        groups_processed,
        skipped_groups,
        canonical_nodes_created,
        canonical_nodes_existing,
        alias_nodes_merged,
        alias_nodes_deprecated,
        items,
        warnings,
        metadata: json!({
            "source": "legacy_canonical_memory_cleanup",
            "automatic": true,
            "trigger": "memory_autopilot",
            "mark_aliases_deprecated": mark_aliases_deprecated,
            "dry_run": request.dry_run.unwrap_or(false),
            "metadata_only": true,
        }),
    })
}

fn detect_legacy_profile_candidate(node: MemoryNode) -> Option<CleanupCandidate> {
    let text = format!(
        "{}\n{}\n{}\n{}",
        node.title,
        node.summary,
        node.content.as_deref().unwrap_or_default(),
        node.source.as_deref().unwrap_or_default()
    );
    let normalized = normalize_text(&text);
    let tags = node.tags.iter().map(|tag| tag.to_ascii_lowercase()).collect::<Vec<_>>();
    let source = node.source.clone().unwrap_or_default();

    if source.starts_with("astra://memory/profile/user/") {
        return None;
    }

    if let Some((predicate, object, reason)) = metadata_profile_fact(&node.metadata) {
        let canonical_source = format!("astra://memory/profile/user/{predicate}");
        return Some(candidate_from_parts(node, canonical_source, predicate, object, reason));
    }

    let profile_signal = tags.iter().any(|tag| matches!(tag.as_str(),
        "user_profile" | "profile_fact" | "identity" | "name" | "canonical_memory" | "schema_first_memory" | "durable_fact" | "long_term_memory"
    ));
    if profile_signal {
        if let Some(name) = extract_declared_name(&text) {
            return Some(candidate_from_parts(
                node,
                "astra://memory/profile/user/has_name".into(),
                "has_name".into(),
                name,
                "tagged_profile_name_alias".into(),
            ));
        }
    }

    if normalized.contains("subject user") && normalized.contains("has_name") {
        if let Some(name) = extract_declared_name(&text) {
            return Some(candidate_from_parts(
                node,
                "astra://memory/profile/user/has_name".into(),
                "has_name".into(),
                name,
                "schema_text_name_alias".into(),
            ));
        }
    }

    if looks_like_user_name_memory(&normalized) {
        if let Some(name) = extract_declared_name(&text) {
            return Some(candidate_from_parts(
                node,
                "astra://memory/profile/user/has_name".into(),
                "has_name".into(),
                name,
                "legacy_natural_language_name_alias".into(),
            ));
        }
    }

    None
}

fn metadata_profile_fact(metadata: &Value) -> Option<(String, String, String)> {
    let canonical = metadata.get("canonical_memory").unwrap_or(metadata);
    let subject = canonical
        .get("canonical_subject")
        .or_else(|| metadata.get("subject"))
        .and_then(Value::as_str)
        .map(normalize_text)?;
    let predicate = canonical
        .get("canonical_predicate")
        .or_else(|| metadata.get("predicate"))
        .and_then(Value::as_str)
        .map(normalize_predicate)?;
    if subject != "user" || !is_profile_predicate(&predicate) {
        return None;
    }
    let object = metadata
        .get("object")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| metadata.get("metadata").and_then(|inner| inner.get("object")).and_then(Value::as_str).map(ToOwned::to_owned))
        .unwrap_or_else(|| "profile fact".into());
    Some((predicate, object, "metadata_schema_profile_fact".into()))
}

fn candidate_from_parts(
    node: MemoryNode,
    canonical_source: String,
    predicate: String,
    object: String,
    reason: String,
) -> CleanupCandidate {
    let (title, summary) = match predicate.as_str() {
        "has_name" => (
            "User self-introduction: name".into(),
            format!("The user introduced themselves as {}.", object.trim()),
        ),
        "prefers" => (
            "User profile: preference".into(),
            format!("The user stated a durable preference: {}.", object.trim()),
        ),
        "works_on" => (
            "User profile: project/work context".into(),
            format!("The user stated they are working on {}.", object.trim()),
        ),
        "works_as" => (
            "User profile: role".into(),
            format!("The user stated a durable role/work identity: {}.", object.trim()),
        ),
        "uses" => (
            "User profile: tools or stack".into(),
            format!("The user stated they use {}.", object.trim()),
        ),
        "wants" => (
            "User profile: goal".into(),
            format!("The user stated a durable goal: {}.", object.trim()),
        ),
        "requires" => (
            "User profile: constraint".into(),
            format!("The user stated a durable requirement or constraint: {}.", object.trim()),
        ),
        _ => (
            format!("User profile: {}", predicate.replace('_', " ")),
            format!("Durable user profile fact: user {} {}.", predicate.replace('_', " "), object.trim()),
        ),
    };
    let confidence = node.confidence.max(0.74).min(0.94);
    CleanupCandidate { node, canonical_source, title, summary, object: Some(object), confidence, reason }
}

fn render_canonical_content(candidates: &[CleanupCandidate]) -> String {
    let mut lines = vec!["Canonicalized durable user memory.".to_string(), "".to_string(), "Evidence aliases:".to_string()];
    for candidate in candidates.iter().take(12) {
        lines.push(format!("- {} — {}", candidate.node.title.trim(), candidate.reason));
    }
    lines.join("\n")
}

fn normalize_predicate(value: &str) -> String {
    let normalized = normalize_text(value).replace('-', "_").replace(' ', "_");
    match normalized.as_str() {
        "name" | "is_name" | "called" | "is_called" | "mi_chiamo" | "si_chiama" | "preferred_name" => "has_name".into(),
        "likes" | "like" | "preference" | "has_preference" | "preferisce" => "prefers".into(),
        "working_on" | "sta_lavorando_su" | "lavora_su" => "works_on".into(),
        "role" | "job" | "is_role" | "ruolo" => "works_as".into(),
        "usa" | "uses_tool" | "uses_stack" => "uses".into(),
        "vuole" | "goal" | "objective" => "wants".into(),
        "constraint" | "vincolo" | "needs" => "requires".into(),
        _ => normalized,
    }
}

fn is_profile_predicate(predicate: &str) -> bool {
    matches!(predicate, "has_name" | "prefers" | "works_on" | "works_as" | "uses" | "wants" | "requires")
}

fn looks_like_user_name_memory(normalized: &str) -> bool {
    let has_user = normalized.contains("user")
        || normalized.contains("utente")
        || normalized.contains("io sono")
        || normalized.contains("mi chiamo")
        || normalized.contains("mio nome")
        || normalized.contains("my name")
        || normalized.contains("i am");
    let has_name_signal = normalized.contains("has_name")
        || normalized.contains("name")
        || normalized.contains("nome")
        || normalized.contains("chiamo")
        || normalized.contains("introduced themselves")
        || normalized.contains("si e presentato");
    has_user && has_name_signal
}

fn extract_declared_name(text: &str) -> Option<String> {
    let compact = text.replace('\n', " ");
    let patterns = [
        "mi chiamo ",
        "il mio nome è ",
        "il mio nome e ",
        "sono ",
        "i am ",
        "my name is ",
        "introduced themselves as ",
        "si chiama ",
        "has_name ",
        "object: ",
    ];
    let lower = compact.to_ascii_lowercase();
    for pattern in patterns {
        if let Some(index) = lower.find(pattern) {
            let start = index + pattern.len();
            let original = compact.get(start..).unwrap_or_default();
            if let Some(value) = clean_name_candidate(original) {
                return Some(value);
            }
        }
    }
    None
}

fn clean_name_candidate(value: &str) -> Option<String> {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_alphabetic() || ch == '\'' || ch == '-' || ch == ' ' {
            out.push(ch);
        } else {
            break;
        }
        if out.len() > 80 {
            break;
        }
    }
    let cleaned = out
        .trim()
        .trim_matches(|ch: char| ch == '.' || ch == ',' || ch == ':' || ch == ';' || ch == '"' || ch == '\'')
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ");
    if cleaned.len() < 2 || cleaned.eq_ignore_ascii_case("the user") || cleaned.eq_ignore_ascii_case("utente") {
        None
    } else {
        Some(cleaned)
    }
}

fn normalize_text(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .replace('è', "e")
        .replace('é', "e")
        .replace('à', "a")
        .replace('ò', "o")
        .replace('ù', "u")
        .replace('ì', "i")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
