use crate::memory::{
    errors::MemoryResult,
    store::MemoryGraphStore,
    types::{now_ms, MemoryNode, MemoryNodeGovernanceUpdateRequest, MemoryNodeKind, MemoryVerificationStatus},
};
use serde_json::json;
use std::collections::HashSet;

use super::{
    autopilot::run_deep_search_knowledge_autopilot,
    types::{
        DeepSearchKnowledgeAutopilotReceipt, DeepSearchKnowledgeAutopilotRequest,
        DeepSearchKnowledgeRefreshCandidate, DeepSearchKnowledgeRefreshReceipt,
        DeepSearchKnowledgeRefreshRequest,
    },
};

pub fn run_deep_search_knowledge_refresh(
    store: &MemoryGraphStore,
    request: DeepSearchKnowledgeRefreshRequest,
) -> MemoryResult<DeepSearchKnowledgeRefreshReceipt> {
    let started_at = now_ms();
    if !request.enabled {
        return Ok(DeepSearchKnowledgeRefreshReceipt {
            accepted: false,
            reason: "deep-search knowledge refresh is disabled by request".into(),
            started_at,
            completed_at: now_ms(),
            dry_run: request.dry_run,
            candidates_scanned: 0,
            stale_candidates: 0,
            tagged_for_refresh: 0,
            refresh_runs: 0,
            sources_accepted: 0,
            claims_promoted: 0,
            candidate_claims: 0,
            candidates: Vec::new(),
            autopilot: None,
            warnings: Vec::new(),
            recommendations: vec!["Enable knowledge refresh before scheduling stale-memory repair.".into()],
            metadata: json!({"schema_version": 1, "source": "deep_search_knowledge_refresh", "enabled": false}),
        });
    }

    let snapshot = store.snapshot(request.snapshot_limit.clamp(25, 500))?;
    let mut candidates = mine_stale_candidates(&snapshot.nodes, &request, started_at);
    let candidates_scanned = snapshot.nodes.len();
    candidates.sort_by(|left, right| right.priority.partial_cmp(&left.priority).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(request.max_candidates.clamp(1, 80));

    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    if candidates.is_empty() {
        recommendations.push("No stale or temporally fragile memory candidates were found in the sampled graph.".into());
    }

    let mut tagged_for_refresh = 0usize;
    if !request.dry_run && request.tag_candidates_for_refresh {
        for candidate in candidates.iter().take(request.max_tags.clamp(0, 80)) {
            let update = MemoryNodeGovernanceUpdateRequest {
                node_id: candidate.node.id.clone(),
                verification_status: None,
                confidence: Some((candidate.node.confidence * 0.98).clamp(0.05, 1.0)),
                salience: Some((candidate.node.salience + 0.03).clamp(0.0, 1.0)),
                add_tags: vec!["stale_candidate".into(), "refresh_requested".into()],
                remove_tags: Vec::new(),
                reason: Some("knowledge_refresh_marked_candidate_for_bounded_refresh".into()),
                actor: Some("astra_knowledge_refresh".into()),
                metadata: json!({
                    "schema_version": 1,
                    "source": "deep_search_knowledge_refresh",
                    "candidate_reason": candidate.reason,
                    "priority": candidate.priority,
                    "age_days": candidate.age_days,
                    "temporal": candidate.temporal,
                    "metadata_only": true,
                }),
            };
            match store.update_node_governance(update) {
                Ok(_) => tagged_for_refresh += 1,
                Err(error) => warnings.push(format!("refresh_candidate_tag_failed:{}:{error}", candidate.node.id)),
            }
        }
    }

    let mut autopilot: Option<DeepSearchKnowledgeAutopilotReceipt> = None;
    if !request.dry_run && request.run_refresh_research && !candidates.is_empty() {
        let seed_topics = build_seed_topics(&candidates, request.max_refresh_topics.clamp(1, 24));
        let autopilot_request = DeepSearchKnowledgeAutopilotRequest {
            enabled: true,
            dry_run: false,
            max_topics: seed_topics.len().clamp(1, request.max_refresh_topics.clamp(1, 24)),
            max_runs: request.max_refresh_runs.clamp(1, 8),
            max_sources_per_topic: request.max_sources_per_topic.clamp(2, 24),
            min_topic_priority: 0.0,
            include_low_confidence_claims: true,
            include_user_context_topics: false,
            include_topic_mining: false,
            seed_topics,
            blocked_topics: request.blocked_topics.clone(),
            search_providers: request.search_providers.clone(),
            reason: Some("deep_search_knowledge_refresh".into()),
            deep_search_defaults: request.deep_search_defaults.clone(),
            metadata: json!({
                "schema_version": 1,
                "source": "deep_search_knowledge_refresh",
                "refresh_cycle": true,
                "metadata_only": true,
            }),
        };
        match run_deep_search_knowledge_autopilot(store, autopilot_request) {
            Ok(receipt) => autopilot = Some(receipt),
            Err(error) => warnings.push(format!("knowledge_refresh_autopilot_failed:{error}")),
        }
    }

    let sources_accepted = autopilot.as_ref().map(|receipt| receipt.sources_accepted).unwrap_or(0);
    let claims_promoted = autopilot.as_ref().map(|receipt| receipt.claims_promoted).unwrap_or(0);
    let candidate_claims = autopilot.as_ref().map(|receipt| receipt.candidate_claims).unwrap_or(0);
    let refresh_runs = autopilot.as_ref().map(|receipt| receipt.runs_executed).unwrap_or(0);

    if request.dry_run {
        recommendations.push("Dry run only: stale candidates were identified but no memory tag or web refresh was executed.".into());
    } else if request.run_refresh_research && refresh_runs == 0 && !candidates.is_empty() {
        recommendations.push("Refresh candidates were found, but no refresh run completed. Check provider/network policy or lower blocked topic constraints.".into());
    }
    if claims_promoted == 0 && refresh_runs > 0 {
        recommendations.push("Refresh completed without automatic claim promotion. This is safe; it means refreshed evidence did not pass promotion thresholds yet.".into());
    }

    let completed_at = now_ms();
    let receipt = DeepSearchKnowledgeRefreshReceipt {
        accepted: true,
        reason: "knowledge refresh completed bounded stale-memory detection and optional deep-search refresh".into(),
        started_at,
        completed_at,
        dry_run: request.dry_run,
        candidates_scanned,
        stale_candidates: candidates.len(),
        tagged_for_refresh,
        refresh_runs,
        sources_accepted,
        claims_promoted,
        candidate_claims,
        candidates,
        autopilot,
        warnings: dedup(warnings, 16),
        recommendations: dedup(recommendations, 12),
        metadata: json!({
            "schema_version": 1,
            "phase": "v0.7.1_knowledge_refresh_stale_memory_detection",
            "bounded": true,
            "dry_run": request.dry_run,
            "run_refresh_research": request.run_refresh_research,
            "tag_candidates_for_refresh": request.tag_candidates_for_refresh,
            "metadata_only": true,
        }),
    };

    let _ = store.append_memory_note("deep_search_knowledge_refresh_ran", json!({
        "schema_version": 1,
        "phase": "v0.7.1_knowledge_refresh_stale_memory_detection",
        "dry_run": receipt.dry_run,
        "candidates_scanned": receipt.candidates_scanned,
        "stale_candidates": receipt.stale_candidates,
        "tagged_for_refresh": receipt.tagged_for_refresh,
        "refresh_runs": receipt.refresh_runs,
        "sources_accepted": receipt.sources_accepted,
        "claims_promoted": receipt.claims_promoted,
        "candidate_claims": receipt.candidate_claims,
        "metadata_only": true,
    }));

    Ok(receipt)
}

fn mine_stale_candidates(
    nodes: &[MemoryNode],
    request: &DeepSearchKnowledgeRefreshRequest,
    now: i64,
) -> Vec<DeepSearchKnowledgeRefreshCandidate> {
    let stale_after_ms = days_to_ms(request.stale_after_days.max(1));
    let temporal_stale_after_ms = days_to_ms(request.temporal_stale_after_days.max(1));
    let blocked = request
        .blocked_topics
        .iter()
        .map(|topic| topic.trim().to_ascii_lowercase())
        .filter(|topic| !topic.is_empty())
        .collect::<Vec<_>>();

    let mut candidates = Vec::new();
    for node in nodes {
        if !is_refresh_eligible(node) {
            continue;
        }
        let text = searchable_text(node);
        let lowered = text.to_ascii_lowercase();
        if blocked.iter().any(|blocked| lowered.contains(blocked)) {
            continue;
        }
        let age_ms = now.saturating_sub(node.updated_at.max(node.created_at));
        let temporal = is_temporal_or_fast_changing(&lowered, node);
        let low_confidence = matches!(node.verification_status, MemoryVerificationStatus::Unverified)
            || node.confidence < request.low_confidence_threshold.clamp(0.05, 0.95);
        let stale = age_ms >= stale_after_ms || (temporal && age_ms >= temporal_stale_after_ms);
        if !stale && !(request.include_low_confidence_candidates && low_confidence) {
            continue;
        }

        let mut reasons = Vec::new();
        if stale { reasons.push("age_exceeds_refresh_window"); }
        if temporal { reasons.push("temporal_or_fast_changing_topic"); }
        if low_confidence { reasons.push("low_confidence_or_unverified"); }
        if node.tags.iter().any(|tag| tag.eq_ignore_ascii_case("refresh_requested")) { reasons.push("already_marked_refresh_requested"); }

        let age_days = (age_ms as f32 / 86_400_000.0).max(0.0);
        let mut priority = (age_days / request.stale_after_days.max(1) as f32).min(1.0) * 0.42;
        if temporal { priority += 0.24; }
        if low_confidence { priority += 0.18; }
        if matches!(node.kind, MemoryNodeKind::Claim | MemoryNodeKind::ResearchFinding) { priority += 0.10; }
        if has_external_source(node) { priority += 0.06; }
        priority = priority.clamp(0.05, 1.0);

        candidates.push(DeepSearchKnowledgeRefreshCandidate {
            node: node.clone(),
            topic: candidate_topic(node),
            reason: reasons.join(","),
            priority,
            age_days,
            temporal,
            low_confidence,
            metadata: json!({
                "schema_version": 1,
                "source": "deep_search_knowledge_refresh",
                "reasons": reasons,
                "metadata_only": true,
            }),
        });
    }
    candidates
}

fn is_refresh_eligible(node: &MemoryNode) -> bool {
    if matches!(node.verification_status, MemoryVerificationStatus::Deprecated | MemoryVerificationStatus::Contradicted) {
        return false;
    }
    matches!(
        node.kind,
        MemoryNodeKind::Claim
            | MemoryNodeKind::ResearchFinding
            | MemoryNodeKind::ResearchTopic
            | MemoryNodeKind::SourceDocument
            | MemoryNodeKind::Concept
            | MemoryNodeKind::Procedure
            | MemoryNodeKind::Decision
            | MemoryNodeKind::CodePattern
            | MemoryNodeKind::Workflow
    )
}

fn is_temporal_or_fast_changing(text: &str, node: &MemoryNode) -> bool {
    const KEYWORDS: &[&str] = &[
        "latest", "current", "today", "recent", "now", "version", "release", "schedule",
        "price", "pricing", "law", "regulation", "policy", "api", "dependency", "library",
        "security", "vulnerability", "cve", "benchmark", "leaderboard", "ceo", "president",
        "attuale", "recente", "oggi", "prezzo", "legge", "normativa", "rilascio", "versione",
        "sicurezza", "vulnerabilità", "aggiornamento", "benchmark",
    ];
    KEYWORDS.iter().any(|keyword| text.contains(keyword))
        || node.tags.iter().any(|tag| {
            let lowered = tag.to_ascii_lowercase();
            lowered.contains("temporal") || lowered.contains("refresh") || lowered.contains("current")
        })
}

fn has_external_source(node: &MemoryNode) -> bool {
    node.source
        .as_deref()
        .map(|source| source.starts_with("http://") || source.starts_with("https://") || source.contains("doi.org") || source.contains("arxiv"))
        .unwrap_or(false)
}

fn searchable_text(node: &MemoryNode) -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}",
        node.title,
        node.summary,
        node.content.as_deref().unwrap_or_default(),
        node.tags.join(" "),
        node.source.as_deref().unwrap_or_default(),
    )
}

fn candidate_topic(node: &MemoryNode) -> String {
    let title = node.title.trim();
    if !title.is_empty() {
        return cap_topic(title);
    }
    let summary = node.summary.trim();
    if !summary.is_empty() {
        return cap_topic(summary);
    }
    node.id.clone()
}

fn cap_topic(value: &str) -> String {
    let trimmed = value.split_whitespace().take(18).collect::<Vec<_>>().join(" ");
    trimmed.chars().take(180).collect()
}

fn build_seed_topics(candidates: &[DeepSearchKnowledgeRefreshCandidate], limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut topics = Vec::new();
    for candidate in candidates.iter().take(limit.saturating_mul(2).max(1)) {
        let topic = candidate.topic.trim();
        if topic.is_empty() { continue; }
        let key = topic.to_ascii_lowercase();
        if seen.insert(key) {
            topics.push(topic.to_string());
        }
        if topics.len() >= limit { break; }
    }
    topics
}

fn days_to_ms(days: u64) -> i64 {
    days.saturating_mul(86_400_000).min(i64::MAX as u64) as i64
}

fn dedup(values: Vec<String>, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let key = value.to_ascii_lowercase();
        if seen.insert(key) {
            out.push(value);
        }
        if out.len() >= limit { break; }
    }
    out
}
