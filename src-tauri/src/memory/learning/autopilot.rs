use crate::memory::{
    deep_search::{run_deep_search_foundation, DeepSearchRequest},
    errors::MemoryResult,
    store::MemoryGraphStore,
    types::now_ms,
};
use serde_json::json;

use super::{topic_mining::build_learning_agenda, types::{DeepSearchKnowledgeAutopilotReceipt, DeepSearchKnowledgeAutopilotRequest, DeepSearchLearningAgendaItem, DeepSearchLearningRunReceipt}};

pub fn run_deep_search_knowledge_autopilot(
    store: &MemoryGraphStore,
    request: DeepSearchKnowledgeAutopilotRequest,
) -> MemoryResult<DeepSearchKnowledgeAutopilotReceipt> {
    let started_at = now_ms();
    if !request.enabled {
        return Ok(DeepSearchKnowledgeAutopilotReceipt {
            accepted: false,
            reason: "deep-search knowledge autopilot is disabled by request".into(),
            started_at,
            completed_at: now_ms(),
            dry_run: request.dry_run,
            agenda_items: 0,
            runs_executed: 0,
            sources_accepted: 0,
            claims_extracted: 0,
            findings_extracted: 0,
            claims_promoted: 0,
            candidate_claims: 0,
            agenda: Vec::new(),
            runs: Vec::new(),
            warnings: Vec::new(),
            recommendations: vec!["Enable the Deep Search Knowledge Autopilot toggle before scheduling learning cycles.".into()],
            metadata: json!({"schema_version": 1, "source": "deep_search_knowledge_autopilot", "enabled": false}),
        });
    }

    let snapshot = store.snapshot(240)?;
    let agenda = build_learning_agenda(&snapshot, &request);
    let max_runs = request.max_runs.clamp(0, request.max_topics.clamp(1, 24));
    let mut runs = Vec::new();
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();

    if agenda.is_empty() {
        recommendations.push("No sufficiently strong learning agenda item was mined from current memory. Add seed_topics or create more memory through normal use.".into());
    }

    if !request.dry_run {
        for item in agenda.iter().take(max_runs) {
            let deep_request = build_deep_search_request(&request, item);
            match run_deep_search_foundation(store, deep_request) {
                Ok(receipt) => {
                    let promoted = receipt.promotion.as_ref().map(|report| report.promoted_claims).unwrap_or(0);
                    let candidates = receipt.promotion.as_ref().map(|report| report.candidate_claims).unwrap_or(0);
                    runs.push(DeepSearchLearningRunReceipt {
                        agenda_item: item.clone(),
                        accepted: receipt.accepted,
                        reason: receipt.reason.clone(),
                        accepted_sources: receipt.accepted_sources.len(),
                        extracted_claims: receipt.extracted_claims,
                        extracted_findings: receipt.extracted_findings,
                        promoted_claims: promoted,
                        candidate_claims: candidates,
                        stop_reason: receipt.saturation.stop_reason.as_ref().map(|reason| format!("{reason:?}")),
                        warnings: receipt.warnings.clone(),
                        receipt: Some(receipt),
                    });
                }
                Err(error) => {
                    warnings.push(format!("learning_autopilot_deep_search_failed:{}:{error}", item.topic));
                    runs.push(DeepSearchLearningRunReceipt {
                        agenda_item: item.clone(),
                        accepted: false,
                        reason: format!("deep-search failed: {error}"),
                        accepted_sources: 0,
                        extracted_claims: 0,
                        extracted_findings: 0,
                        promoted_claims: 0,
                        candidate_claims: 0,
                        stop_reason: None,
                        warnings: vec![error.to_string()],
                        receipt: None,
                    });
                }
            }
        }
    }

    let sources_accepted = runs.iter().map(|run| run.accepted_sources).sum();
    let claims_extracted = runs.iter().map(|run| run.extracted_claims).sum();
    let findings_extracted = runs.iter().map(|run| run.extracted_findings).sum();
    let claims_promoted = runs.iter().map(|run| run.promoted_claims).sum();
    let candidate_claims = runs.iter().map(|run| run.candidate_claims).sum();

    if claims_promoted == 0 && !request.dry_run && !runs.is_empty() {
        recommendations.push("Autopilot ran, but no claim reached the automatic promotion threshold. Consider lowering topic scope or increasing max_sources_per_topic, not disabling verification.".into());
    }
    if request.dry_run {
        recommendations.push("Dry run only: agenda was produced but no web/document research was executed.".into());
    }

    let completed_at = now_ms();
    let receipt = DeepSearchKnowledgeAutopilotReceipt {
        accepted: !agenda.is_empty(),
        reason: if request.dry_run {
            "deep-search knowledge autopilot produced a bounded learning agenda".into()
        } else {
            "deep-search knowledge autopilot completed a bounded continuous-learning cycle".into()
        },
        started_at,
        completed_at,
        dry_run: request.dry_run,
        agenda_items: agenda.len(),
        runs_executed: runs.len(),
        sources_accepted,
        claims_extracted,
        findings_extracted,
        claims_promoted,
        candidate_claims,
        agenda: agenda.clone(),
        runs,
        warnings: dedup(warnings, 16),
        recommendations: dedup(recommendations, 12),
        metadata: json!({
            "schema_version": 1,
            "phase": "v0.7_deep_search_knowledge_autopilot",
            "bounded": true,
            "dry_run": request.dry_run,
            "max_topics": request.max_topics,
            "max_runs": max_runs,
            "max_sources_per_topic": request.max_sources_per_topic,
            "source": "memory_learning_autopilot",
            "metadata_only": true,
        }),
    };

    let _ = store.append_memory_note("deep_search_knowledge_autopilot_ran", json!({
        "schema_version": 1,
        "phase": "v0.7_deep_search_knowledge_autopilot",
        "dry_run": receipt.dry_run,
        "agenda_items": receipt.agenda_items,
        "runs_executed": receipt.runs_executed,
        "sources_accepted": receipt.sources_accepted,
        "claims_extracted": receipt.claims_extracted,
        "findings_extracted": receipt.findings_extracted,
        "claims_promoted": receipt.claims_promoted,
        "candidate_claims": receipt.candidate_claims,
        "reason": request.reason.clone(),
        "metadata_only": true,
    }));

    Ok(receipt)
}

fn build_deep_search_request(
    request: &DeepSearchKnowledgeAutopilotRequest,
    item: &DeepSearchLearningAgendaItem,
) -> DeepSearchRequest {
    let mut deep = request.deep_search_defaults.clone().unwrap_or_else(|| DeepSearchRequest {
        topic: item.topic.clone(),
        objective: Some(item.objective.clone()),
        query: Some(item.topic.clone()),
        seed_urls: Vec::new(),
        enable_web_discovery: Some(true),
        search_providers: request.search_providers.clone(),
        include_general_web: Some(true),
        include_academic_sources: Some(true),
        document_ingestion: Some(true),
        prefer_academic_landing_pages: Some(false),
        enable_pdf_text_extraction: Some(true),
        max_discovery_results_per_provider: Some(10),
        max_discovered_sources: Some(192),
        initial_query_count: Some(6),
        allowed_domains: Vec::new(),
        blocked_domains: Vec::new(),
        tags: Vec::new(),
        max_sources: Some(request.max_sources_per_topic.clamp(2, 24)),
        autonomous_loop: Some(true),
        max_research_passes: Some(5),
        min_research_passes: Some(2),
        max_sources_per_pass: Some((request.max_sources_per_topic.clamp(2, 24) / 2).max(4)),
        min_new_information_gain: Some(0.08),
        min_coverage_score: Some(0.66),
        min_supported_claim_ratio: Some(0.55),
        enable_claim_graph: Some(true),
        min_independent_sources_for_claim: Some(2),
        enable_contradiction_detection: Some(true),
        enable_memory_promotion_policy: Some(true),
        auto_promote_supported_claims: Some(true),
        require_user_confirmation_for_system_verified: Some(true),
        min_promotion_confidence: Some(0.62),
        min_promotion_independent_sources: Some(2),
        enable_source_reliability_scoring: Some(true),
        min_reliable_source_score_for_promotion: Some(0.50),
        min_sources_for_learning: Some(2),
        max_bytes_per_source: Some(2_000_000),
        timeout_ms: Some(180_000),
        require_cross_source_verification: true,
        allow_http_localhost: false,
        metadata: json!({}),
    });

    deep.topic = item.topic.clone();
    deep.objective = Some(item.objective.clone());
    deep.query = Some(item.topic.clone());
    deep.max_sources = Some(request.max_sources_per_topic.clamp(2, 24));
    deep.search_providers = if request.search_providers.is_empty() { deep.search_providers } else { request.search_providers.clone() };
    if deep.tags.is_empty() {
        deep.tags = vec!["knowledge_autopilot".into(), "continuous_learning".into()];
    }
    for tag in &item.tags {
        if !deep.tags.iter().any(|existing| existing.eq_ignore_ascii_case(tag)) {
            deep.tags.push(tag.clone());
        }
    }
    deep.metadata = json!({
        "source": "deep_search_knowledge_autopilot",
        "agenda_topic": item.topic.clone(),
        "agenda_reason": item.reason.clone(),
        "agenda_priority": item.priority,
        "source_node_ids": item.source_node_ids.clone(),
        "signals": item.signals.clone(),
        "external_content_untrusted": true,
        "system_verified_blocked_for_external_content": true,
    });
    deep
}

fn dedup(values: Vec<String>, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for value in values {
        if value.trim().is_empty() || out.iter().any(|existing| existing == &value) {
            continue;
        }
        out.push(value);
        if out.len() >= limit { break; }
    }
    out
}
