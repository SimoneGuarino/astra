//! Governed Deep-Search Foundation for AstraOS memory.
//!
//! This module deliberately does not create a second RAG system. It performs a
//! bounded, policy-checked acquisition pass and converts accepted evidence into
//! the existing `ResearchMemoryBundle`, so Rust Memory Graph consolidation,
//! verification state, embeddings and audit remain the source of truth.

pub mod academic;
pub mod claim_graph;
pub mod coverage;
pub mod discovery;
pub mod document;
pub mod pdf;
pub mod policy;
pub mod promotion;
pub mod query_expansion;
pub mod reliability;
pub mod saturation;
pub mod session;
pub mod source_graph;
pub mod types;

use crate::memory::{
    consolidation::consolidate_research_bundle,
    errors::{MemoryError, MemoryResult},
    store::MemoryGraphStore,
    types::{now_ms, MemoryVerificationStatus},
};
use crate::memory::consolidation::research::{ResearchClaim, ResearchFinding, ResearchMemoryBundle, ResearchProcedure, ResearchRecommendation, ResearchSource};
use academic::academic_hint_for_url;
use claim_graph::build_claim_graph;
use coverage::analyze_coverage;
use discovery::{discover_sources, DeepSearchDiscoveredSource};
use document::fetch_and_normalize_document;
use policy::{evaluate_source, DeepSearchPolicyDecision};
use promotion::apply_memory_promotion_policy;
use query_expansion::{expand_queries, initial_research_queries};
use reliability::score_source_reliability;
use saturation::evaluate_saturation;
use session::DeepSearchSessionState;
use source_graph::DeepSearchSourceGraphBuilder;
use reqwest::blocking::Client;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::{HashSet, VecDeque}, time::Duration};
pub use types::*;

const DEFAULT_USER_AGENT: &str = "AstraOS-DeepSearchFoundation/0.7.0 (+local-governed-memory)";
const MAX_TEXT_CHARS_PER_SOURCE: usize = 80_000;
const MAX_FINDINGS_PER_SOURCE: usize = 3;
const MAX_CLAIMS_PER_SOURCE: usize = 4;
const MAX_PROCEDURE_STEPS: usize = 8;

pub fn run_deep_search_foundation(
    store: &MemoryGraphStore,
    request: DeepSearchRequest,
) -> MemoryResult<DeepSearchReceipt> {
    validate_request(&request)?;
    let policy = DeepSearchPolicy::from_request(&request);
    let budget = DeepSearchBudget::from_request(&request, &policy);
    let mut session = DeepSearchSessionState::new();
    let mut warnings = Vec::<String>::new();
    let mut rejected_sources = Vec::<DeepSearchRejectedSource>::new();
    let mut accepted_documents = Vec::<DeepSearchDocument>::new();
    let mut seen_urls = HashSet::<String>::new();
    let mut source_graph = DeepSearchSourceGraphBuilder::default();
    let mut query_history = Vec::<String>::new();
    let mut query_queue = VecDeque::<String>::new();
    let mut stop_reason = DeepSearchStopReason::MaxPassesReached;

    for initial_query in initial_research_queries(
        &request.topic,
        request.query.as_deref().unwrap_or(&request.topic),
        request.initial_query_count.unwrap_or(5).clamp(1, 8),
    ) {
        if !query_queue.iter().any(|queued| queued.eq_ignore_ascii_case(&initial_query)) {
            query_queue.push_back(initial_query);
        }
    }
    if query_queue.is_empty() {
        query_queue.push_back(request.query.clone().unwrap_or_else(|| request.topic.clone()));
    }

    let client = Client::builder()
        .timeout(Duration::from_millis(budget.max_runtime_ms.clamp(10_000, 240_000)))
        .user_agent(DEFAULT_USER_AGENT)
        .redirect(reqwest::redirect::Policy::limited(8))
        .build()
        .map_err(|error| MemoryError::Storage(format!("deep-search http client build failed: {error}")))?;

    let autonomous_loop = request.autonomous_loop.unwrap_or(true);
    let max_passes = if autonomous_loop { budget.max_passes } else { 1 };

    for pass_index in 0..max_passes {
        if accepted_documents.len() >= budget.max_total_sources {
            stop_reason = DeepSearchStopReason::BudgetExhausted;
            break;
        }

        let Some(query) = query_queue.pop_front() else {
            stop_reason = DeepSearchStopReason::NoFollowUpQueries;
            break;
        };
        if query_history.iter().any(|existing| existing.eq_ignore_ascii_case(&query)) {
            continue;
        }
        query_history.push(query.clone());

        let previous_document_count = accepted_documents.len();
        let mut pass_request = request.clone();
        pass_request.query = Some(query.clone());
        pass_request.max_sources = Some(budget.max_sources_per_pass.min(budget.max_total_sources.saturating_sub(accepted_documents.len())).max(1));
        pass_request.max_discovered_sources = Some(budget.max_total_candidates);
        // Explicit seed URLs are consumed in the first pass only. Later passes should be
        // driven by query expansion, otherwise user-provided seeds can dominate the loop.
        if pass_index > 0 {
            pass_request.seed_urls.clear();
        }

        let pass_result = execute_deep_search_pass(
            &client,
            &pass_request,
            &policy,
            &budget,
            &mut seen_urls,
            &mut source_graph,
            &mut warnings,
        );

        rejected_sources.extend(pass_result.rejected_sources);
        accepted_documents.extend(pass_result.accepted_documents);

        let coverage = analyze_coverage(&request.topic, &query_history, &accepted_documents);
        let saturation = evaluate_saturation(pass_index, &budget, &coverage, previous_document_count, &accepted_documents);
        let follow_up_queries = if saturation.is_saturated || pass_index + 1 >= max_passes {
            Vec::new()
        } else {
            expand_queries(&request.topic, &query, &coverage, &accepted_documents, &query_history, 3)
        };
        for follow_up in &follow_up_queries {
            if !query_queue.iter().any(|queued| queued.eq_ignore_ascii_case(follow_up))
                && !query_history.iter().any(|used| used.eq_ignore_ascii_case(follow_up))
            {
                query_queue.push_back(follow_up.clone());
            }
        }

        session.passes.push(DeepSearchPassSummary {
            pass_index: pass_index + 1,
            query,
            candidates_seen: pass_result.candidate_count,
            accepted_sources: pass_result.accepted_count,
            rejected_sources: pass_result.rejected_count,
            new_information_gain: saturation.new_information_gain,
            coverage_score: coverage.overall_score,
            saturation_score: saturation.score,
            generated_follow_up_queries: follow_up_queries,
        });

        if saturation.is_saturated {
            stop_reason = DeepSearchStopReason::Saturated;
            break;
        }
        if query_queue.is_empty() && pass_index + 1 < max_passes {
            stop_reason = DeepSearchStopReason::NoFollowUpQueries;
            break;
        }
        if pass_index + 1 >= max_passes {
            stop_reason = DeepSearchStopReason::MaxPassesReached;
        }
    }

    if accepted_documents.is_empty() {
        if matches!(stop_reason, DeepSearchStopReason::MaxPassesReached) {
            stop_reason = DeepSearchStopReason::NoSourcesAccepted;
        }
        let completed_at = session.complete(stop_reason.clone());
        let coverage = DeepSearchCoverageReport::default();
        let mut saturation = DeepSearchSaturationReport::default();
        saturation.stop_reason = Some(stop_reason.clone());
        return Ok(DeepSearchReceipt {
            accepted: false,
            reason: "deep-search autonomous loop did not accept any source after governed discovery and source filtering".into(),
            run: session.run_summary(&request.topic, request.objective.clone(), source_graph.summary().total_seen, 0, rejected_sources.len()),
            consolidated: None,
            accepted_sources: Vec::new(),
            rejected_sources,
            extracted_claims: 0,
            extracted_findings: 0,
            warnings,
            passes: session.passes,
            coverage,
            saturation,
            claim_graph: None,
            promotion: None,
            metadata: json!({
                "schema_version": 2,
                "phase": "v0.7_2_source_reliability_scoring",
                "native_web_discovery": request.enable_web_discovery.unwrap_or(true),
                "autonomous_loop": autonomous_loop,
                "completed_at": completed_at,
                "source_graph": source_graph.summary(),
                "budget": budget,
                "query_history": query_history,
                "metadata_only": true,
            }),
        });
    }

    let final_coverage = analyze_coverage(&request.topic, &query_history, &accepted_documents);
    let mut final_saturation = evaluate_saturation(
        session.passes.len().saturating_sub(1),
        &budget,
        &final_coverage,
        accepted_documents.len().saturating_sub(1),
        &accepted_documents,
    );
    final_saturation.stop_reason = Some(stop_reason.clone());
    let _completed_at = session.complete(stop_reason.clone());

    let (bundle, claim_graph, promotion_report) = build_research_bundle(
        &request,
        &accepted_documents,
        &policy,
        &final_coverage,
        &final_saturation,
        &mut warnings,
    )?;
    let extracted_claims = bundle.claims.len();
    let extracted_findings = bundle.findings.len();
    final_saturation.supported_claim_ratio = claim_graph.cross_source_verified_ratio;
    let consolidation = consolidate_research_bundle(store, bundle)?;
    let accepted_sources = accepted_documents.iter().map(DeepSearchAcceptedSource::from_document).collect::<Vec<_>>();
    let source_graph_summary = source_graph.summary();

    let _ = store.append_memory_note("deep_search_autonomous_research_loop", json!({
        "schema_version": 2,
        "topic": request.topic.clone(),
        "objective": request.objective.clone(),
        "accepted_sources": accepted_sources.len(),
        "rejected_sources": rejected_sources.len(),
        "extracted_claims": extracted_claims,
        "extracted_findings": extracted_findings,
        "topic_node_id": consolidation.topic_node.id.clone(),
        "created_node_ids": consolidation.created_node_ids.clone(),
        "bounded": true,
        "autonomous_loop": autonomous_loop,
        "passes": session.passes.clone(),
        "coverage": final_coverage.clone(),
        "saturation": final_saturation.clone(),
        "claim_graph": {
            "cluster_count": claim_graph.clusters.len(),
            "supported_claims": claim_graph.supported_claims,
            "contradicted_claims": claim_graph.contradicted_claims,
            "cross_source_verified_ratio": claim_graph.cross_source_verified_ratio,
        },
        "promotion_policy": {
            "enabled": promotion_report.enabled,
            "promoted_claims": promotion_report.promoted_claims,
            "candidate_claims": promotion_report.candidate_claims,
            "review_required_claims": promotion_report.review_required_claims,
            "blocked_claims": promotion_report.blocked_claims,
            "external_content_never_system_verified": true,
        },
        "stop_reason": stop_reason,
        "query_history": query_history,
        "source_graph": source_graph_summary.clone(),
        "metadata_only": true,
    }));

    Ok(DeepSearchReceipt {
        accepted: true,
        reason: "deep-search autonomous research loop completed and consolidated into the governed Memory Graph".into(),
        run: session.run_summary(
            &request.topic,
            request.objective.clone(),
            source_graph_summary.total_seen,
            accepted_sources.len(),
            rejected_sources.len(),
        ),
        consolidated: Some(consolidation.into()),
        accepted_sources,
        rejected_sources,
        extracted_claims,
        extracted_findings,
        warnings,
        passes: session.passes,
        coverage: final_coverage.clone(),
        saturation: final_saturation,
        claim_graph: Some(claim_graph.clone()),
        promotion: Some(promotion_report.clone()),
        metadata: json!({
            "schema_version": 3,
            "phase": "v0.7_2_source_reliability_scoring",
            "source_of_truth": "memory_graph_research_consolidation",
            "native_web_discovery": request.enable_web_discovery.unwrap_or(true),
            "autonomous_loop": autonomous_loop,
            "source_graph": source_graph_summary,
            "claim_graph": {
                "cluster_count": claim_graph.clusters.len(),
                "supported_claims": claim_graph.supported_claims,
                "contradicted_claims": claim_graph.contradicted_claims,
                "unverified_claims": claim_graph.unverified_claims,
                "independent_source_ratio": claim_graph.independent_source_ratio,
                "cross_source_verified_ratio": claim_graph.cross_source_verified_ratio,
            },
            "source_reliability": final_coverage.source_reliability.clone(),
            "promotion_policy": {
                "enabled": promotion_report.enabled,
                "promoted_claims": promotion_report.promoted_claims,
                "candidate_claims": promotion_report.candidate_claims,
                "review_required_claims": promotion_report.review_required_claims,
                "blocked_claims": promotion_report.blocked_claims,
                "external_content_never_system_verified": true,
            },
            "budget": budget,
            "untrusted_external_content": true,
            "metadata_only": true,
        }),
    })
}

struct DeepSearchPassResult {
    candidate_count: usize,
    accepted_count: usize,
    rejected_count: usize,
    accepted_documents: Vec<DeepSearchDocument>,
    rejected_sources: Vec<DeepSearchRejectedSource>,
}

fn execute_deep_search_pass(
    client: &Client,
    request: &DeepSearchRequest,
    policy: &DeepSearchPolicy,
    budget: &DeepSearchBudget,
    seen_urls: &mut HashSet<String>,
    source_graph: &mut DeepSearchSourceGraphBuilder,
    warnings: &mut Vec<String>,
) -> DeepSearchPassResult {
    let pass_started_at = now_ms();
    let mut rejected_sources = Vec::<DeepSearchRejectedSource>::new();
    let mut accepted_documents = Vec::<DeepSearchDocument>::new();

    let mut candidates = request
        .seed_urls
        .iter()
        .enumerate()
        .map(|(index, url)| DeepSearchDiscoveredSource {
            url: url.clone(),
            provider: "user_seed".into(),
            source_type: "explicit_seed".into(),
            title: None,
            rank: index + 1,
            discovered_at: pass_started_at,
        })
        .collect::<Vec<_>>();
    candidates.extend(discover_sources(client, request, policy, warnings));

    let candidate_count = candidates.len();
    let max_sources_this_pass = request
        .max_sources
        .unwrap_or(budget.max_sources_per_pass)
        .min(budget.max_sources_per_pass)
        .max(1);

    for candidate in candidates.into_iter() {
        if accepted_documents.len() >= max_sources_this_pass {
            break;
        }
        let canonical_url = canonicalize_url(&candidate.url);
        if canonical_url.is_empty() {
            continue;
        }
        if !source_graph.note_candidate(&canonical_url, &candidate.provider) {
            continue;
        }
        let url_key = canonical_url.to_ascii_lowercase();
        if !seen_urls.insert(url_key) {
            continue;
        }

        match evaluate_source(&canonical_url, policy) {
            DeepSearchPolicyDecision::Allow => {}
            DeepSearchPolicyDecision::Reject { reason } => {
                source_graph.note_rejected();
                rejected_sources.push(DeepSearchRejectedSource { url: canonical_url, reason });
                continue;
            }
        }

        let document_result = if request.document_ingestion.unwrap_or(true) {
            fetch_and_normalize_document(client, &canonical_url, request)
        } else {
            fetch_and_normalize(client, &canonical_url, request)
        };

        match document_result {
            Ok(mut document) => {
                document.discovered_by = Some(candidate.provider);
                document.source_type = Some(candidate.source_type);
                document.discovery_rank = Some(candidate.rank);
                document.reliability = score_source_reliability(&document);
                source_graph.note_accepted_with_reliability(&document.url, &document.reliability);
                accepted_documents.push(document);
            }
            Err(error) => {
                source_graph.note_rejected();
                rejected_sources.push(DeepSearchRejectedSource {
                    url: canonical_url,
                    reason: error.to_string(),
                });
            }
        }
    }

    DeepSearchPassResult {
        candidate_count,
        accepted_count: accepted_documents.len(),
        rejected_count: rejected_sources.len(),
        accepted_documents,
        rejected_sources,
    }
}

fn validate_request(request: &DeepSearchRequest) -> MemoryResult<()> {
    if request.topic.trim().is_empty() {
        return Err(MemoryError::Validation("deep-search topic is required".into()));
    }
    if request.topic.chars().count() > 240 {
        return Err(MemoryError::Validation("deep-search topic is too long".into()));
    }
    if request.max_sources.unwrap_or(12) > 48 {
        return Err(MemoryError::Validation("deep-search max_sources cannot exceed 48 in v0.6 foundation".into()));
    }
    if request.seed_urls.is_empty() && request.enable_web_discovery == Some(false) {
        return Err(MemoryError::Validation(
            "deep-search requires seed_urls when native web discovery is explicitly disabled".into(),
        ));
    }
    if request.max_discovered_sources.unwrap_or(96) > 256 {
        return Err(MemoryError::Validation("deep-search max_discovered_sources cannot exceed 256".into()));
    }
    if request.max_discovery_results_per_provider.unwrap_or(8) > 16 {
        return Err(MemoryError::Validation("deep-search max_discovery_results_per_provider cannot exceed 16".into()));
    }
    if request.max_research_passes.unwrap_or(5) > 8 {
        return Err(MemoryError::Validation("deep-search max_research_passes cannot exceed 8".into()));
    }
    if request.max_sources_per_pass.unwrap_or(4) > 16 {
        return Err(MemoryError::Validation("deep-search max_sources_per_pass cannot exceed 16".into()));
    }
    if request.min_independent_sources_for_claim.unwrap_or(2) > 4 {
        return Err(MemoryError::Validation("deep-search min_independent_sources_for_claim cannot exceed 4 in v0.6.6".into()));
    }
    if request.min_promotion_independent_sources.unwrap_or(2) > 4 {
        return Err(MemoryError::Validation("deep-search min_promotion_independent_sources cannot exceed 4 in v0.6.7".into()));
    }
    if request.min_promotion_confidence.unwrap_or(0.62) > 0.95 {
        return Err(MemoryError::Validation("deep-search min_promotion_confidence cannot exceed 0.95 in v0.6.7".into()));
    }

    if request.min_reliable_source_score_for_promotion.unwrap_or(0.50) > 0.95 {
        return Err(MemoryError::Validation("deep-search min_reliable_source_score_for_promotion cannot exceed 0.95 in v0.7.2".into()));
    }
    Ok(())
}

fn fetch_and_normalize(client: &Client, url: &str, request: &DeepSearchRequest) -> MemoryResult<DeepSearchDocument> {
    let response = client
        .get(url)
        .send()
        .map_err(|error| MemoryError::Storage(format!("source fetch failed: {error}")))?;
    let final_url = response.url().to_string();
    let status = response.status();
    if !status.is_success() {
        return Err(MemoryError::Validation(format!("source returned non-success status {status}")));
    }
    let headers = response.headers().clone();
    let content_type = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !content_type.is_empty()
        && !(content_type.contains("text/html") || content_type.contains("text/plain") || content_type.contains("application/xhtml"))
    {
        return Err(MemoryError::Validation(format!("unsupported content-type for deep-search: {content_type}")));
    }
    let body = response
        .text()
        .map_err(|error| MemoryError::Storage(format!("source body read failed: {error}")))?;
    let max_bytes = request.max_bytes_per_source.unwrap_or(512_000).clamp(16_000, 2_000_000);
    if body.len() > max_bytes {
        return Err(MemoryError::Validation(format!("source exceeded max_bytes_per_source ({max_bytes})")));
    }
    let text = redact_sensitive_text_basic(&html_to_text(&body));
    let normalized = cap_chars(collapse_ws(&text), MAX_TEXT_CHARS_PER_SOURCE);
    if normalized.chars().count() < 160 {
        return Err(MemoryError::Validation("source text is too short after normalization".into()));
    }
    let title = extract_title(&body).unwrap_or_else(|| final_url.clone());
    let content_hash = sha256_hex(&normalized);
    Ok(DeepSearchDocument {
        url: final_url,
        title: cap_chars(title, 220),
        text: normalized.clone(),
        content_hash,
        fetched_at: now_ms(),
        content_type: if content_type.is_empty() { None } else { Some(content_type) },
        discovered_by: None,
        source_type: None,
        discovery_rank: None,
        document_kind: Some("web_document".into()),
        doi: None,
        academic_id: None,
        published_at: None,
        abstract_present: false,
        section_count: 1,
        sections: vec![DeepSearchDocumentSection { ordinal: 1, title: "Document body".into(), text: normalized.clone(), char_count: normalized.chars().count() }],
        pdf_extracted: false,
        extraction_method: Some("legacy_html_text_normalizer".into()),
        reliability: Default::default(),
        metadata: json!({"legacy_normalizer": true}),
    })
}

fn build_research_bundle(
    request: &DeepSearchRequest,
    documents: &[DeepSearchDocument],
    policy: &DeepSearchPolicy,
    coverage: &DeepSearchCoverageReport,
    saturation: &DeepSearchSaturationReport,
    warnings: &mut Vec<String>,
) -> MemoryResult<(ResearchMemoryBundle, DeepSearchClaimGraphReport, DeepSearchPromotionReport)> {
    let mut sources = Vec::<ResearchSource>::new();
    let mut findings = Vec::<ResearchFinding>::new();
    let mut claims = Vec::<ResearchClaim>::new();
    let mut procedures = Vec::<ResearchProcedure>::new();
    let mut recommendations = Vec::<ResearchRecommendation>::new();

    let claim_graph = if request.enable_claim_graph.unwrap_or(true) {
        build_claim_graph(
            &request.topic,
            documents,
            policy.require_cross_source_verification,
            request.min_independent_sources_for_claim.unwrap_or(2).clamp(1, 4),
            request.enable_contradiction_detection.unwrap_or(true),
        )
    } else {
        DeepSearchClaimGraphReport::default()
    };
    warnings.extend(claim_graph.warnings.iter().cloned());
    let promotion_report = apply_memory_promotion_policy(request, &claim_graph, coverage, saturation);
    warnings.extend(promotion_report.warnings.iter().cloned());

    for (index, doc) in documents.iter().enumerate() {
        let source_ref = format!("source_{}", index + 1);
        let summary = summarize_text(&doc.text, 520);
        let academic_hint = academic_hint_for_url(&doc.url, doc.source_type.as_deref());
        sources.push(ResearchSource {
            title: doc.title.clone(),
            uri: Some(doc.url.clone()),
            source_type: Some("web_document".into()),
            summary: Some(summary.clone()),
            confidence: Some((0.42 + (doc.reliability.score * 0.42)).clamp(0.35, 0.84)),
            metadata: json!({
                "source_ref": source_ref.clone(),
                "content_hash": doc.content_hash.clone(),
                "discovered_by": doc.discovered_by.clone(),
                "source_type": doc.source_type.clone(),
                "discovery_rank": doc.discovery_rank,
                "fetched_at": doc.fetched_at,
                "content_type": doc.content_type.clone(),
                "document_kind": doc.document_kind.clone(),
                "doi": doc.doi.clone(),
                "academic_id": doc.academic_id.clone(),
                "published_at": doc.published_at.clone(),
                "abstract_present": doc.abstract_present,
                "section_count": doc.section_count,
                "section_titles": doc.sections.iter().take(16).map(|section| section.title.clone()).collect::<Vec<_>>(),
                "pdf_extracted": doc.pdf_extracted,
                "extraction_method": doc.extraction_method.clone(),
                "document_metadata": doc.metadata.clone(),
                "academic_hint": academic_hint.clone(),
                "source_reliability": doc.reliability.clone(),
                "source_reliability_score": doc.reliability.score,
                "source_reliability_tier": doc.reliability.tier.clone(),
                "trust_class": "untrusted_external_web",
                "requires_evidence": true,
                "metadata_only": false,
            }),
        });

        for (ordinal, evidence) in select_evidence_chunks(doc, &request.topic, MAX_FINDINGS_PER_SOURCE).into_iter().enumerate() {
            findings.push(ResearchFinding {
                title: cap_chars(format!("{} — {} evidence {}", doc.title, evidence.section_title, ordinal + 1), 180),
                summary: cap_chars(evidence.text.clone(), 1_200),
                evidence: vec![cap_chars(evidence.text.clone(), 1_600)],
                source_refs: vec![source_ref.clone()],
                confidence: Some((0.36 + (doc.reliability.score * 0.34) + if doc.pdf_extracted { 0.03 } else { 0.0 }).clamp(0.35, 0.75)),
                tags: vec!["deep_search".into(), "web_evidence".into(), "section_grounded".into()],
                metadata: json!({
                    "document_url": doc.url.clone(),
                    "content_hash": doc.content_hash.clone(),
                    "discovered_by": doc.discovered_by.clone(),
                    "source_type": doc.source_type.clone(),
                    "extraction": "section_aware_document_ingestion_v0_6_5",
                    "section_title": evidence.section_title,
                    "section_ordinal": evidence.section_ordinal,
                    "document_kind": doc.document_kind.clone(),
                    "doi": doc.doi.clone(),
                    "academic_id": doc.academic_id.clone().or_else(|| academic_hint.identifier.clone()),
                    "published_at": doc.published_at.clone(),
                    "abstract_present": doc.abstract_present,
                    "section_count": doc.section_count,
                    "pdf_extracted": doc.pdf_extracted,
                    "extraction_method": doc.extraction_method.clone(),
                    "source_reliability": doc.reliability.clone(),
                    "source_reliability_score": doc.reliability.score,
                    "source_reliability_tier": doc.reliability.tier.clone(),
                    "verification_policy": "source_grounded_unverified_by_default",
                }),
            });
        }

        let procedure_steps = extract_procedure_steps(&doc.text, MAX_PROCEDURE_STEPS);
        if !procedure_steps.is_empty() {
            procedures.push(ResearchProcedure {
                title: cap_chars(format!("Procedure hints from {}", doc.title), 180),
                steps: procedure_steps,
                rationale: Some("Heuristic procedure extraction from web source; must remain advisory until verified.".into()),
                confidence: Some(0.42),
                metadata: json!({
                    "source_ref": source_ref.clone(),
                    "document_url": doc.url.clone(),
                    "discovered_by": doc.discovered_by.clone(),
                    "source_type": doc.source_type.clone(),
                    "source_reliability": doc.reliability.clone(),
                    "source_reliability_score": doc.reliability.score,
                    "external_untrusted": true
                }),
            });
        }
    }

    let promotion_by_cluster = promotion_report
        .decisions
        .iter()
        .map(|decision| (decision.claim_cluster_id.clone(), decision.clone()))
        .collect::<std::collections::HashMap<_, _>>();

    for cluster in claim_graph.clusters.iter().take(MAX_CLAIMS_PER_SOURCE * documents.len().max(1)).take(32) {
        let promotion_decision = promotion_by_cluster.get(&cluster.id);
        let promoted_confidence = promotion_decision
            .map(|decision| decision.confidence)
            .unwrap_or(cluster.confidence);
        let promoted_status = promotion_decision
            .map(|decision| decision.verification_status.clone())
            .unwrap_or_else(|| cluster.verification_status.clone());
        claims.push(ResearchClaim {
            claim: cap_chars(cluster.representative_claim.clone(), 800),
            rationale: Some(format!(
                "Claim graph cluster {}: support_count={}, independent_domains={}, contradiction_risk={:.2}. Promotion stage={}. External sources remain advisory evidence; system_verified requires a separate governed confirmation path.",
                cluster.id,
                cluster.support_count,
                cluster.independent_domain_count,
                cluster.contradiction_risk,
                promotion_decision
                    .map(|decision| format!("{:?}", decision.stage))
                    .unwrap_or_else(|| "unknown".into())
            )),
            source_refs: cluster.source_refs.clone(),
            confidence: Some(promoted_confidence),
            verification_status: Some(promoted_status),
            metadata: json!({
                "claim_cluster_id": cluster.id.clone(),
                "normalized_claim": cluster.normalized_claim.clone(),
                "support_count": cluster.support_count,
                "independent_domain_count": cluster.independent_domain_count,
                "contradiction_risk": cluster.contradiction_risk,
                "evidence_refs": cluster.evidence_refs.clone(),
                "external_untrusted": true,
                "evidence_required": true,
                "verification_pipeline": "claim_graph_cross_source_v0_6_6",
                "promotion_pipeline": "memory_promotion_policy_v0_6_7",
                "promotion_decision": promotion_decision.cloned(),
                "system_verified_blocked_for_external_content": true,
            }),
        });
    }

    if documents.len() < request.min_sources_for_learning.unwrap_or(1).clamp(1, 12) {
        warnings.push("deep-search accepted fewer sources than min_sources_for_learning; consolidated memory remains unverified".into());
    }

    recommendations.push(ResearchRecommendation {
        title: "Deep-search follow-up verification".into(),
        summary: "Run another bounded pass with additional independent sources before promoting claims to system_verified.".into(),
        actionability: Some("memory_quality_follow_up".into()),
        confidence: Some(0.76),
        metadata: json!({"policy": "do_not_promote_external_claims_without_cross_source_evidence"}),
    });

    let verification_status = if policy.require_cross_source_verification && documents.len() < 2 {
        MemoryVerificationStatus::Unverified
    } else {
        MemoryVerificationStatus::LlmInferred
    };

    Ok((ResearchMemoryBundle {
        topic: request.topic.clone(),
        objective: request.objective.clone(),
        query: request.query.clone().or_else(|| Some(request.topic.clone())),
        summary: Some(format!(
            "Governed deep-search run over {} accepted source(s). Claims were clustered in a cross-source claim graph; external content remains advisory until promotion policy review.",
            documents.len()
        )),
        confidence: Some(0.58),
        verification_status: Some(verification_status),
        tags: merge_tags(&request.tags, &["deep_search", "multi_provider_web_research", "academic_discovery", "document_ingestion", "astra_v0_6_6"]),
        sources,
        findings,
        claims,
        procedures,
        recommendations,
        metadata: json!({
            "schema_version": 1,
            "pipeline": "astra_memory_deep_search_memory_promotion_policy_v0_6_7",
            "bounded": true,
            "external_untrusted": true,
            "requires_source_evidence": true,
            "max_sources": request.max_sources,
            "min_sources_for_learning": request.min_sources_for_learning,
            "source_policy": policy,
            "claim_graph": {
                "cluster_count": claim_graph.clusters.len(),
                "supported_claims": claim_graph.supported_claims,
                "contradicted_claims": claim_graph.contradicted_claims,
                "unverified_claims": claim_graph.unverified_claims,
                "independent_source_ratio": claim_graph.independent_source_ratio,
                "cross_source_verified_ratio": claim_graph.cross_source_verified_ratio,
            },
            "source_reliability": coverage.source_reliability.clone(),
            "promotion_policy": {
                "enabled": promotion_report.enabled,
                "promoted_claims": promotion_report.promoted_claims,
                "candidate_claims": promotion_report.candidate_claims,
                "review_required_claims": promotion_report.review_required_claims,
                "blocked_claims": promotion_report.blocked_claims,
                "external_content_never_system_verified": true,
            },
        }),
    }, claim_graph, promotion_report))
}


#[derive(Debug, Clone)]
struct SectionEvidenceChunk {
    section_title: String,
    section_ordinal: usize,
    text: String,
}

fn select_evidence_chunks(doc: &DeepSearchDocument, topic: &str, limit: usize) -> Vec<SectionEvidenceChunk> {
    let topic_tokens = normalized_tokens(topic);
    let mut scored = Vec::<(usize, usize, String, usize, String)>::new();
    for section in &doc.sections {
        for paragraph in section.text.split('\n').map(str::trim).filter(|p| p.chars().count() >= 120) {
            let lower = paragraph.to_ascii_lowercase();
            let score = topic_tokens.iter().filter(|token| lower.contains(token.as_str())).count();
            scored.push((score, paragraph.len(), section.title.clone(), section.ordinal, paragraph.to_string()));
        }
    }
    if scored.is_empty() {
        return select_evidence_paragraphs(&doc.text, topic, limit)
            .into_iter()
            .enumerate()
            .map(|(index, text)| SectionEvidenceChunk {
                section_title: "Document body".into(),
                section_ordinal: index + 1,
                text,
            })
            .collect();
    }
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    scored.into_iter().take(limit).map(|(_, _, section_title, section_ordinal, text)| SectionEvidenceChunk {
        section_title,
        section_ordinal,
        text: cap_chars(text, 1_600),
    }).collect()
}

fn select_evidence_paragraphs(text: &str, topic: &str, limit: usize) -> Vec<String> {
    let topic_tokens = normalized_tokens(topic);
    let mut scored = text
        .split('\n')
        .map(str::trim)
        .filter(|p| p.chars().count() >= 120)
        .map(|p| {
            let lower = p.to_ascii_lowercase();
            let score = topic_tokens.iter().filter(|token| lower.contains(token.as_str())).count();
            (score, p.to_string())
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.len().cmp(&a.1.len())));
    scored.into_iter().take(limit).map(|(_, p)| cap_chars(p, 1_600)).collect()
}

fn extract_claim_sentences(text: &str, topic: &str, limit: usize) -> Vec<String> {
    let topic_tokens = normalized_tokens(topic);
    let mut sentences = split_sentences(text)
        .into_iter()
        .filter(|s| s.chars().count() >= 80 && s.chars().count() <= 700)
        .map(|s| {
            let lower = s.to_ascii_lowercase();
            let topic_score = topic_tokens.iter().filter(|token| lower.contains(token.as_str())).count();
            let signal_score = ["is ", "are ", "can ", "should ", "shows ", "requires ", "because ", "therefore ", "study", "research", "evidence"]
                .iter()
                .filter(|needle| lower.contains(**needle))
                .count();
            (topic_score + signal_score, s)
        })
        .filter(|(score, _)| *score > 0)
        .collect::<Vec<_>>();
    sentences.sort_by(|a, b| b.0.cmp(&a.0));
    sentences.into_iter().take(limit).map(|(_, s)| s).collect()
}

fn extract_procedure_steps(text: &str, limit: usize) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            lower.starts_with("step ") || lower.starts_with("1.") || lower.starts_with("2.") || lower.starts_with("- ") || lower.contains(" install ") || lower.contains(" configure ")
        })
        .map(|line| cap_chars(line.trim_start_matches("- ").to_string(), 280))
        .take(limit)
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '?' | '!') && current.chars().count() >= 40 {
            values.push(collapse_ws(&current));
            current.clear();
        }
    }
    if current.chars().count() >= 40 { values.push(collapse_ws(&current)); }
    values
}

fn summarize_text(text: &str, max_chars: usize) -> String {
    select_evidence_paragraphs(text, "", 1).into_iter().next().unwrap_or_else(|| cap_chars(text.to_string(), max_chars))
}

fn html_to_text(html: &str) -> String {
    let mut output = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => { in_tag = true; output.push(' '); }
            '>' => { in_tag = false; output.push(' '); }
            _ if !in_tag => output.push(ch),
            _ => {}
        }
    }
    decode_basic_entities(&output)
}

fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let after_start = lower[start..].find('>')? + start + 1;
    let end = lower[after_start..].find("</title>")? + after_start;
    Some(collapse_ws(&decode_basic_entities(&html[after_start..end])))
}

fn decode_basic_entities(value: &str) -> String {
    value
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn canonicalize_url(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() { return String::new(); }
    trimmed.trim_end_matches('/').to_string()
}

fn collapse_ws(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cap_chars(value: impl Into<String>, max_chars: usize) -> String {
    let value = value.into();
    if value.chars().count() <= max_chars { return value; }
    value.chars().take(max_chars).collect()
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn normalized_tokens(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 3)
        .map(|token| token.to_ascii_lowercase())
        .take(24)
        .collect()
}

fn merge_tags(existing: &[String], extra: &[&str]) -> Vec<String> {
    let mut seen = HashSet::new();
    existing.iter().map(|s| s.as_str()).chain(extra.iter().copied())
        .map(|tag| tag.trim().to_ascii_lowercase())
        .filter(|tag| !tag.is_empty() && seen.insert(tag.clone()))
        .take(32)
        .collect()
}

fn url_encode(value: &str) -> String {
    value.bytes().map(|b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
        b' ' => "+".into(),
        _ => format!("%{b:02X}"),
    }).collect()
}

#[derive(Debug, Clone)]
pub(crate) struct DeepSearchDocument {
    pub(crate) url: String,
    pub(crate) title: String,
    pub(crate) text: String,
    pub(crate) content_hash: String,
    pub(crate) fetched_at: i64,
    pub(crate) content_type: Option<String>,
    pub(crate) discovered_by: Option<String>,
    pub(crate) source_type: Option<String>,
    pub(crate) discovery_rank: Option<usize>,
    pub(crate) document_kind: Option<String>,
    pub(crate) doi: Option<String>,
    pub(crate) academic_id: Option<String>,
    pub(crate) published_at: Option<String>,
    pub(crate) abstract_present: bool,
    pub(crate) section_count: usize,
    pub(crate) sections: Vec<DeepSearchDocumentSection>,
    pub(crate) pdf_extracted: bool,
    pub(crate) extraction_method: Option<String>,
    pub(crate) reliability: types::DeepSearchSourceReliability,
    pub(crate) metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub(crate) struct DeepSearchDocumentSection {
    pub(crate) ordinal: usize,
    pub(crate) title: String,
    pub(crate) text: String,
    pub(crate) char_count: usize,
}


fn redact_sensitive_text_basic(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let lower = line.to_ascii_lowercase();
            if lower.contains("api_key") || lower.contains("apikey") || lower.contains("authorization:") || lower.contains("bearer ") || lower.contains("password") || lower.contains("secret") || lower.contains("private_key") {
                "[REDACTED_SENSITIVE_LINE]".to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
