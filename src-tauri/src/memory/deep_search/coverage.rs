//! Heuristic coverage analysis for autonomous deep-search sessions.
//!
//! This intentionally uses deterministic Rust heuristics. The LLM may help with
//! later summarization, but the runtime decides whether a research loop is still
//! useful based on bounded measurable signals.

use super::DeepSearchDocument;
use super::types::{DeepSearchCoverageReport, DeepSearchSourceReliabilitySummary};
use std::collections::BTreeMap;
use std::collections::HashSet;

const ACADEMIC_MARKERS: &[&str] = &["arxiv", "pubmed", "crossref", "semantic_scholar", "europe_pmc", "doi.org", "ncbi", "paper", "study", "journal"];
const AUTHORITY_MARKERS: &[&str] = &["wikipedia", ".edu", ".gov", "docs.", "research", "institute", "foundation"];

pub(crate) fn analyze_coverage(topic: &str, query_history: &[String], documents: &[DeepSearchDocument]) -> DeepSearchCoverageReport {
    let topic_tokens = normalized_tokens(topic);
    let mut unique_domains = HashSet::<String>::new();
    let mut academic_sources = 0usize;
    let mut authoritative_sources = 0usize;
    let mut high_reliability_sources = 0usize;
    let mut low_reliability_sources = 0usize;
    let mut reliability_total = 0.0_f32;
    let mut reliability_min = 1.0_f32;
    let mut reliability_max = 0.0_f32;
    let mut tier_counts = BTreeMap::<String, usize>::new();
    let mut token_hits = 0usize;

    for document in documents {
        if let Some(domain) = host_from_url(&document.url) {
            unique_domains.insert(domain);
        }
        let url_lower = document.url.to_ascii_lowercase();
        let provider_lower = document.discovered_by.clone().unwrap_or_default().to_ascii_lowercase();
        let source_type_lower = document.source_type.clone().unwrap_or_default().to_ascii_lowercase();
        let combined = format!("{url_lower} {provider_lower} {source_type_lower}");
        if ACADEMIC_MARKERS.iter().any(|marker| combined.contains(marker)) {
            academic_sources += 1;
        }
        if AUTHORITY_MARKERS.iter().any(|marker| combined.contains(marker)) || academic_sources > 0 {
            authoritative_sources += 1;
        }
        reliability_total += document.reliability.score;
        reliability_min = reliability_min.min(document.reliability.score);
        reliability_max = reliability_max.max(document.reliability.score);
        if document.reliability.score >= 0.66 {
            high_reliability_sources += 1;
        }
        if document.reliability.score < 0.38 {
            low_reliability_sources += 1;
        }
        *tier_counts.entry(format!("{:?}", document.reliability.tier)).or_insert(0) += 1;
        let text_lower = document.text.to_ascii_lowercase();
        token_hits += topic_tokens.iter().filter(|token| text_lower.contains(token.as_str())).count();
    }

    let source_count = documents.len().max(1) as f32;
    let domain_diversity_score = (unique_domains.len() as f32 / source_count).clamp(0.0, 1.0);
    let academic_coverage_score = (academic_sources as f32 / source_count).clamp(0.0, 1.0);
    let authoritative_source_score = (authoritative_sources as f32 / source_count).clamp(0.0, 1.0);
    let source_reliability_score = if documents.is_empty() { 0.0 } else { (reliability_total / source_count).clamp(0.0, 1.0) };
    let topic_token_coverage = if topic_tokens.is_empty() {
        0.0
    } else {
        (token_hits as f32 / (topic_tokens.len().max(1) * documents.len().max(1)) as f32).clamp(0.0, 1.0)
    };
    let query_diversity_score = (query_history.iter().collect::<HashSet<_>>().len() as f32 / query_history.len().max(1) as f32).clamp(0.0, 1.0);
    let overall_score = ((domain_diversity_score * 0.24)
        + (academic_coverage_score * 0.20)
        + (authoritative_source_score * 0.16)
        + (source_reliability_score * 0.16)
        + (topic_token_coverage * 0.20)
        + (query_diversity_score * 0.08))
        .clamp(0.0, 1.0);

    let mut missing_subtopics = Vec::new();
    if academic_coverage_score < 0.20 { missing_subtopics.push("academic_sources".into()); }
    if domain_diversity_score < 0.45 { missing_subtopics.push("independent_domains".into()); }
    if authoritative_source_score < 0.35 { missing_subtopics.push("authoritative_sources".into()); }
    if source_reliability_score < 0.50 { missing_subtopics.push("higher_reliability_sources".into()); }
    if topic_token_coverage < 0.25 { missing_subtopics.push("topic_specific_evidence".into()); }

    DeepSearchCoverageReport {
        overall_score,
        domain_diversity_score,
        academic_coverage_score,
        authoritative_source_score,
        source_reliability_score,
        topic_token_coverage,
        query_diversity_score,
        unique_domains: unique_domains.len(),
        academic_sources,
        authoritative_sources,
        high_reliability_sources,
        low_reliability_sources,
        source_reliability: DeepSearchSourceReliabilitySummary {
            average_score: source_reliability_score,
            min_score: if documents.is_empty() { 0.0 } else { reliability_min },
            max_score: reliability_max,
            high_reliability_sources,
            low_reliability_sources,
            tier_counts,
        },
        missing_subtopics,
    }
}

fn host_from_url(url: &str) -> Option<String> {
    let without_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let host = without_scheme.split('/').next()?.split('@').last()?.split(':').next()?.trim().to_ascii_lowercase();
    if host.is_empty() { None } else { Some(host) }
}

fn normalized_tokens(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 4)
        .map(|token| token.to_ascii_lowercase())
        .take(32)
        .collect()
}
