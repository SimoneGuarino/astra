//! Claim graph and cross-source verification for governed Deep Search.
//!
//! This module does not promote web content directly to truth. It builds a
//! bounded, evidence-linked claim graph from already accepted documents and
//! assigns conservative verification states that the Memory Graph can audit.

use super::{DeepSearchDocument, DeepSearchDocumentSection};
use super::types::{
    DeepSearchClaimCluster, DeepSearchClaimEvidenceRef, DeepSearchClaimGraphReport,
};
use crate::memory::types::MemoryVerificationStatus;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const MAX_CANDIDATES_PER_DOCUMENT: usize = 8;
const MAX_EVIDENCE_PER_CLUSTER: usize = 8;
const MAX_CLUSTER_COUNT: usize = 48;

#[derive(Debug, Clone)]
struct CandidateClaim {
    claim: String,
    normalized_key: String,
    negated: bool,
    evidence: DeepSearchClaimEvidenceRef,
}

pub(crate) fn build_claim_graph(
    topic: &str,
    documents: &[DeepSearchDocument],
    require_cross_source_verification: bool,
    min_independent_sources: usize,
    enable_contradiction_detection: bool,
) -> DeepSearchClaimGraphReport {
    let min_independent_sources = min_independent_sources.clamp(1, 4);
    let mut clusters = HashMap::<String, Vec<CandidateClaim>>::new();

    for (doc_index, document) in documents.iter().enumerate() {
        let source_ref = format!("source_{}", doc_index + 1);
        for candidate in extract_candidate_claims(topic, document, &source_ref, MAX_CANDIDATES_PER_DOCUMENT) {
            clusters.entry(candidate.normalized_key.clone()).or_default().push(candidate);
        }
    }

    let mut cluster_values = clusters
        .into_iter()
        .filter_map(|(key, candidates)| {
            if key.trim().is_empty() || candidates.is_empty() {
                return None;
            }
            Some(to_cluster(
                key,
                candidates,
                require_cross_source_verification,
                min_independent_sources,
                enable_contradiction_detection,
            ))
        })
        .collect::<Vec<_>>();

    cluster_values.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.support_count.cmp(&a.support_count))
            .then_with(|| b.independent_domain_count.cmp(&a.independent_domain_count))
    });
    cluster_values.truncate(MAX_CLUSTER_COUNT);

    let supported_claims = cluster_values
        .iter()
        .filter(|cluster| cluster.support_count >= 2 && cluster.independent_domain_count >= min_independent_sources)
        .count();
    let contradicted_claims = cluster_values
        .iter()
        .filter(|cluster| cluster.verification_status.eq(&MemoryVerificationStatus::Contradicted))
        .count();
    let unverified_claims = cluster_values
        .iter()
        .filter(|cluster| cluster.verification_status.eq(&MemoryVerificationStatus::Unverified))
        .count();
    let total = cluster_values.len().max(1) as f32;
    let independent_source_ratio = (cluster_values
        .iter()
        .filter(|cluster| cluster.independent_domain_count >= min_independent_sources)
        .count() as f32
        / total)
        .clamp(0.0, 1.0);
    let cross_source_verified_ratio = (supported_claims as f32 / total).clamp(0.0, 1.0);

    let mut warnings = Vec::new();
    if cluster_values.is_empty() {
        warnings.push("claim_graph produced no claim clusters from accepted documents".into());
    }
    if require_cross_source_verification && supported_claims == 0 && !cluster_values.is_empty() {
        warnings.push("claim_graph found claims, but none were independently supported by multiple sources".into());
    }
    if contradicted_claims > 0 {
        warnings.push(format!(
            "claim_graph detected {contradicted_claims} possible contradiction cluster(s); keep affected claims untrusted until reviewed"
        ));
    }

    DeepSearchClaimGraphReport {
        clusters: cluster_values,
        supported_claims,
        contradicted_claims,
        unverified_claims,
        independent_source_ratio,
        cross_source_verified_ratio,
        warnings,
        metadata: json!({
            "schema_version": 1,
            "pipeline": "astra_deep_search_claim_graph_v0_6_6",
            "bounded": true,
            "external_content_is_untrusted": true,
            "min_independent_sources": min_independent_sources,
            "require_cross_source_verification": require_cross_source_verification,
            "contradiction_detection": enable_contradiction_detection,
            "max_candidates_per_document": MAX_CANDIDATES_PER_DOCUMENT,
            "max_evidence_per_cluster": MAX_EVIDENCE_PER_CLUSTER,
        }),
    }
}

fn extract_candidate_claims(
    topic: &str,
    document: &DeepSearchDocument,
    source_ref: &str,
    limit: usize,
) -> Vec<CandidateClaim> {
    let topic_tokens = normalized_tokens(topic);
    let mut scored = Vec::<(usize, CandidateClaim)>::new();

    for section in candidate_sections(document) {
        for sentence in split_sentences(&section.text) {
            if sentence.chars().count() < 80 || sentence.chars().count() > 900 {
                continue;
            }
            let lower = sentence.to_ascii_lowercase();
            let topic_score = topic_tokens.iter().filter(|token| lower.contains(token.as_str())).count();
            let signal_score = claim_signal_score(&lower);
            if topic_score == 0 && signal_score < 2 {
                continue;
            }
            let normalized_key = claim_signature(&sentence);
            if normalized_key.is_empty() {
                continue;
            }
            let evidence = DeepSearchClaimEvidenceRef {
                source_ref: source_ref.to_string(),
                url: document.url.clone(),
                title: document.title.clone(),
                document_kind: document.document_kind.clone(),
                doi: document.doi.clone(),
                academic_id: document.academic_id.clone(),
                evidence: cap_chars(sentence.clone(), 1_200),
                section_title: Some(section.title.clone()),
                section_ordinal: Some(section.ordinal),
                content_hash: document.content_hash.clone(),
            };
            scored.push((
                topic_score + signal_score,
                CandidateClaim {
                    claim: cap_chars(sentence, 900),
                    normalized_key,
                    negated: contains_negation(&lower),
                    evidence,
                },
            ));
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.claim.len().cmp(&a.1.claim.len())));
    scored.into_iter().map(|(_, candidate)| candidate).take(limit).collect()
}

fn candidate_sections(document: &DeepSearchDocument) -> Vec<DeepSearchDocumentSection> {
    if !document.sections.is_empty() {
        return document.sections.iter().take(24).cloned().collect();
    }
    vec![DeepSearchDocumentSection {
        ordinal: 1,
        title: "Document body".into(),
        text: document.text.clone(),
        char_count: document.text.chars().count(),
    }]
}

fn to_cluster(
    normalized_key: String,
    candidates: Vec<CandidateClaim>,
    require_cross_source_verification: bool,
    min_independent_sources: usize,
    enable_contradiction_detection: bool,
) -> DeepSearchClaimCluster {
    let mut source_refs = HashSet::<String>::new();
    let mut domains = HashSet::<String>::new();
    let mut evidence_refs = Vec::<DeepSearchClaimEvidenceRef>::new();
    let mut positive = 0usize;
    let mut negated = 0usize;

    for candidate in &candidates {
        if candidate.negated { negated += 1; } else { positive += 1; }
        source_refs.insert(candidate.evidence.source_ref.clone());
        if let Some(domain) = domain_from_url(&candidate.evidence.url) {
            domains.insert(domain);
        }
        if evidence_refs.len() < MAX_EVIDENCE_PER_CLUSTER {
            evidence_refs.push(candidate.evidence.clone());
        }
    }

    let support_count = source_refs.len();
    let independent_domain_count = domains.len().max(support_count.min(1));
    let contradiction_risk = if enable_contradiction_detection && positive > 0 && negated > 0 {
        0.82
    } else {
        0.0
    };
    let cross_source_supported = support_count >= 2 && independent_domain_count >= min_independent_sources;
    let verification_status = if contradiction_risk >= 0.75 {
        MemoryVerificationStatus::Contradicted
    } else if require_cross_source_verification && !cross_source_supported {
        MemoryVerificationStatus::Unverified
    } else if cross_source_supported {
        // Conservative: cross-source support upgrades claims from unverified to inferred,
        // but does not promote untrusted web content to SystemVerified in this phase.
        MemoryVerificationStatus::LlmInferred
    } else {
        MemoryVerificationStatus::Unverified
    };
    let confidence = claim_confidence(support_count, independent_domain_count, contradiction_risk, cross_source_supported);
    let representative_claim = candidates
        .iter()
        .max_by(|a, b| a.claim.len().cmp(&b.claim.len()))
        .map(|candidate| candidate.claim.clone())
        .unwrap_or_else(|| normalized_key.clone());
    let source_ref_list = {
        let mut values = source_refs.into_iter().collect::<Vec<_>>();
        values.sort();
        values
    };

    DeepSearchClaimCluster {
        id: format!("claim_cluster_{}", short_hash(&normalized_key)),
        normalized_claim: normalized_key,
        representative_claim,
        support_count,
        source_refs: source_ref_list,
        independent_domain_count,
        confidence,
        verification_status,
        contradiction_risk,
        evidence_refs,
        metadata: json!({
            "schema_version": 1,
            "candidate_count": candidates.len(),
            "positive_candidate_count": positive,
            "negated_candidate_count": negated,
            "cross_source_supported": cross_source_supported,
            "min_independent_sources": min_independent_sources,
        }),
    }
}

fn claim_confidence(
    support_count: usize,
    independent_domain_count: usize,
    contradiction_risk: f32,
    cross_source_supported: bool,
) -> f32 {
    let support_component = (support_count as f32 / 4.0).clamp(0.0, 0.35);
    let independence_component = (independent_domain_count as f32 / 4.0).clamp(0.0, 0.30);
    let base = if cross_source_supported { 0.42 } else { 0.30 };
    (base + support_component + independence_component - (contradiction_risk * 0.35)).clamp(0.18, 0.82)
}

fn claim_signal_score(lower: &str) -> usize {
    [
        " is ", " are ", " can ", " may ", " should ", " shows ", " found ", " suggests ",
        " indicates ", " requires ", " improves ", " reduces ", " increases ", " because ",
        " therefore ", " evidence", " study", " research", " benchmark", " compared", " results",
    ]
    .iter()
    .filter(|needle| lower.contains(**needle))
    .count()
}

fn claim_signature(sentence: &str) -> String {
    let mut tokens = normalized_tokens(sentence)
        .into_iter()
        .filter(|token| !is_stopword(token))
        .collect::<Vec<_>>();
    tokens.sort();
    tokens.dedup();
    tokens.truncate(12);
    if tokens.len() < 4 {
        return String::new();
    }
    tokens.join(" ")
}

fn contains_negation(lower: &str) -> bool {
    [" not ", " no ", " never ", " cannot ", " can't ", " without ", " fails to ", " does not ", " do not ", " non-"]
        .iter()
        .any(|needle| lower.contains(*needle))
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
    if current.chars().count() >= 40 {
        values.push(collapse_ws(&current));
    }
    values
}

fn normalized_tokens(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 3)
        .map(|token| token.to_ascii_lowercase())
        .take(64)
        .collect()
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "the" | "and" | "for" | "that" | "this" | "with" | "from" | "have" | "has" | "had" |
        "are" | "was" | "were" | "will" | "would" | "can" | "could" | "should" | "may" |
        "might" | "into" | "than" | "then" | "also" | "such" | "their" | "there" | "these" |
        "those" | "using" | "used" | "use" | "between" | "within" | "while" | "where" | "when" |
        "come" | "comes" | "more" | "most" | "less" | "about" | "over" | "under" | "per" |
        "una" | "uno" | "gli" | "che" | "con" | "del" | "della" | "delle" | "degli"
    )
}

fn domain_from_url(url: &str) -> Option<String> {
    let without_scheme = url
        .trim()
        .strip_prefix("https://")
        .or_else(|| url.trim().strip_prefix("http://"))
        .unwrap_or(url.trim());
    let host = without_scheme.split('/').next()?.split('@').last()?.split(':').next()?.trim();
    if host.is_empty() { None } else { Some(host.to_ascii_lowercase()) }
}

fn collapse_ws(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cap_chars(value: impl Into<String>, max_chars: usize) -> String {
    let value = value.into();
    if value.chars().count() <= max_chars { return value; }
    value.chars().take(max_chars).collect()
}

fn short_hash(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize()).chars().take(16).collect()
}
