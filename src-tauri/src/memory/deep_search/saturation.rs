//! Saturation scoring for bounded autonomous deep-search.

use super::DeepSearchDocument;
use super::types::{DeepSearchBudget, DeepSearchCoverageReport, DeepSearchSaturationReport, DeepSearchStopReason};
use std::collections::HashSet;

pub(crate) fn evaluate_saturation(
    pass_index: usize,
    budget: &DeepSearchBudget,
    coverage: &DeepSearchCoverageReport,
    previous_document_count: usize,
    documents: &[DeepSearchDocument],
) -> DeepSearchSaturationReport {
    let new_docs = documents.len().saturating_sub(previous_document_count);
    let new_information_gain = estimate_new_information_gain(previous_document_count, documents, new_docs);
    let duplicate_ratio = estimate_duplicate_ratio(documents);
    let supported_claim_ratio = if documents.len() >= 2 && coverage.unique_domains >= 2 { 0.62 } else { 0.28 };
    let score = ((coverage.overall_score * 0.52)
        + ((1.0 - duplicate_ratio) * 0.18)
        + (supported_claim_ratio * 0.18)
        + ((1.0 - new_information_gain).clamp(0.0, 1.0) * 0.12))
        .clamp(0.0, 1.0);

    let reached_saturation = pass_index + 1 >= budget.min_passes
        && coverage.overall_score >= budget.min_coverage_score
        && supported_claim_ratio >= budget.min_supported_claim_ratio
        && new_information_gain < budget.min_new_information_gain;

    DeepSearchSaturationReport {
        is_saturated: reached_saturation,
        score,
        new_information_gain,
        supported_claim_ratio,
        duplicate_ratio,
        missing_subtopics: coverage.missing_subtopics.clone(),
        stop_reason: if reached_saturation { Some(DeepSearchStopReason::Saturated) } else { None },
    }
}

fn estimate_new_information_gain(previous_document_count: usize, documents: &[DeepSearchDocument], new_docs: usize) -> f32 {
    if documents.is_empty() || new_docs == 0 { return 0.0; }
    if previous_document_count == 0 { return 1.0; }
    let previous_terms = documents.iter().take(previous_document_count).flat_map(|doc| signature_terms(&doc.text)).collect::<HashSet<_>>();
    let new_terms = documents.iter().skip(previous_document_count).flat_map(|doc| signature_terms(&doc.text)).collect::<HashSet<_>>();
    if new_terms.is_empty() { return 0.0; }
    let novel = new_terms.difference(&previous_terms).count();
    (novel as f32 / new_terms.len() as f32).clamp(0.0, 1.0)
}

fn estimate_duplicate_ratio(documents: &[DeepSearchDocument]) -> f32 {
    if documents.len() < 2 { return 0.0; }
    let mut seen = HashSet::<String>::new();
    let mut duplicates = 0usize;
    for doc in documents {
        let signature = signature_terms(&doc.text).into_iter().take(24).collect::<Vec<_>>().join("|");
        if !seen.insert(signature) { duplicates += 1; }
    }
    (duplicates as f32 / documents.len() as f32).clamp(0.0, 1.0)
}

fn signature_terms(text: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    text
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 6)
        .map(|token| token.to_ascii_lowercase())
        .filter(|token| seen.insert(token.clone()))
        .take(96)
        .collect()
}
