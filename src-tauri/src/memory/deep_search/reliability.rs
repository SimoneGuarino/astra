//! Deterministic source reliability scoring for AstraOS Deep Search.
//!
//! This module does not let the LLM decide whether a source is trustworthy.
//! It assigns a bounded, auditable reliability tier from URL/provider/document
//! metadata so downstream coverage and promotion gates can weight evidence
//! without turning external web content into `SystemVerified` memory.

use super::DeepSearchDocument;
use super::types::{DeepSearchSourceReliability, DeepSearchSourceReliabilityTier};

const OFFICIAL_DOC_MARKERS: &[&str] = &[
    "docs.", "developer.", "developers.", "learn.", "reference.", "api.", "manual.",
];
const ACADEMIC_MARKERS: &[&str] = &[
    "arxiv.org", "pubmed", "ncbi.nlm.nih.gov", "semanticscholar.org", "crossref.org",
    "doi.org", "springer.com", "sciencedirect.com", "nature.com", "acm.org", "ieee.org",
    "jstor.org", "frontiersin.org", "plos.org", "mdpi.com", "biorxiv.org", "medrxiv.org",
];
const GOVERNMENT_MARKERS: &[&str] = &[".gov", ".gov.", "europa.eu", "who.int", "nih.gov", "nist.gov"];
const ENCYCLOPEDIA_MARKERS: &[&str] = &["wikipedia.org", "wikidata.org", "britannica.com"];
const COMMUNITY_MARKERS: &[&str] = &["reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com", "news.ycombinator.com"];
const LOW_QUALITY_MARKERS: &[&str] = &["pinterest.", "tiktok.", "facebook.", "instagram.", "medium.com/p/", "clickbank", "casino", "betting"];

pub(crate) fn score_source_reliability(document: &DeepSearchDocument) -> DeepSearchSourceReliability {
    let host = host_from_url(&document.url).unwrap_or_else(|| "unknown".into());
    let url_lower = document.url.to_ascii_lowercase();
    let provider = document.discovered_by.clone().unwrap_or_default().to_ascii_lowercase();
    let source_type = document.source_type.clone().unwrap_or_default().to_ascii_lowercase();
    let document_kind = document.document_kind.clone().unwrap_or_default().to_ascii_lowercase();
    let content_type = document.content_type.clone().unwrap_or_default().to_ascii_lowercase();
    let mut score = 0.45_f32;
    let mut signals = Vec::<String>::new();
    let mut penalties = Vec::<String>::new();

    if document.doi.as_ref().is_some_and(|doi| !doi.trim().is_empty()) {
        score += 0.20;
        signals.push("doi_present".into());
    }
    if document.academic_id.as_ref().is_some_and(|id| !id.trim().is_empty()) {
        score += 0.14;
        signals.push("academic_identifier_present".into());
    }
    if document.published_at.as_ref().is_some_and(|date| !date.trim().is_empty()) {
        score += 0.05;
        signals.push("published_date_present".into());
    }
    if document.abstract_present {
        score += 0.05;
        signals.push("abstract_present".into());
    }
    if document.section_count >= 3 {
        score += 0.06;
        signals.push("sectioned_document".into());
    }
    if document.pdf_extracted {
        score += 0.03;
        signals.push("pdf_full_text_extracted".into());
    }
    if ACADEMIC_MARKERS.iter().any(|marker| url_lower.contains(marker))
        || provider.contains("arxiv")
        || provider.contains("pubmed")
        || provider.contains("crossref")
        || provider.contains("semantic")
        || provider.contains("europe_pmc")
        || source_type.contains("academic")
        || document_kind.contains("academic")
    {
        score += 0.20;
        signals.push("academic_or_scholarly_source".into());
    }
    if GOVERNMENT_MARKERS.iter().any(|marker| host.contains(marker) || url_lower.contains(marker)) {
        score += 0.18;
        signals.push("government_or_public_institution_source".into());
    }
    if OFFICIAL_DOC_MARKERS.iter().any(|marker| host.contains(marker) || url_lower.contains(marker))
        || source_type.contains("documentation")
        || document_kind.contains("documentation")
    {
        score += 0.16;
        signals.push("official_documentation_signal".into());
    }
    if ENCYCLOPEDIA_MARKERS.iter().any(|marker| host.contains(marker)) {
        score += 0.08;
        signals.push("encyclopedic_source".into());
    }
    if content_type.contains("application/pdf") {
        score += 0.04;
        signals.push("pdf_content_type".into());
    }

    if COMMUNITY_MARKERS.iter().any(|marker| host.contains(marker)) {
        score -= 0.16;
        penalties.push("community_or_forum_source".into());
    }
    if LOW_QUALITY_MARKERS.iter().any(|marker| host.contains(marker) || url_lower.contains(marker)) {
        score -= 0.22;
        penalties.push("low_quality_domain_marker".into());
    }
    if document.text.chars().count() < 1_000 {
        score -= 0.08;
        penalties.push("thin_extracted_text".into());
    }
    if document.title.trim().is_empty() || document.title.eq_ignore_ascii_case("untitled source") {
        score -= 0.05;
        penalties.push("missing_title".into());
    }
    if host == "unknown" {
        score -= 0.10;
        penalties.push("unknown_host".into());
    }

    let score = score.clamp(0.05, 0.98);
    let tier = classify_tier(score, &signals, &penalties);
    let reason = build_reason(&tier, score, &signals, &penalties);

    DeepSearchSourceReliability {
        score,
        tier,
        reason,
        signals,
        penalties,
    }
}

fn classify_tier(score: f32, signals: &[String], penalties: &[String]) -> DeepSearchSourceReliabilityTier {
    if penalties.iter().any(|value| value == "low_quality_domain_marker") && score < 0.45 {
        return DeepSearchSourceReliabilityTier::LowQuality;
    }
    if signals.iter().any(|value| value == "government_or_public_institution_source") && score >= 0.72 {
        return DeepSearchSourceReliabilityTier::GovernmentInstitutional;
    }
    if signals.iter().any(|value| value == "doi_present") && score >= 0.74 {
        return DeepSearchSourceReliabilityTier::AcademicPeerReviewedLike;
    }
    if signals.iter().any(|value| value == "academic_or_scholarly_source") && score >= 0.66 {
        return DeepSearchSourceReliabilityTier::AcademicPreprintOrIndex;
    }
    if signals.iter().any(|value| value == "official_documentation_signal") && score >= 0.66 {
        return DeepSearchSourceReliabilityTier::OfficialDocumentation;
    }
    if signals.iter().any(|value| value == "encyclopedic_source") && score >= 0.52 {
        return DeepSearchSourceReliabilityTier::EncyclopedicReference;
    }
    if penalties.iter().any(|value| value == "community_or_forum_source") {
        return DeepSearchSourceReliabilityTier::CommunityDiscussion;
    }
    if score >= 0.58 {
        DeepSearchSourceReliabilityTier::GeneralWebModerate
    } else if score >= 0.38 {
        DeepSearchSourceReliabilityTier::UnknownUnranked
    } else {
        DeepSearchSourceReliabilityTier::LowQuality
    }
}

fn build_reason(tier: &DeepSearchSourceReliabilityTier, score: f32, signals: &[String], penalties: &[String]) -> String {
    let signal_text = if signals.is_empty() { "no positive reliability signals".into() } else { signals.join(", ") };
    let penalty_text = if penalties.is_empty() { "no major penalties".into() } else { penalties.join(", ") };
    format!("tier={tier:?}; score={score:.2}; signals=[{signal_text}]; penalties=[{penalty_text}]")
}

fn host_from_url(url: &str) -> Option<String> {
    let without_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let host = without_scheme
        .split('/')
        .next()?
        .split('@')
        .last()?
        .split(':')
        .next()?
        .trim()
        .trim_start_matches("www.")
        .to_ascii_lowercase();
    if host.is_empty() { None } else { Some(host) }
}
