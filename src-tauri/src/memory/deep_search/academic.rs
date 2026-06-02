//! Academic-source helpers for Deep Search document ingestion.
//!
//! This file contains deterministic URL/type utilities only. Provider fetch and
//! policy remain in `discovery`/`policy`, while full Memory Graph promotion still
//! goes through the existing research consolidation pipeline.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcademicSourceHint {
    pub is_academic: bool,
    pub canonical_kind: String,
    pub identifier: Option<String>,
}

pub(crate) fn academic_hint_for_url(url: &str, source_type: Option<&str>) -> AcademicSourceHint {
    let lower = url.to_ascii_lowercase();
    let source_type = source_type.unwrap_or_default().to_ascii_lowercase();
    if let Some(id) = extract_arxiv_id(url) {
        return AcademicSourceHint { is_academic: true, canonical_kind: "academic_preprint".into(), identifier: Some(format!("arxiv:{id}")) };
    }
    if let Some(id) = extract_pubmed_id(url) {
        return AcademicSourceHint { is_academic: true, canonical_kind: "academic_biomedical".into(), identifier: Some(format!("pmid:{id}")) };
    }
    if let Some(doi) = extract_doi_from_url(url) {
        return AcademicSourceHint { is_academic: true, canonical_kind: "academic_doi".into(), identifier: Some(format!("doi:{doi}")) };
    }
    let is_academic = source_type.contains("academic")
        || lower.contains("scholar.google")
        || lower.contains("semanticscholar.org")
        || lower.contains("crossref.org")
        || lower.contains("pubmed.ncbi.nlm.nih.gov")
        || lower.contains("ncbi.nlm.nih.gov/pmc")
        || lower.contains("europepmc.org");
    AcademicSourceHint {
        is_academic,
        canonical_kind: if is_academic { "academic_landing_page".into() } else { "web_document".into() },
        identifier: None,
    }
}

fn extract_arxiv_id(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    for marker in ["arxiv.org/abs/", "arxiv.org/pdf/"] {
        if let Some(pos) = lower.find(marker) {
            let rest = &url[pos + marker.len()..];
            let id = rest.trim_end_matches(".pdf").split(['?', '#', '/']).next().unwrap_or("").trim();
            if !id.is_empty() { return Some(id.to_string()); }
        }
    }
    None
}

fn extract_pubmed_id(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    let pos = lower.find("pubmed.ncbi.nlm.nih.gov/")?;
    let rest = &url[pos + "pubmed.ncbi.nlm.nih.gov/".len()..];
    let id = rest.split(['?', '#', '/']).next().unwrap_or("").trim();
    (!id.is_empty() && id.chars().all(|ch| ch.is_ascii_digit())).then(|| id.to_string())
}

fn extract_doi_from_url(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    let pos = lower.find("doi.org/")?;
    let rest = &url[pos + "doi.org/".len()..];
    let doi = rest.split(['?', '#']).next().unwrap_or("").trim().trim_matches('/');
    (doi.starts_with("10.") && doi.len() > 6).then(|| doi.to_string())
}
