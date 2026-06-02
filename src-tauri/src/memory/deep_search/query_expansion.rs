//! Deterministic query expansion for autonomous research loops.

use super::DeepSearchDocument;
use super::types::DeepSearchCoverageReport;
use std::collections::{HashSet, VecDeque};

pub(crate) fn expand_queries(
    topic: &str,
    current_query: &str,
    coverage: &DeepSearchCoverageReport,
    documents: &[DeepSearchDocument],
    already_used: &[String],
    max_queries: usize,
) -> Vec<String> {
    let mut output = Vec::<String>::new();
    let used = already_used.iter().map(|q| normalize_query(q)).collect::<HashSet<_>>();
    let mut candidates = VecDeque::<String>::new();

    for gap in &coverage.missing_subtopics {
        match gap.as_str() {
            "academic_sources" => {
                candidates.push_back(format!("{topic} research paper systematic review"));
                candidates.push_back(format!("{topic} arxiv study benchmark"));
            }
            "independent_domains" => {
                candidates.push_back(format!("{topic} independent analysis evidence"));
                candidates.push_back(format!("{topic} implementation case study"));
            }
            "authoritative_sources" => {
                candidates.push_back(format!("{topic} official documentation guide"));
                candidates.push_back(format!("{topic} technical report"));
            }
            "topic_specific_evidence" => {
                candidates.push_back(format!("{current_query} evidence limitations"));
                candidates.push_back(format!("{topic} advantages risks limitations"));
            }
            _ => {}
        }
    }

    for term in salient_terms(documents).into_iter().take(8) {
        candidates.push_back(format!("{topic} {term}"));
    }

    while let Some(candidate) = candidates.pop_front() {
        let normalized = normalize_query(&candidate);
        if normalized.is_empty() || used.contains(&normalized) || output.iter().any(|q| normalize_query(q) == normalized) {
            continue;
        }
        output.push(candidate.chars().take(220).collect());
        if output.len() >= max_queries { break; }
    }
    output
}

fn salient_terms(documents: &[DeepSearchDocument]) -> Vec<String> {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let stop = ["about", "after", "again", "against", "because", "between", "could", "first", "their", "there", "these", "those", "through", "using", "where", "which", "would", "should", "research", "study"];
    for document in documents.iter().rev().take(4) {
        for token in document.text.split(|ch: char| !ch.is_alphanumeric()) {
            let token = token.trim().to_ascii_lowercase();
            if token.chars().count() < 7 || stop.contains(&token.as_str()) { continue; }
            *counts.entry(token).or_insert(0) += 1;
        }
    }
    let mut values = counts.into_iter().collect::<Vec<_>>();
    values.sort_by(|a, b| b.1.cmp(&a.1));
    values.into_iter().map(|(term, _)| term).take(16).collect()
}


pub(crate) fn initial_research_queries(topic: &str, raw_query: &str, max_queries: usize) -> Vec<String> {
    let compact_topic = compact_research_topic(topic, raw_query);
    let mut candidates = Vec::<String>::new();

    if !compact_topic.is_empty() {
        candidates.push(compact_topic.clone());
        candidates.push(format!("{compact_topic} survey paper architecture"));
        candidates.push(format!("{compact_topic} LLM agents long term memory RAG episodic semantic procedural memory"));
    }

    let raw_lower = raw_query.to_ascii_lowercase();
    if raw_lower.contains("rag") || raw_lower.contains("retrieval") {
        candidates.push("retrieval augmented generation long term memory LLM agents evaluation".into());
    }
    if raw_lower.contains("episodic") || raw_lower.contains("semant") || raw_lower.contains("procedural") || raw_lower.contains("procedur") {
        candidates.push("episodic semantic procedural memory LLM agents architecture".into());
    }
    if raw_lower.contains("reflection") || raw_lower.contains("riflession") {
        candidates.push("reflection loop memory consolidation LLM agents paper".into());
    }
    if raw_lower.contains("knowledge graph") || raw_lower.contains("grafo") {
        candidates.push("knowledge graph memory LLM agents retrieval augmentation".into());
    }
    if raw_lower.contains("claim") || raw_lower.contains("verifica") || raw_lower.contains("verification") || raw_lower.contains("contamin") {
        candidates.push("LLM agent memory claim verification contamination mitigation".into());
    }
    if raw_lower.contains("retrieval evaluation") || raw_lower.contains("valutazione") {
        candidates.push("RAG retrieval evaluation metrics LLM agents memory benchmark".into());
    }

    // The user's full instruction can be very long and conversational. Keep one compact,
    // search-engine friendly query derived from the salient tokens, but never use the raw
    // prompt verbatim as the only discovery query.
    let salient = salient_query_from_instruction(raw_query);
    if !salient.is_empty() {
        candidates.push(salient);
    }

    let mut output = Vec::<String>::new();
    let mut seen = HashSet::<String>::new();
    for candidate in candidates {
        let cleaned = sanitize_search_query(&candidate);
        let key = normalize_query(&cleaned);
        if cleaned.is_empty() || !seen.insert(key) {
            continue;
        }
        output.push(cleaned.chars().take(180).collect());
        if output.len() >= max_queries {
            break;
        }
    }
    output
}

fn compact_research_topic(topic: &str, raw_query: &str) -> String {
    let candidates = [topic, raw_query];
    for value in candidates {
        let lower = value.to_ascii_lowercase();
        for marker in ["argomento complesso:", "topic:", "argomento:", "su:"] {
            if let Some(pos) = lower.find(marker) {
                let after = &value[pos + marker.len()..];
                let sentence = after
                    .split(|ch: char| ch == '.' || ch == '\n' || ch == ';')
                    .next()
                    .unwrap_or(after)
                    .trim();
                let cleaned = sanitize_search_query(sentence);
                if cleaned.chars().count() >= 18 {
                    return cleaned;
                }
            }
        }
    }
    sanitize_search_query(topic)
}

fn salient_query_from_instruction(value: &str) -> String {
    let stop = [
        "voglio", "che", "faccia", "completa", "autonoma", "argomento", "complesso", "studia",
        "cerca", "fonti", "web", "documentazione", "tecnica", "paper", "articoli", "affidabili",
        "limitarti", "sola", "fonte", "confronta", "identifica", "eventuali", "limiti", "alla", "fine",
        "dammi", "sintesi", "cosa", "imparato", "quali", "sono", "state", "parti", "restano",
        "incerte", "approfondire", "the", "and", "for", "with", "from", "into", "using", "about",
        "complete", "autonomous", "search", "sources", "technical", "documentation", "reliable",
    ];
    let mut tokens = Vec::<String>::new();
    for token in value.split(|ch: char| !ch.is_alphanumeric() && ch != '-') {
        let token = token.trim().to_ascii_lowercase();
        if token.chars().count() < 3 || stop.contains(&token.as_str()) {
            continue;
        }
        if !tokens.iter().any(|existing| existing == &token) {
            tokens.push(token);
        }
        if tokens.len() >= 18 {
            break;
        }
    }
    sanitize_search_query(&tokens.join(" "))
}

fn sanitize_search_query(value: &str) -> String {
    value
        .split_whitespace()
        .map(|part| part.trim_matches(|ch: char| matches!(ch, ',' | ';' | ':' | '.' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}')))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_query(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ").to_ascii_lowercase()
}
