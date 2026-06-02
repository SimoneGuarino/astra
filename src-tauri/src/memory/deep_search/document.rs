//! Document and academic landing-page ingestion for AstraOS Deep Search.
//!
//! This module is intentionally conservative: it extracts text from bounded
//! HTML/plain/XML/JSON documents, known academic landing pages and, in v0.6.5,
//! unencrypted PDF files through a bounded stream-text extractor. Raw PDF
//! binaries are never persisted into the Memory Graph. If PDF extraction is not
//! useful enough, the source is rejected explicitly instead of pretending Astra
//! read the paper.

use super::{pdf::extract_pdf_text, types::DeepSearchRequest, DeepSearchDocument, DeepSearchDocumentSection};
use crate::memory::{errors::{MemoryError, MemoryResult}, types::now_ms};
use reqwest::blocking::Client;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::time::Duration;

const MAX_TEXT_CHARS_PER_SOURCE: usize = 96_000;
const DOCUMENT_USER_AGENT: &str = "AstraOS-DeepSearchDocumentIngestion/0.6.5 (+local-governed-memory)";

pub(crate) fn fetch_and_normalize_document(
    client: &Client,
    url: &str,
    request: &DeepSearchRequest,
) -> MemoryResult<DeepSearchDocument> {
    let pdf_extraction_enabled = request.enable_pdf_text_extraction.unwrap_or(false);
    let prefer_landing = request.prefer_academic_landing_pages.unwrap_or(true) && !pdf_extraction_enabled;
    let bounded_url = if prefer_landing {
        normalize_known_academic_document_url(url)
    } else {
        url.trim().to_string()
    };
    let response = client
        .get(&bounded_url)
        .header(reqwest::header::USER_AGENT, DOCUMENT_USER_AGENT)
        .timeout(Duration::from_millis(request.timeout_ms.unwrap_or(90_000).clamp(10_000, 240_000)))
        .send()
        .map_err(|error| MemoryError::Storage(format!("document fetch failed: {error}")))?;

    let final_url = response.url().to_string();
    let status = response.status();
    if !status.is_success() {
        return Err(MemoryError::Validation(format!("document returned non-success status {status}")));
    }

    let headers = response.headers().clone();
    let content_type = headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let bytes = response
        .bytes()
        .map_err(|error| MemoryError::Storage(format!("document body read failed: {error}")))?;
    let max_bytes = request.max_bytes_per_source.unwrap_or(768_000).clamp(16_000, 3_000_000);
    if bytes.len() > max_bytes {
        return Err(MemoryError::Validation(format!("document exceeded max_bytes_per_source ({max_bytes})")));
    }

    if is_pdf_payload(&content_type, bytes.as_ref()) {
        if pdf_extraction_enabled {
            let extracted = extract_pdf_text(bytes.as_ref(), MAX_TEXT_CHARS_PER_SOURCE)?;
            let normalized = redact_sensitive_text_basic(&extracted.text);
            let sections = sectionize_text(&normalized, true);
            let section_count = sections.len().max(extracted.section_count);
            let title = title_from_url(&final_url).unwrap_or_else(|| final_url.clone());
            return Ok(DeepSearchDocument {
                url: final_url,
                title: cap_chars(title, 220),
                text: cap_chars(normalized, MAX_TEXT_CHARS_PER_SOURCE),
                content_hash: extracted.content_hash,
                fetched_at: now_ms(),
                content_type: if content_type.is_empty() { Some("application/pdf".into()) } else { Some(content_type) },
                discovered_by: None,
                source_type: None,
                discovery_rank: None,
                document_kind: Some("pdf_full_text".into()),
                doi: extract_doi_from_url(&bounded_url),
                academic_id: extract_academic_id(&bounded_url),
                published_at: None,
                abstract_present: false,
                section_count,
                sections,
                pdf_extracted: true,
                extraction_method: Some(extracted.method),
                reliability: Default::default(),
                metadata: json!({
                    "schema_version": 2,
                    "phase": "v0.6_5_pdf_full_text_section_chunking",
                    "source_url_before_known_pdf_landing_normalization": url,
                    "known_pdf_landing_normalized_url": bounded_url,
                    "bounded_pdf_text_extraction": true,
                    "raw_pdf_persistence": false,
                    "external_untrusted": true,
                    "page_count_hint": extracted.page_count_hint,
                    "pdf_warnings": extracted.warnings,
                    "section_aware_chunking": true,
                }),
            });
        }
        if let Some(landing_url) = pdf_url_to_landing_page(&final_url) {
            if landing_url != final_url {
                return fetch_and_normalize_document(client, &landing_url, request);
            }
        }
        return Err(MemoryError::Validation(
            "PDF source detected but full-text PDF extraction is disabled; enable_pdf_text_extraction is required for direct PDF ingestion".into(),
        ));
    }

    if !is_supported_textual_content_type(&content_type) {
        return Err(MemoryError::Validation(format!(
            "unsupported content-type for document ingestion: {}",
            if content_type.is_empty() { "unknown" } else { content_type.as_str() }
        )));
    }

    let raw = String::from_utf8_lossy(bytes.as_ref()).to_string();
    let title = extract_title(&raw).unwrap_or_else(|| extract_heading(&raw).unwrap_or_else(|| final_url.clone()));
    let abstract_text = extract_abstract(&raw);
    let doi = extract_doi(&raw).or_else(|| extract_doi_from_url(&final_url));
    let academic_id = extract_academic_id(&final_url);
    let published_at = extract_published_at(&raw);
    let document_kind = classify_document_kind(&final_url, &content_type, &raw);
    let section_count = count_probable_sections(&raw);
    let normalized_body = match document_kind.as_str() {
        "json_document" => json_to_text(&raw),
        "xml_document" | "academic_feed" => xml_to_text(&raw),
        _ => html_to_text(&raw),
    };
    let mut normalized = collapse_ws(&redact_sensitive_text_basic(&normalized_body));
    if let Some(abstract_text) = abstract_text.as_ref() {
        let abstract_text = collapse_ws(&html_to_text(abstract_text));
        if abstract_text.chars().count() >= 80 && !normalized.to_ascii_lowercase().contains(&abstract_text.to_ascii_lowercase()) {
            normalized = format!("Abstract: {abstract_text}\n\n{normalized}");
        }
    }
    let normalized = cap_chars(normalized, MAX_TEXT_CHARS_PER_SOURCE);
    let sections = sectionize_text(&normalized, false);
    if normalized.chars().count() < 160 {
        return Err(MemoryError::Validation("document text is too short after normalization".into()));
    }

    let content_hash = sha256_hex(&normalized);
    Ok(DeepSearchDocument {
        url: final_url,
        title: cap_chars(collapse_ws(&html_to_text(&title)), 220),
        text: normalized,
        content_hash,
        fetched_at: now_ms(),
        content_type: if content_type.is_empty() { None } else { Some(content_type) },
        discovered_by: None,
        source_type: None,
        discovery_rank: None,
        document_kind: Some(document_kind),
        doi,
        academic_id,
        published_at,
        abstract_present: abstract_text.as_ref().is_some_and(|value| collapse_ws(value).chars().count() >= 80),
        section_count: section_count.max(sections.len()),
        sections,
        pdf_extracted: false,
        extraction_method: Some("bounded_document_text_normalizer_v0_6_5".into()),
        reliability: Default::default(),
        metadata: json!({
            "schema_version": 1,
            "phase": "v0.6_5_pdf_full_text_section_chunking",
            "source_url_before_known_pdf_landing_normalization": url,
            "known_pdf_landing_normalized_url": bounded_url,
            "bounded_text_extraction": true,
            "raw_pdf_persistence": false,
            "external_untrusted": true,
        }),
    })
}

pub(crate) fn normalize_known_academic_document_url(value: &str) -> String {
    let trimmed = value.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.contains("arxiv.org/pdf/") {
        return trimmed
            .replace("http://arxiv.org/pdf/", "https://arxiv.org/abs/")
            .replace("https://arxiv.org/pdf/", "https://arxiv.org/abs/")
            .trim_end_matches(".pdf")
            .to_string();
    }
    if lower.ends_with(".pdf") && lower.contains("arxiv.org") {
        return trimmed.trim_end_matches(".pdf").replace("/pdf/", "/abs/");
    }
    trimmed.to_string()
}

fn is_pdf_payload(content_type: &str, bytes: &[u8]) -> bool {
    content_type.contains("application/pdf") || bytes.starts_with(b"%PDF")
}

fn pdf_url_to_landing_page(url: &str) -> Option<String> {
    let normalized = normalize_known_academic_document_url(url);
    (normalized != url).then_some(normalized)
}

fn is_supported_textual_content_type(content_type: &str) -> bool {
    content_type.is_empty()
        || content_type.contains("text/html")
        || content_type.contains("text/plain")
        || content_type.contains("application/xhtml")
        || content_type.contains("application/xml")
        || content_type.contains("text/xml")
        || content_type.contains("application/atom+xml")
        || content_type.contains("application/rss+xml")
        || content_type.contains("application/json")
        || content_type.contains("application/ld+json")
}

fn classify_document_kind(url: &str, content_type: &str, body: &str) -> String {
    let lower_url = url.to_ascii_lowercase();
    let lower_body = body.to_ascii_lowercase();
    if content_type.contains("json") { return "json_document".into(); }
    if content_type.contains("xml") || content_type.contains("atom") || content_type.contains("rss") { return "xml_document".into(); }
    if lower_url.contains("arxiv.org/abs/") { return "academic_preprint_abstract".into(); }
    if lower_url.contains("pubmed.ncbi.nlm.nih.gov") { return "academic_biomedical_abstract".into(); }
    if lower_url.contains("doi.org/") { return "academic_doi_landing_page".into(); }
    if lower_body.contains("citation_doi") || lower_body.contains("dc.identifier") || lower_body.contains("scholarlyarticle") { return "academic_landing_page".into(); }
    "web_document".into()
}

fn extract_title(input: &str) -> Option<String> {
    extract_between_case_insensitive(input, "<title", "</title>")
        .and_then(|value| value.find('>').map(|index| value[index + 1..].to_string()))
        .map(|value| collapse_ws(&decode_basic_entities(&value)))
        .filter(|value| !value.trim().is_empty())
        .or_else(|| extract_meta_content(input, "citation_title"))
        .or_else(|| extract_meta_content(input, "dc.title"))
        .or_else(|| extract_meta_content(input, "og:title"))
}

fn extract_heading(input: &str) -> Option<String> {
    extract_between_case_insensitive(input, "<h1", "</h1>")
        .and_then(|value| value.find('>').map(|index| value[index + 1..].to_string()))
        .map(|value| collapse_ws(&html_to_text(&value)))
        .filter(|value| !value.trim().is_empty())
}

fn extract_abstract(input: &str) -> Option<String> {
    extract_meta_content(input, "citation_abstract")
        .or_else(|| extract_meta_content(input, "dc.description"))
        .or_else(|| extract_jsonish_value(input, "abstract"))
        .or_else(|| extract_between_case_insensitive(input, "abstract", "</section>").map(|value| cap_chars(html_to_text(&value), 2_400)))
}

fn extract_published_at(input: &str) -> Option<String> {
    extract_meta_content(input, "citation_publication_date")
        .or_else(|| extract_meta_content(input, "citation_date"))
        .or_else(|| extract_meta_content(input, "dc.date"))
        .or_else(|| extract_meta_content(input, "article:published_time"))
        .map(|value| cap_chars(value, 64))
}

fn extract_doi(input: &str) -> Option<String> {
    extract_meta_content(input, "citation_doi")
        .or_else(|| extract_meta_content(input, "dc.identifier"))
        .and_then(|value| normalize_doi(&value))
        .or_else(|| extract_doi_from_text(input))
}

fn extract_doi_from_url(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    lower.find("doi.org/").and_then(|pos| normalize_doi(&url[pos + "doi.org/".len()..]))
}

fn extract_doi_from_text(input: &str) -> Option<String> {
    let lower = input.to_ascii_lowercase();
    let pos = lower.find("10.")?;
    let slice = &input[pos..input.len().min(pos + 160)];
    let doi = slice
        .split(|ch: char| ch.is_whitespace() || matches!(ch, '<' | '>' | '"' | '\'' | ')' | '(' | ','))
        .next()
        .unwrap_or("");
    normalize_doi(doi)
}

fn normalize_doi(value: &str) -> Option<String> {
    let trimmed = value
        .trim()
        .trim_start_matches("doi:")
        .trim_start_matches("https://doi.org/")
        .trim_start_matches("http://doi.org/")
        .trim_matches(|ch: char| matches!(ch, '.' | ',' | ';' | ')' | '(' | '"' | '\''));
    (trimmed.starts_with("10.") && trimmed.chars().count() >= 6).then(|| trimmed.to_string())
}

fn extract_academic_id(url: &str) -> Option<String> {
    let lower = url.to_ascii_lowercase();
    if let Some(pos) = lower.find("arxiv.org/abs/") {
        return Some(format!("arxiv:{}", url[pos + "arxiv.org/abs/".len()..].trim_matches('/')));
    }
    if let Some(pos) = lower.find("pubmed.ncbi.nlm.nih.gov/") {
        let rest = &url[pos + "pubmed.ncbi.nlm.nih.gov/".len()..];
        let pmid = rest.split('/').next().unwrap_or("").trim();
        if !pmid.is_empty() { return Some(format!("pmid:{pmid}")); }
    }
    extract_doi_from_url(url).map(|doi| format!("doi:{doi}"))
}

fn extract_meta_content(input: &str, name: &str) -> Option<String> {
    let lower = input.to_ascii_lowercase();
    let needle_name = format!("name=\"{}\"", name.to_ascii_lowercase());
    let needle_prop = format!("property=\"{}\"", name.to_ascii_lowercase());
    let mut start = 0usize;
    while let Some(pos) = lower[start..].find("<meta") {
        let tag_start = start + pos;
        let tag_end = input[tag_start..].find('>').map(|rel| tag_start + rel).unwrap_or(input.len());
        let tag = &input[tag_start..tag_end];
        let tag_lower = tag.to_ascii_lowercase();
        if tag_lower.contains(&needle_name) || tag_lower.contains(&needle_prop) {
            if let Some(content) = extract_attr(tag, "content") {
                return Some(collapse_ws(&decode_basic_entities(&content)));
            }
        }
        start = tag_end.saturating_add(1);
    }
    None
}

fn extract_attr(tag: &str, attr: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let needle = format!("{attr}=");
    let pos = lower.find(&needle)? + needle.len();
    let rest = tag[pos..].trim_start();
    let first = rest.chars().next()?;
    if first == '"' || first == '\'' {
        let end = rest[1..].find(first)? + 1;
        return Some(rest[1..end].to_string());
    }
    let end = rest.find(|ch: char| ch.is_whitespace()).unwrap_or(rest.len());
    Some(rest[..end].to_string())
}

fn extract_jsonish_value(input: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let pos = input.find(&needle)?;
    let after = &input[pos + needle.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    if !rest.starts_with('"') { return None; }
    let mut value = String::new();
    let mut escaped = false;
    for ch in rest[1..].chars() {
        if escaped {
            value.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            break;
        } else {
            value.push(ch);
        }
    }
    (!value.trim().is_empty()).then(|| collapse_ws(&value))
}

fn count_probable_sections(input: &str) -> usize {
    let lower = input.to_ascii_lowercase();
    let html_sections = lower.matches("<section").count() + lower.matches("<h2").count() + lower.matches("<h3").count();
    if html_sections > 0 { return html_sections.min(64); }
    input.lines().filter(|line| {
        let trimmed = line.trim();
        trimmed.chars().count() <= 80 && trimmed.chars().count() >= 4 && trimmed.chars().any(|ch| ch.is_alphabetic())
    }).count().min(64)
}

fn extract_between_case_insensitive(input: &str, start_needle: &str, end_needle: &str) -> Option<String> {
    let lower = input.to_ascii_lowercase();
    let start = lower.find(&start_needle.to_ascii_lowercase())?;
    let end = lower[start..].find(&end_needle.to_ascii_lowercase()).map(|rel| start + rel + end_needle.len())?;
    Some(input[start..end].to_string())
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

fn xml_to_text(input: &str) -> String {
    html_to_text(input)
}

fn json_to_text(input: &str) -> String {
    serde_json::from_str::<serde_json::Value>(input)
        .map(|value| flatten_json_value(&value, 0))
        .unwrap_or_else(|_| input.to_string())
}

fn flatten_json_value(value: &serde_json::Value, depth: usize) -> String {
    if depth > 5 { return String::new(); }
    match value {
        serde_json::Value::String(value) => value.clone(),
        serde_json::Value::Array(values) => values.iter().map(|value| flatten_json_value(value, depth + 1)).collect::<Vec<_>>().join("\n"),
        serde_json::Value::Object(map) => map.iter().map(|(key, value)| format!("{key}: {}", flatten_json_value(value, depth + 1))).collect::<Vec<_>>().join("\n"),
        other => other.to_string(),
    }
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

fn redact_sensitive_text_basic(input: &str) -> String {
    input
        .lines()
        .map(|line| {
            let lower = line.to_ascii_lowercase();
            if lower.contains("api_key")
                || lower.contains("apikey")
                || lower.contains("authorization:")
                || lower.contains("bearer ")
                || lower.contains("password")
                || lower.contains("secret")
                || lower.contains("private_key")
            {
                "[REDACTED_SENSITIVE_LINE]".to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
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


fn title_from_url(url: &str) -> Option<String> {
    let without_query = url.split(['?', '#']).next().unwrap_or(url).trim_end_matches('/');
    let slug = without_query.rsplit('/').next()?.trim();
    if slug.is_empty() { return None; }
    Some(slug.trim_end_matches(".pdf").replace('_', " ").replace('-', " "))
}

pub(crate) fn sectionize_text(text: &str, from_pdf: bool) -> Vec<DeepSearchDocumentSection> {
    let mut sections = Vec::<DeepSearchDocumentSection>::new();
    let mut current_title = if from_pdf { "PDF extracted text".to_string() } else { "Document body".to_string() };
    let mut current = String::new();
    let mut ordinal = 1usize;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        if is_probable_section_heading(trimmed) && current.chars().count() >= 240 {
            push_section(&mut sections, ordinal, &current_title, &current);
            ordinal += 1;
            current_title = cap_chars(trimmed.to_string(), 160);
            current.clear();
            continue;
        }
        current.push_str(trimmed);
        current.push('\n');
        if current.chars().count() >= 8_000 {
            push_section(&mut sections, ordinal, &current_title, &current);
            ordinal += 1;
            current_title = format!("{} continuation", current_title);
            current.clear();
        }
        if sections.len() >= 64 { break; }
    }
    if current.chars().count() >= 120 && sections.len() < 64 {
        push_section(&mut sections, ordinal, &current_title, &current);
    }
    if sections.is_empty() && text.chars().count() >= 120 {
        push_section(&mut sections, 1, &current_title, text);
    }
    sections
}

fn push_section(sections: &mut Vec<DeepSearchDocumentSection>, ordinal: usize, title: &str, text: &str) {
    let normalized = collapse_ws(text);
    if normalized.chars().count() < 120 { return; }
    sections.push(DeepSearchDocumentSection {
        ordinal,
        title: cap_chars(title.to_string(), 160),
        text: cap_chars(normalized, 12_000),
        char_count: text.chars().count(),
    });
}

fn is_probable_section_heading(line: &str) -> bool {
    let len = line.chars().count();
    if !(3..=96).contains(&len) { return false; }
    let lower = line.to_ascii_lowercase();
    let canonical = [
        "abstract", "introduction", "background", "related work", "method", "methods",
        "methodology", "results", "evaluation", "discussion", "limitations", "conclusion",
        "conclusions", "references", "appendix", "materials and methods",
    ];
    if canonical.iter().any(|heading| lower == *heading || lower.starts_with(format!("{heading} ").as_str())) {
        return true;
    }
    let alpha = line.chars().filter(|ch| ch.is_alphabetic()).count();
    let uppercase = line.chars().filter(|ch| ch.is_alphabetic() && ch.is_uppercase()).count();
    alpha >= 3 && uppercase * 100 / alpha.max(1) >= 70
}
