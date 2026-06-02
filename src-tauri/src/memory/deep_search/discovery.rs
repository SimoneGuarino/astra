//! Multi-provider source discovery for AstraOS Deep Search.
//!
//! Discovery is intentionally bounded and provider-agnostic. It does not persist
//! anything and it does not treat search-result snippets as truth. It only
//! proposes candidate URLs. The caller still applies Rust policy, fetch limits,
//! normalization, evidence extraction and Memory Graph consolidation.

use super::types::{DeepSearchPolicy, DeepSearchRequest};
use crate::memory::{errors::{MemoryError, MemoryResult}, types::now_ms};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;

const DEFAULT_DISCOVERY_LIMIT_PER_PROVIDER: usize = 8;
const DISCOVERY_MAX_BODY_BYTES: usize = 1_500_000;
const DISCOVERY_USER_AGENT: &str = "AstraOS-DeepSearchDiscovery/0.6.3 (+local-governed-memory)";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchDiscoveredSource {
    pub url: String,
    pub provider: String,
    pub source_type: String,
    pub title: Option<String>,
    pub rank: usize,
    pub discovered_at: i64,
}

pub(crate) fn discover_sources(
    client: &Client,
    request: &DeepSearchRequest,
    _policy: &DeepSearchPolicy,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    if !request.enable_web_discovery.unwrap_or(true) {
        return Vec::new();
    }

    let query = request.query.clone().unwrap_or_else(|| request.topic.clone());
    let query = query.trim();
    if query.is_empty() {
        warnings.push("deep-search discovery skipped because query/topic is empty".into());
        return Vec::new();
    }

    let encoded_query = url_encode(query);
    let per_provider_limit = request
        .max_discovery_results_per_provider
        .unwrap_or(DEFAULT_DISCOVERY_LIMIT_PER_PROVIDER)
        .clamp(1, 16);
    let provider_filter = request
        .search_providers
        .iter()
        .map(|provider| provider.trim().to_ascii_lowercase())
        .filter(|provider| !provider.is_empty())
        .collect::<HashSet<_>>();

    let include_general_web = request.include_general_web.unwrap_or(true);
    let include_academic_sources = request.include_academic_sources.unwrap_or(true);
    let mut discovered = Vec::<DeepSearchDiscoveredSource>::new();

    if let Some(template) = std::env::var("ASTRA_DEEP_SEARCH_SEARCH_URL_TEMPLATE").ok().filter(|value| !value.trim().is_empty()) {
        if template.contains("{query}") {
            push_provider_results(
                &mut discovered,
                fetch_search_html_candidates(client, &template.replace("{query}", &encoded_query), "custom_template", "general_web", per_provider_limit, warnings),
            );
        } else {
            warnings.push("ASTRA_DEEP_SEARCH_SEARCH_URL_TEMPLATE ignored because it does not contain {query}".into());
        }
    }

    if include_general_web {
        if provider_enabled(&provider_filter, "duckduckgo_html") {
            push_provider_results(
                &mut discovered,
                fetch_search_html_candidates(
                    client,
                    &format!("https://duckduckgo.com/html/?q={encoded_query}"),
                    "duckduckgo_html",
                    "general_web",
                    per_provider_limit,
                    warnings,
                ),
            );
        }
        if provider_enabled(&provider_filter, "duckduckgo_lite") {
            push_provider_results(
                &mut discovered,
                fetch_search_html_candidates(
                    client,
                    &format!("https://lite.duckduckgo.com/lite/?q={encoded_query}"),
                    "duckduckgo_lite",
                    "general_web",
                    per_provider_limit,
                    warnings,
                ),
            );
        }
        if provider_enabled(&provider_filter, "bing") {
            push_provider_results(
                &mut discovered,
                fetch_search_html_candidates(
                    client,
                    &format!("https://www.bing.com/search?q={encoded_query}"),
                    "bing",
                    "general_web",
                    per_provider_limit,
                    warnings,
                ),
            );
        }
        if provider_enabled(&provider_filter, "wikipedia") {
            push_provider_results(
                &mut discovered,
                fetch_wikipedia_candidates(client, &encoded_query, per_provider_limit, warnings),
            );
        }
    }

    if include_academic_sources {
        if provider_enabled(&provider_filter, "arxiv") {
            push_provider_results(
                &mut discovered,
                fetch_arxiv_candidates(client, &encoded_query, per_provider_limit, warnings),
            );
        }
        if provider_enabled(&provider_filter, "crossref") {
            push_provider_results(
                &mut discovered,
                fetch_crossref_candidates(client, &encoded_query, per_provider_limit, warnings),
            );
        }
        if provider_enabled(&provider_filter, "pubmed") {
            push_provider_results(
                &mut discovered,
                fetch_pubmed_candidates(client, &encoded_query, per_provider_limit, warnings),
            );
        }
        if provider_enabled(&provider_filter, "semantic_scholar") {
            push_provider_results(
                &mut discovered,
                fetch_semantic_scholar_candidates(client, &encoded_query, per_provider_limit, warnings),
            );
        }
        if provider_enabled(&provider_filter, "europe_pmc") {
            push_provider_results(
                &mut discovered,
                fetch_europe_pmc_candidates(client, &encoded_query, per_provider_limit, warnings),
            );
        }
    }

    let mut seen = HashSet::<String>::new();
    discovered
        .into_iter()
        .filter_map(|mut source| {
            source.url = canonicalize_candidate_url(&source.url);
            if source.url.is_empty() || !is_probably_fetchable_document(&source.url) {
                return None;
            }
            let key = source.url.to_ascii_lowercase();
            if !seen.insert(key) {
                return None;
            }
            Some(source)
        })
        .take(request.max_discovered_sources.unwrap_or(96).clamp(8, 256))
        .collect()
}

fn provider_enabled(filter: &HashSet<String>, name: &str) -> bool {
    filter.is_empty() || filter.contains(name)
}

fn push_provider_results(target: &mut Vec<DeepSearchDiscoveredSource>, values: Vec<DeepSearchDiscoveredSource>) {
    target.extend(values);
}

fn fetch_search_html_candidates(
    client: &Client,
    url: &str,
    provider: &str,
    source_type: &str,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    match fetch_text(client, url) {
        Ok(body) => extract_links_from_search_html(&body, provider, source_type, limit),
        Err(error) => {
            warnings.push(format!("deep-search provider {provider} discovery failed: {error}"));
            Vec::new()
        }
    }
}

fn fetch_wikipedia_candidates(
    client: &Client,
    encoded_query: &str,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    let url = format!("https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded_query}&limit={limit}&namespace=0&format=json");
    let Ok(body) = fetch_text(client, &url) else {
        warnings.push("deep-search provider wikipedia discovery failed".into());
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&body) else {
        warnings.push("deep-search provider wikipedia returned non-json payload".into());
        return Vec::new();
    };
    let titles = value.get(1).and_then(Value::as_array).cloned().unwrap_or_default();
    let urls = value.get(3).and_then(Value::as_array).cloned().unwrap_or_default();
    urls.into_iter()
        .enumerate()
        .filter_map(|(index, value)| {
            Some(DeepSearchDiscoveredSource {
                url: value.as_str()?.to_string(),
                provider: "wikipedia".into(),
                source_type: "encyclopedic_web".into(),
                title: titles.get(index).and_then(Value::as_str).map(str::to_string),
                rank: index + 1,
                discovered_at: now_ms(),
            })
        })
        .take(limit)
        .collect()
}

fn fetch_arxiv_candidates(
    client: &Client,
    encoded_query: &str,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    let url = format!("https://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={limit}");
    let Ok(body) = fetch_text(client, &url) else {
        warnings.push("deep-search provider arxiv discovery failed".into());
        return Vec::new();
    };
    body.split("<entry>")
        .skip(1)
        .enumerate()
        .filter_map(|(index, entry)| {
            let id = extract_xml_text(entry, "id")?;
            let title = extract_xml_text(entry, "title").map(|value| collapse_ws(&value));
            Some(DeepSearchDiscoveredSource {
                url: id.replace("http://", "https://"),
                provider: "arxiv".into(),
                source_type: "academic_preprint".into(),
                title,
                rank: index + 1,
                discovered_at: now_ms(),
            })
        })
        .take(limit)
        .collect()
}

fn fetch_crossref_candidates(
    client: &Client,
    encoded_query: &str,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    let url = format!("https://api.crossref.org/works?query={encoded_query}&rows={limit}");
    let Ok(body) = fetch_text(client, &url) else {
        warnings.push("deep-search provider crossref discovery failed".into());
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&body) else {
        warnings.push("deep-search provider crossref returned non-json payload".into());
        return Vec::new();
    };
    value.pointer("/message/items")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, item)| {
            let url = item
                .get("URL")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| item.get("DOI").and_then(Value::as_str).map(|doi| format!("https://doi.org/{doi}")))?;
            let title = item
                .get("title")
                .and_then(Value::as_array)
                .and_then(|titles| titles.first())
                .and_then(Value::as_str)
                .map(str::to_string);
            Some(DeepSearchDiscoveredSource {
                url,
                provider: "crossref".into(),
                source_type: "academic_index".into(),
                title,
                rank: index + 1,
                discovered_at: now_ms(),
            })
        })
        .take(limit)
        .collect()
}

fn fetch_pubmed_candidates(
    client: &Client,
    encoded_query: &str,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    let url = format!("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encoded_query}&retmode=json&retmax={limit}");
    let Ok(body) = fetch_text(client, &url) else {
        warnings.push("deep-search provider pubmed discovery failed".into());
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&body) else {
        warnings.push("deep-search provider pubmed returned non-json payload".into());
        return Vec::new();
    };
    value.pointer("/esearchresult/idlist")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, item)| {
            let pmid = item.as_str()?;
            Some(DeepSearchDiscoveredSource {
                url: format!("https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                provider: "pubmed".into(),
                source_type: "academic_biomedical".into(),
                title: Some(format!("PubMed record {pmid}")),
                rank: index + 1,
                discovered_at: now_ms(),
            })
        })
        .take(limit)
        .collect()
}

fn fetch_semantic_scholar_candidates(
    client: &Client,
    encoded_query: &str,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    let url = format!("https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={limit}&fields=title,url,externalIds");
    let Ok(body) = fetch_text(client, &url) else {
        warnings.push("deep-search provider semantic_scholar discovery failed".into());
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&body) else {
        warnings.push("deep-search provider semantic_scholar returned non-json payload".into());
        return Vec::new();
    };
    value.get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, item)| {
            let url = item
                .get("url")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| item.pointer("/externalIds/DOI").and_then(Value::as_str).map(|doi| format!("https://doi.org/{doi}")))?;
            Some(DeepSearchDiscoveredSource {
                url,
                provider: "semantic_scholar".into(),
                source_type: "academic_index".into(),
                title: item.get("title").and_then(Value::as_str).map(str::to_string),
                rank: index + 1,
                discovered_at: now_ms(),
            })
        })
        .take(limit)
        .collect()
}

fn fetch_europe_pmc_candidates(
    client: &Client,
    encoded_query: &str,
    limit: usize,
    warnings: &mut Vec<String>,
) -> Vec<DeepSearchDiscoveredSource> {
    let url = format!("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={encoded_query}&format=json&pageSize={limit}");
    let Ok(body) = fetch_text(client, &url) else {
        warnings.push("deep-search provider europe_pmc discovery failed".into());
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&body) else {
        warnings.push("deep-search provider europe_pmc returned non-json payload".into());
        return Vec::new();
    };
    value.pointer("/resultList/result")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, item)| {
            let url = item
                .get("doi")
                .and_then(Value::as_str)
                .map(|doi| format!("https://doi.org/{doi}"))
                .or_else(|| item.get("pmid").and_then(Value::as_str).map(|pmid| format!("https://pubmed.ncbi.nlm.nih.gov/{pmid}/")))?;
            Some(DeepSearchDiscoveredSource {
                url,
                provider: "europe_pmc".into(),
                source_type: "academic_biomedical".into(),
                title: item.get("title").and_then(Value::as_str).map(str::to_string),
                rank: index + 1,
                discovered_at: now_ms(),
            })
        })
        .take(limit)
        .collect()
}

fn extract_links_from_search_html(
    body: &str,
    provider: &str,
    source_type: &str,
    limit: usize,
) -> Vec<DeepSearchDiscoveredSource> {
    let mut urls = Vec::<String>::new();
    for href in extract_href_values(body) {
        let decoded = decode_basic_entities(&href);
        let unwrapped = unwrap_redirect_url(&decoded).unwrap_or(decoded);
        let candidate = canonicalize_candidate_url(&url_decode(&unwrapped));
        if is_probably_fetchable_document(&candidate) && !is_search_provider_url(&candidate) && !urls.iter().any(|existing| existing.eq_ignore_ascii_case(&candidate)) {
            urls.push(candidate);
        }
        if urls.len() >= limit {
            break;
        }
    }
    urls.into_iter()
        .enumerate()
        .map(|(index, url)| DeepSearchDiscoveredSource {
            url,
            provider: provider.into(),
            source_type: source_type.into(),
            title: None,
            rank: index + 1,
            discovered_at: now_ms(),
        })
        .collect()
}

fn fetch_text(client: &Client, url: &str) -> MemoryResult<String> {
    let response = client
        .get(url)
        .header(reqwest::header::USER_AGENT, DISCOVERY_USER_AGENT)
        .send()
        .map_err(|error| MemoryError::Storage(format!("discovery fetch failed: {error}")))?;
    if !response.status().is_success() {
        return Err(MemoryError::Validation(format!("discovery provider returned {}", response.status())));
    }
    let body = response
        .text()
        .map_err(|error| MemoryError::Storage(format!("discovery body read failed: {error}")))?;
    if body.len() > DISCOVERY_MAX_BODY_BYTES {
        return Err(MemoryError::Validation("discovery provider response exceeded bounded body limit".into()));
    }
    Ok(body)
}

fn extract_href_values(html: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut index = 0usize;
    let lower = html.to_ascii_lowercase();
    while let Some(relative) = lower[index..].find("href") {
        let href_pos = index + relative;
        let after_href = &html[href_pos + 4..];
        let Some(eq_rel) = after_href.find('=') else {
            index = href_pos + 4;
            continue;
        };
        let value_start = href_pos + 4 + eq_rel + 1;
        let rest = html[value_start..].trim_start();
        if rest.is_empty() {
            break;
        }
        let consumed_ws = html[value_start..].len() - rest.len();
        let actual_start = value_start + consumed_ws;
        let first = rest.chars().next().unwrap_or_default();
        if first == '"' || first == '\'' {
            if let Some(end) = rest[1..].find(first) {
                values.push(rest[1..1 + end].to_string());
                index = actual_start + end + 2;
                continue;
            }
        } else {
            let end = rest.find(|ch: char| ch.is_whitespace() || ch == '>').unwrap_or(rest.len());
            values.push(rest[..end].to_string());
            index = actual_start + end;
            continue;
        }
        index = href_pos + 4;
    }
    values
}

fn unwrap_redirect_url(value: &str) -> Option<String> {
    let value = value.trim();
    for key in ["uddg=", "u=", "url=", "q="] {
        if let Some(pos) = value.find(key) {
            let start = pos + key.len();
            let end = value[start..].find('&').map(|relative| start + relative).unwrap_or(value.len());
            let decoded = url_decode(&value[start..end]);
            if decoded.starts_with("http://") || decoded.starts_with("https://") {
                return Some(decoded);
            }
        }
    }
    if value.starts_with("//") {
        return Some(format!("https:{value}"));
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        return Some(value.to_string());
    }
    None
}

fn is_search_provider_url(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    [
        "duckduckgo.com",
        "lite.duckduckgo.com",
        "bing.com",
        "microsoft.com",
        "go.microsoft.com",
        "wikipedia.org/w/api.php",
        "api.crossref.org",
        "semanticscholar.org/graph",
        "eutils.ncbi.nlm.nih.gov",
        "ebi.ac.uk/europepmc/webservices",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn is_probably_fetchable_document(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("https://") || lower.starts_with("http://")) {
        return false;
    }
    if lower.contains("javascript:") || lower.contains("data:") || lower.contains("/search?") || lower.contains("/images/search") {
        return false;
    }
    ![
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".mp4", ".mp3", ".zip", ".rar", ".7z", ".tar", ".gz", ".exe", ".dmg",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
}

fn extract_xml_text(input: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = input.find(&start_tag)? + start_tag.len();
    let end = input[start..].find(&end_tag)? + start;
    Some(decode_basic_entities(input[start..end].trim()))
}

fn canonicalize_candidate_url(value: &str) -> String {
    let mut url = decode_basic_entities(value.trim()).trim().to_string();
    if url.starts_with("//") {
        url = format!("https:{url}");
    }
    if let Some(hash_pos) = url.find('#') {
        url.truncate(hash_pos);
    }
    while url.ends_with('/') && url.matches('/').count() > 2 {
        url.pop();
    }
    url
}

fn decode_basic_entities(value: &str) -> String {
    value
        .replace("&amp;", "&")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
}

fn collapse_ws(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn url_encode(value: &str) -> String {
    value.bytes().map(|b| match b {
        b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
        b' ' => "+".into(),
        _ => format!("%{b:02X}"),
    }).collect()
}

fn url_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                output.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hi = hex_value(bytes[i + 1]);
                let lo = hex_value(bytes[i + 2]);
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    output.push((hi << 4) | lo);
                    i += 3;
                } else {
                    output.push(bytes[i]);
                    i += 1;
                }
            }
            other => {
                output.push(other);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&output).to_string()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
