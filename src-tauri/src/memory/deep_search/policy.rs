use super::types::DeepSearchPolicy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeepSearchPolicyDecision {
    Allow,
    Reject { reason: String },
}

pub fn evaluate_source(url: &str, policy: &DeepSearchPolicy) -> DeepSearchPolicyDecision {
    let normalized = url.trim().to_ascii_lowercase();
    if !(normalized.starts_with("https://") || (policy.allow_http_localhost && normalized.starts_with("http://127.0.0.1")) || (policy.allow_http_localhost && normalized.starts_with("http://localhost"))) {
        return DeepSearchPolicyDecision::Reject { reason: "only https sources are allowed, except explicit localhost development sources".into() };
    }
    if normalized.contains("javascript:") || normalized.contains("data:") || normalized.contains("file:") {
        return DeepSearchPolicyDecision::Reject { reason: "unsafe URL scheme or embedded payload".into() };
    }
    let host = host_from_url(&normalized);
    if host.is_empty() {
        return DeepSearchPolicyDecision::Reject { reason: "source host could not be parsed".into() };
    }
    if policy.blocked_domains.iter().any(|domain| domain_matches(&host, domain)) {
        return DeepSearchPolicyDecision::Reject { reason: format!("domain is blocked by deep-search policy: {host}") };
    }
    if !policy.allowed_domains.is_empty() && !policy.allowed_domains.iter().any(|domain| domain_matches(&host, domain)) {
        return DeepSearchPolicyDecision::Reject { reason: format!("domain is not in allowed_domains: {host}") };
    }
    DeepSearchPolicyDecision::Allow
}

fn host_from_url(url: &str) -> String {
    let without_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    without_scheme.split('/').next().unwrap_or_default().split('@').last().unwrap_or_default().split(':').next().unwrap_or_default().to_string()
}

fn domain_matches(host: &str, domain: &str) -> bool {
    let domain = domain.trim().trim_start_matches('.').to_ascii_lowercase();
    !domain.is_empty() && (host == domain || host.ends_with(&format!(".{domain}")))
}
