//! Temporary source graph for one governed deep-search run.

use super::types::DeepSearchSourceReliability;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepSearchSourceGraphSummary {
    pub total_seen: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub duplicate_candidates: usize,
    pub accepted_domains: Vec<String>,
    pub provider_counts: BTreeMap<String, usize>,
    pub reliability_average_score: f32,
    pub reliability_tier_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Default)]
pub(crate) struct DeepSearchSourceGraphBuilder {
    seen_urls: HashSet<String>,
    accepted_domains: HashSet<String>,
    provider_counts: BTreeMap<String, usize>,
    total_seen: usize,
    accepted: usize,
    rejected: usize,
    duplicate_candidates: usize,
    reliability_total: f32,
    reliability_tier_counts: BTreeMap<String, usize>,
}

impl DeepSearchSourceGraphBuilder {
    pub fn note_candidate(&mut self, url: &str, provider: &str) -> bool {
        self.total_seen += 1;
        *self.provider_counts.entry(provider.to_string()).or_insert(0) += 1;
        let key = url.trim().to_ascii_lowercase();
        if key.is_empty() || !self.seen_urls.insert(key) {
            self.duplicate_candidates += 1;
            return false;
        }
        true
    }

    pub fn note_accepted_with_reliability(&mut self, url: &str, reliability: &DeepSearchSourceReliability) {
        self.accepted += 1;
        self.reliability_total += reliability.score;
        *self
            .reliability_tier_counts
            .entry(format!("{:?}", reliability.tier))
            .or_insert(0) += 1;
        if let Some(domain) = host_from_url(url) {
            self.accepted_domains.insert(domain);
        }
    }

    pub fn note_rejected(&mut self) {
        self.rejected += 1;
    }

    pub fn summary(&self) -> DeepSearchSourceGraphSummary {
        let mut accepted_domains = self.accepted_domains.iter().cloned().collect::<Vec<_>>();
        accepted_domains.sort();
        DeepSearchSourceGraphSummary {
            total_seen: self.total_seen,
            accepted: self.accepted,
            rejected: self.rejected,
            duplicate_candidates: self.duplicate_candidates,
            accepted_domains,
            provider_counts: self.provider_counts.clone(),
            reliability_average_score: if self.accepted == 0 { 0.0 } else { (self.reliability_total / self.accepted as f32).clamp(0.0, 1.0) },
            reliability_tier_counts: self.reliability_tier_counts.clone(),
        }
    }
}

fn host_from_url(url: &str) -> Option<String> {
    let without_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    let host = without_scheme.split('/').next()?.split('@').last()?.split(':').next()?.trim().to_ascii_lowercase();
    if host.is_empty() { None } else { Some(host) }
}
