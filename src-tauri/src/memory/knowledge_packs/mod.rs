pub mod types;

use crate::memory::{
    errors::MemoryResult,
    store::MemoryGraphStore,
    types::{
        now_ms, CreateMemoryEdgeRequest, CreateMemoryNodeRequest, MemoryNode,
        MemoryNodeKind, MemoryRelationKind, MemoryVerificationStatus,
    },
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

pub use types::{
    KnowledgePackBuildRequest, KnowledgePackBuildReceipt, KnowledgePackKindCount,
    KnowledgePackMember, KnowledgePackSummary,
};

pub fn build_local_knowledge_packs(
    store: &MemoryGraphStore,
    request: KnowledgePackBuildRequest,
) -> MemoryResult<KnowledgePackBuildReceipt> {
    let started_at = now_ms();
    if !request.enabled {
        return Ok(KnowledgePackBuildReceipt {
            accepted: false,
            reason: "knowledge_pack_builder_disabled".into(),
            started_at,
            completed_at: now_ms(),
            dry_run: request.dry_run,
            snapshot_nodes: 0,
            packs_built: 0,
            packs_persisted: 0,
            created_node_ids: Vec::new(),
            created_edge_ids: Vec::new(),
            packs: Vec::new(),
            warnings: Vec::new(),
            recommendations: vec!["Enable knowledge pack building before running domain brain consolidation.".into()],
            metadata: json!({"schema_version": 1}),
        });
    }

    let snapshot_limit = request.snapshot_limit.clamp(20, 1_000);
    let snapshot = store.snapshot(snapshot_limit)?;
    let mut warnings = Vec::new();
    let candidates = filter_pack_candidates(snapshot.nodes, &request);
    if candidates.is_empty() {
        return Ok(KnowledgePackBuildReceipt {
            accepted: false,
            reason: "no_packable_memory_nodes".into(),
            started_at,
            completed_at: now_ms(),
            dry_run: request.dry_run,
            snapshot_nodes: 0,
            packs_built: 0,
            packs_persisted: 0,
            created_node_ids: Vec::new(),
            created_edge_ids: Vec::new(),
            packs: Vec::new(),
            warnings,
            recommendations: vec!["Run Deep Search Knowledge Autopilot or conversation consolidation before building domain packs.".into()],
            metadata: json!({"schema_version": 1}),
        });
    }

    let mut domain_scores = mine_domain_candidates(&candidates, &request);
    if domain_scores.is_empty() {
        domain_scores.insert("general_knowledge".into(), 1.0);
    }

    let max_packs = request.max_packs.clamp(1, 32);
    let max_nodes_per_pack = request.max_nodes_per_pack.clamp(3, 80);
    let min_pack_score = request.min_pack_score.clamp(0.05, 0.95);
    let mut domains = domain_scores.into_iter().collect::<Vec<_>>();
    domains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));

    let mut packs = Vec::new();
    for (domain, domain_signal_score) in domains {
        if packs.len() >= max_packs {
            break;
        }
        if is_blocked_domain(&domain, &request.blocked_domains) {
            continue;
        }
        let mut members = candidates
            .iter()
            .filter_map(|node| score_member_for_domain(node, &domain).map(|score| (node, score)))
            .collect::<Vec<_>>();
        if request.seed_domains.iter().any(|seed| normalize_domain(seed) == domain) && members.is_empty() {
            members = candidates.iter().take(max_nodes_per_pack).map(|node| (node, 0.35)).collect();
        }
        members.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        members.truncate(max_nodes_per_pack);
        if members.len() < request.min_nodes_per_pack.clamp(2, 24) {
            continue;
        }
        let pack = build_pack_summary(&domain, domain_signal_score, members, &request);
        if pack.score >= min_pack_score {
            packs.push(pack);
        }
    }

    packs.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let mut created_node_ids = Vec::new();
    let mut created_edge_ids = Vec::new();
    let mut packs_persisted = 0usize;
    if request.persist_packs && !request.dry_run {
        for pack in &packs {
            let source = format!("astra://knowledge-pack/{}", pack.domain_slug);
            let existing = store.find_node_by_source(&source)?;
            let pack_node = match existing {
                Some(node) => node,
                None => {
                    let node = store.create_node_once_by_source(CreateMemoryNodeRequest {
                        kind: MemoryNodeKind::Summary,
                        title: pack.title.clone(),
                        summary: pack.summary.clone(),
                        content: Some(pack.content.clone()),
                        tags: pack.tags.clone(),
                        source: Some(source.clone()),
                        confidence: pack.confidence,
                        verification_status: MemoryVerificationStatus::LlmInferred,
                        salience: pack.salience,
                        metadata: json!({
                            "source": "knowledge_pack_builder",
                            "schema_version": 1,
                            "domain_slug": pack.domain_slug.clone(),
                            "score": pack.score,
                            "member_count": pack.member_count,
                            "kind_counts": pack.kind_counts.clone(),
                            "request_metadata": request.metadata.clone(),
                        }),
                    })?;
                    created_node_ids.push(node.id.clone());
                    node
                }
            };
            if !created_node_ids.iter().any(|id| id == &pack_node.id) {
                warnings.push(format!("knowledge pack already existed and was not overwritten: {}", pack.domain_slug));
                continue;
            }
            packs_persisted += 1;
            for member in pack.members.iter().take(max_nodes_per_pack) {
                if member.node_id == pack_node.id {
                    continue;
                }
                match store.create_edge(CreateMemoryEdgeRequest {
                    from_node_id: member.node_id.clone(),
                    to_node_id: pack_node.id.clone(),
                    relation: MemoryRelationKind::PartOf,
                    weight: member.score.clamp(0.1, 1.0),
                    confidence: pack.confidence.min(member.score).clamp(0.1, 1.0),
                    metadata: json!({
                        "source": "knowledge_pack_builder",
                        "domain_slug": pack.domain_slug.clone(),
                        "member_title": member.title.clone(),
                    }),
                }) {
                    Ok(edge) => created_edge_ids.push(edge.id),
                    Err(error) => warnings.push(format!("failed_to_link_pack_member:{}:{error}", member.node_id)),
                }
            }
        }
    }

    let recommendations = build_pack_recommendations(&packs, &request);
    Ok(KnowledgePackBuildReceipt {
        accepted: true,
        reason: if request.dry_run { "knowledge_pack_dry_run_completed" } else { "knowledge_pack_build_completed" }.into(),
        started_at,
        completed_at: now_ms(),
        dry_run: request.dry_run,
        snapshot_nodes: candidates.len(),
        packs_built: packs.len(),
        packs_persisted,
        created_node_ids,
        created_edge_ids,
        packs,
        warnings,
        recommendations,
        metadata: json!({
            "schema_version": 1,
            "bounded": true,
            "persist_packs": request.persist_packs,
            "max_packs": max_packs,
            "max_nodes_per_pack": max_nodes_per_pack,
        }),
    })
}

fn filter_pack_candidates(nodes: Vec<MemoryNode>, request: &KnowledgePackBuildRequest) -> Vec<MemoryNode> {
    nodes
        .into_iter()
        .filter(|node| match node.verification_status {
            MemoryVerificationStatus::Deprecated | MemoryVerificationStatus::Contradicted => false,
            MemoryVerificationStatus::Unverified => request.include_unverified,
            _ => true,
        })
        .filter(|node| request.include_source_documents || node.kind != MemoryNodeKind::SourceDocument)
        .filter(|node| request.include_low_confidence || node.confidence >= 0.45)
        .filter(|node| !node.title.trim().is_empty() && !node.summary.trim().is_empty())
        .collect()
}

fn mine_domain_candidates(nodes: &[MemoryNode], request: &KnowledgePackBuildRequest) -> HashMap<String, f32> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for seed in &request.seed_domains {
        let normalized = normalize_domain(seed);
        if !normalized.is_empty() {
            *scores.entry(normalized).or_insert(0.0) += 2.5;
        }
    }
    for node in nodes {
        let base = node.confidence.clamp(0.05, 1.0) * 0.55 + node.salience.clamp(0.05, 1.0) * 0.45;
        for tag in &node.tags {
            let normalized = normalize_domain(tag.strip_prefix("domain:").unwrap_or(tag));
            if is_packable_token(&normalized) {
                *scores.entry(normalized).or_insert(0.0) += base * 1.25;
            }
        }
        for token in extract_domain_tokens(&format!("{} {}", node.title, node.summary)) {
            *scores.entry(token).or_insert(0.0) += base * 0.45;
        }
        if let Some(source_domain) = source_domain_hint(node) {
            *scores.entry(source_domain).or_insert(0.0) += base * 0.35;
        }
    }
    scores.retain(|domain, score| *score >= 0.65 && !is_blocked_domain(domain, &request.blocked_domains));
    scores
}

fn score_member_for_domain(node: &MemoryNode, domain: &str) -> Option<f32> {
    let domain_words = domain.split('_').filter(|value| !value.is_empty()).collect::<Vec<_>>();
    if domain_words.is_empty() {
        return None;
    }
    let searchable = normalize_search_text(&format!(
        "{} {} {} {}",
        node.title,
        node.summary,
        node.tags.join(" "),
        node.source.clone().unwrap_or_default()
    ));
    let mut hits = 0usize;
    for word in &domain_words {
        if searchable.contains(word) {
            hits += 1;
        }
    }
    if hits == 0 {
        return None;
    }
    let coverage = hits as f32 / domain_words.len() as f32;
    let status_boost = match node.verification_status {
        MemoryVerificationStatus::SystemVerified => 1.0,
        MemoryVerificationStatus::UserConfirmed => 0.92,
        MemoryVerificationStatus::LlmInferred => 0.78,
        MemoryVerificationStatus::Unverified => 0.52,
        MemoryVerificationStatus::Contradicted | MemoryVerificationStatus::Deprecated => 0.0,
    };
    Some((coverage * 0.45 + node.confidence * 0.25 + node.salience * 0.20 + status_boost * 0.10).clamp(0.0, 1.0))
}

fn build_pack_summary(domain: &str, domain_signal_score: f32, members: Vec<(&MemoryNode, f32)>, request: &KnowledgePackBuildRequest) -> KnowledgePackSummary {
    let member_count = members.len();
    let mut tags = vec!["knowledge_pack".into(), "domain_brain".into(), format!("domain:{domain}")];
    let mut seen_tags = HashSet::new();
    let mut member_values = Vec::new();
    let mut kind_counts: HashMap<String, usize> = HashMap::new();
    let mut avg_confidence = 0.0f32;
    let mut avg_salience = 0.0f32;
    let mut avg_member_score = 0.0f32;
    let mut source_count = 0usize;
    for (node, score) in members {
        avg_confidence += node.confidence;
        avg_salience += node.salience;
        avg_member_score += score;
        *kind_counts.entry(node.kind.as_str().into()).or_insert(0) += 1;
        if node.kind == MemoryNodeKind::SourceDocument || node.source.is_some() {
            source_count += 1;
        }
        for tag in &node.tags {
            let normalized = normalize_domain(tag.strip_prefix("domain:").unwrap_or(tag));
            if is_packable_token(&normalized) && seen_tags.insert(normalized.clone()) && tags.len() < 16 {
                tags.push(normalized);
            }
        }
        member_values.push(KnowledgePackMember {
            node_id: node.id.clone(),
            title: node.title.clone(),
            kind: node.kind.as_str().into(),
            confidence: node.confidence,
            salience: node.salience,
            verification_status: node.verification_status.as_str().into(),
            score,
            tags: node.tags.clone(),
            signals: member_signals(node, score),
        });
    }
    let denom = member_count.max(1) as f32;
    avg_confidence /= denom;
    avg_salience /= denom;
    avg_member_score /= denom;
    let diversity_score = (kind_counts.len() as f32 / 6.0).clamp(0.0, 1.0);
    let evidence_score = (source_count as f32 / member_count.max(1) as f32).clamp(0.0, 1.0);
    let pack_score = (avg_member_score * 0.42
        + avg_confidence * 0.18
        + avg_salience * 0.16
        + diversity_score * 0.12
        + evidence_score * 0.12)
        .max((domain_signal_score / 10.0).clamp(0.0, 1.0) * 0.35)
        .clamp(0.0, 1.0);
    let title = format!("Knowledge Pack: {}", humanize_domain(domain));
    let summary = format!(
        "Domain brain pack for {} built from {} memory nodes. It groups related claims, findings, procedures, sources and concepts for retrieval and future learning.",
        humanize_domain(domain), member_count
    );
    let content = build_pack_content(&title, &summary, &member_values, &kind_counts, request);
    let mut kind_count_values = kind_counts
        .into_iter()
        .map(|(kind, count)| KnowledgePackKindCount { kind, count })
        .collect::<Vec<_>>();
    kind_count_values.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.kind.cmp(&b.kind)));
    KnowledgePackSummary {
        domain_slug: domain.into(),
        title,
        summary,
        content,
        score: pack_score,
        confidence: (avg_confidence * 0.8 + pack_score * 0.2).clamp(0.1, 0.92),
        salience: (avg_salience * 0.7 + pack_score * 0.3).clamp(0.1, 0.95),
        member_count,
        tags,
        kind_counts: kind_count_values,
        members: member_values,
        metadata: json!({
            "schema_version": 1,
            "diversity_score": diversity_score,
            "evidence_score": evidence_score,
            "domain_signal_score": domain_signal_score,
            "bounded": true,
        }),
    }
}

fn build_pack_content(
    title: &str,
    summary: &str,
    members: &[KnowledgePackMember],
    kind_counts: &HashMap<String, usize>,
    request: &KnowledgePackBuildRequest,
) -> String {
    let mut lines = Vec::new();
    lines.push(title.to_string());
    lines.push(String::new());
    lines.push(summary.to_string());
    lines.push(String::new());
    lines.push("Kind distribution:".into());
    let mut counts = kind_counts.iter().collect::<Vec<_>>();
    counts.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (kind, count) in counts.into_iter().take(12) {
        lines.push(format!("- {kind}: {count}"));
    }
    lines.push(String::new());
    lines.push("Representative memory nodes:".into());
    for member in members.iter().take(request.max_pack_content_members.clamp(5, 30)) {
        lines.push(format!(
            "- [{}] {} | confidence {:.2} | salience {:.2} | pack_score {:.2}",
            member.kind, member.title, member.confidence, member.salience, member.score
        ));
    }
    lines.join("\n")
}

fn member_signals(node: &MemoryNode, score: f32) -> Vec<String> {
    let mut signals = Vec::new();
    signals.push(format!("score:{score:.2}"));
    signals.push(format!("kind:{}", node.kind.as_str()));
    signals.push(format!("status:{}", node.verification_status.as_str()));
    if node.source.is_some() {
        signals.push("has_source".into());
    }
    if node.confidence >= 0.75 {
        signals.push("high_confidence".into());
    }
    if node.salience >= 0.70 {
        signals.push("high_salience".into());
    }
    signals
}

fn build_pack_recommendations(packs: &[KnowledgePackSummary], request: &KnowledgePackBuildRequest) -> Vec<String> {
    let mut recommendations = Vec::new();
    if packs.is_empty() {
        recommendations.push("No domain pack reached the minimum score; run knowledge autopilot or lower min_pack_score for diagnostics.".into());
    }
    if request.dry_run && !packs.is_empty() {
        recommendations.push("Dry-run completed. Re-run with persist_packs=true to create domain-brain summary nodes.".into());
    }
    if packs.iter().any(|pack| pack.member_count > request.max_nodes_per_pack) {
        recommendations.push("Some packs were truncated by max_nodes_per_pack; increase the bound only if retrieval latency remains healthy.".into());
    }
    if packs.iter().any(|pack| pack.kind_counts.iter().all(|kind| kind.kind != "source_document")) {
        recommendations.push("Some packs have little source-document evidence; use knowledge refresh or deep-search autopilot to reinforce them.".into());
    }
    recommendations
}

fn source_domain_hint(node: &MemoryNode) -> Option<String> {
    let source = node.source.as_ref()?.to_lowercase();
    if source.starts_with("http://") || source.starts_with("https://") {
        let without_scheme = source.split_once("://").map(|(_, rest)| rest).unwrap_or(&source);
        let host = without_scheme.split('/').next().unwrap_or("");
        let parts = host.split('.').filter(|part| !part.is_empty()).collect::<Vec<_>>();
        if parts.len() >= 2 {
            return Some(normalize_domain(parts[parts.len() - 2]));
        }
    }
    None
}

fn extract_domain_tokens(text: &str) -> Vec<String> {
    let normalized = normalize_search_text(text);
    let tokens = normalized
        .split_whitespace()
        .filter_map(|token| {
            let value = normalize_domain(token);
            if is_packable_token(&value) { Some(value) } else { None }
        })
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for token in tokens {
        if seen.insert(token.clone()) {
            out.push(token);
        }
        if out.len() >= 12 {
            break;
        }
    }
    out
}

fn normalize_search_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch.to_ascii_lowercase() } else { ' ' })
        .collect::<String>()
}

fn normalize_domain(value: &str) -> String {
    let cleaned = value
        .trim()
        .trim_start_matches("domain:")
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch.to_ascii_lowercase() } else { '_' })
        .collect::<String>();
    cleaned
        .split('_')
        .filter(|part| !part.is_empty())
        .take(5)
        .collect::<Vec<_>>()
        .join("_")
}

fn humanize_domain(value: &str) -> String {
    value
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => format!("{}{}", first.to_ascii_uppercase(), chars.as_str()),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_blocked_domain(domain: &str, blocked: &[String]) -> bool {
    let domain = normalize_domain(domain);
    blocked.iter().any(|item| {
        let blocked = normalize_domain(item);
        !blocked.is_empty() && domain.contains(&blocked)
    })
}

fn is_packable_token(value: &str) -> bool {
    if value.len() < 3 || value.len() > 48 {
        return false;
    }
    !matches!(
        value,
        "the" | "and" | "for" | "with" | "that" | "this" | "from" | "into" | "about" |
        "come" | "che" | "con" | "per" | "del" | "della" | "delle" | "degli" | "una" |
        "uno" | "gli" | "nel" | "nella" | "sono" | "alla" | "allo" | "astra" | "memory" |
        "knowledge" | "pack" | "source" | "node" | "summary" | "claim" | "finding"
    )
}
