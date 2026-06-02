use crate::memory::types::{MemoryGraphSnapshot, MemoryNode, MemoryNodeKind, MemoryVerificationStatus};
use serde_json::json;
use std::collections::{HashMap, HashSet};

use super::types::{DeepSearchKnowledgeAutopilotRequest, DeepSearchLearningAgendaItem};

#[derive(Debug, Default)]
struct TopicAccumulator {
    topic: String,
    score: f32,
    source_node_ids: Vec<String>,
    tags: HashSet<String>,
    signals: HashSet<String>,
}

pub fn build_learning_agenda(
    snapshot: &MemoryGraphSnapshot,
    request: &DeepSearchKnowledgeAutopilotRequest,
) -> Vec<DeepSearchLearningAgendaItem> {
    let mut topics = HashMap::<String, TopicAccumulator>::new();

    for seed in &request.seed_topics {
        let normalized = normalize_topic(seed);
        if normalized.len() >= 3 {
            let entry = topics.entry(normalized.clone()).or_insert_with(|| TopicAccumulator {
                topic: seed.trim().to_string(),
                score: 0.92,
                ..Default::default()
            });
            entry.signals.insert("user_seed_topic".into());
        }
    }

    if request.include_low_confidence_claims {
        for node in &snapshot.nodes {
            if matches!(node.kind, MemoryNodeKind::Claim | MemoryNodeKind::ResearchFinding | MemoryNodeKind::ResearchTopic)
                && should_refresh_node(node)
            {
                let topic = claim_refresh_topic(node);
                add_topic(&mut topics, &topic, node, 0.78, "low_confidence_or_unverified_research_memory");
            }
        }
    }

    if request.include_user_context_topics {
        for node in &snapshot.nodes {
            if matches!(node.kind, MemoryNodeKind::UserPreference | MemoryNodeKind::Workflow | MemoryNodeKind::Task | MemoryNodeKind::Decision)
                && node.salience >= 0.45
            {
                let topic = concise_topic_from_node(node);
                add_topic(&mut topics, &topic, node, 0.48 + node.salience.min(1.0) * 0.22, "user_context_topic");
            }
        }
    }

    if request.include_topic_mining {
        for node in &snapshot.nodes {
            if !is_learning_relevant_kind(&node.kind) || node.salience < 0.30 {
                continue;
            }
            for tag in node.tags.iter().filter_map(|tag| clean_tag(tag)) {
                add_topic_text(&mut topics, &tag, Some(node), 0.36 + node.salience.min(1.0) * 0.18, "memory_tag_frequency");
            }
            for phrase in candidate_phrases(node) {
                add_topic_text(&mut topics, &phrase, Some(node), 0.30 + node.salience.min(1.0) * 0.12, "memory_title_summary_signal");
            }
        }
    }

    let blocked = request.blocked_topics.iter().map(|value| value.to_lowercase()).collect::<Vec<_>>();
    let mut agenda = topics
        .into_values()
        .filter(|topic| topic.score >= request.min_topic_priority.clamp(0.05, 0.95))
        .filter(|topic| !blocked.iter().any(|blocked| topic.topic.to_lowercase().contains(blocked)))
        .map(|topic| {
            let mut tags = topic.tags.into_iter().collect::<Vec<_>>();
            tags.sort();
            tags.truncate(10);
            let mut signals = topic.signals.into_iter().collect::<Vec<_>>();
            signals.sort();
            let objective = if signals.iter().any(|signal| signal.contains("low_confidence")) {
                format!("Refresh and strengthen weak or unverified knowledge about {} using independent web, academic and document sources.", topic.topic)
            } else {
                format!("Continuously learn high-signal external knowledge about {} and consolidate source-grounded claims into AstraOS memory.", topic.topic)
            };
            DeepSearchLearningAgendaItem {
                topic: topic.topic,
                objective,
                reason: signals.first().cloned().unwrap_or_else(|| "memory_topic_mining".into()),
                priority: topic.score.clamp(0.0, 1.0),
                source_node_ids: dedup(topic.source_node_ids, 16),
                tags,
                signals,
                metadata: json!({"source": "deep_search_knowledge_autopilot_topic_mining"}),
            }
        })
        .collect::<Vec<_>>();

    agenda.sort_by(|left, right| right.priority.partial_cmp(&left.priority).unwrap_or(std::cmp::Ordering::Equal));
    agenda.truncate(request.max_topics.clamp(1, 24));
    agenda
}

fn should_refresh_node(node: &MemoryNode) -> bool {
    node.confidence < 0.72
        || matches!(node.verification_status, MemoryVerificationStatus::Unverified | MemoryVerificationStatus::LlmInferred)
}

fn is_learning_relevant_kind(kind: &MemoryNodeKind) -> bool {
    matches!(
        kind,
        MemoryNodeKind::ResearchTopic
            | MemoryNodeKind::ResearchFinding
            | MemoryNodeKind::Claim
            | MemoryNodeKind::Concept
            | MemoryNodeKind::Task
            | MemoryNodeKind::Workflow
            | MemoryNodeKind::Decision
            | MemoryNodeKind::Procedure
            | MemoryNodeKind::CodePattern
            | MemoryNodeKind::Error
            | MemoryNodeKind::Fix
    )
}

fn claim_refresh_topic(node: &MemoryNode) -> String {
    let base = concise_topic_from_node(node);
    if base.to_lowercase().starts_with("verify ") { base } else { format!("verify {base}") }
}

fn concise_topic_from_node(node: &MemoryNode) -> String {
    let mut text = node.title.trim().to_string();
    if text.len() < 12 {
        text = node.summary.trim().to_string();
    }
    clean_phrase(&text).unwrap_or_else(|| node.title.trim().to_string())
}

fn candidate_phrases(node: &MemoryNode) -> Vec<String> {
    let mut phrases = Vec::new();
    if let Some(clean) = clean_phrase(&node.title) { phrases.push(clean); }
    if phrases.len() < 2 {
        if let Some(clean) = clean_phrase(&node.summary) { phrases.push(clean); }
    }
    phrases
}

fn add_topic(map: &mut HashMap<String, TopicAccumulator>, topic: &str, node: &MemoryNode, score: f32, signal: &str) {
    add_topic_text(map, topic, Some(node), score, signal);
}

fn add_topic_text(map: &mut HashMap<String, TopicAccumulator>, topic: &str, node: Option<&MemoryNode>, score: f32, signal: &str) {
    let Some(clean_topic) = clean_phrase(topic) else { return; };
    let key = normalize_topic(&clean_topic);
    if key.len() < 3 || STOP_TOPICS.contains(&key.as_str()) { return; }
    let entry = map.entry(key).or_insert_with(|| TopicAccumulator { topic: clean_topic, ..Default::default() });
    entry.score = (entry.score + score).min(1.0);
    entry.signals.insert(signal.to_string());
    if let Some(node) = node {
        entry.source_node_ids.push(node.id.clone());
        for tag in &node.tags {
            if let Some(clean) = clean_tag(tag) { entry.tags.insert(clean); }
        }
    }
}

fn clean_tag(tag: &str) -> Option<String> {
    let value = tag.trim().trim_matches('#').replace(['_', '-'], " ");
    clean_phrase(&value)
}

fn clean_phrase(value: &str) -> Option<String> {
    let mut out = value
        .replace(['\n', '\r', '\t'], " ")
        .split_whitespace()
        .take(10)
        .collect::<Vec<_>>()
        .join(" ");
    out = out.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
    if out.len() < 3 || out.len() > 140 { None } else { Some(out) }
}

fn normalize_topic(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|token| !STOP_WORDS.contains(token))
        .take(8)
        .collect::<Vec<_>>()
        .join(" ")
}

fn dedup(values: Vec<String>, limit: usize) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value);
        }
        if out.len() >= limit { break; }
    }
    out
}

const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "una", "uno", "del", "della", "delle", "degli", "che", "per", "con", "non", "come", "cosa", "memory", "memoria",
];

const STOP_TOPICS: &[&str] = &["unknown", "summary", "conversation", "task", "error", "fix"];
