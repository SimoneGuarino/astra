use crate::memory::{
    errors::MemoryResult,
    store::MemoryGraphStore,
    types::{MemoryActivation, MemoryActivationRequest, MemoryNode},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReconsolidationCandidate {
    pub node: MemoryNode,
    pub reason: String,
    pub user_message: String,
    pub assistant_answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReconsolidationRequest {
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub include_reprocessed: bool,
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReconsolidationItemReceipt {
    pub source_node_id: String,
    pub accepted: bool,
    pub reason: String,
    pub created_node_ids: Vec<String>,
    pub created_edge_ids: Vec<String>,
    #[serde(default)]
    pub semantic_atom_count: usize,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReconsolidationReceipt {
    pub accepted: bool,
    pub reason: String,
    pub scanned_candidates: usize,
    pub processed_candidates: usize,
    pub semantic_nodes_created: usize,
    pub semantic_edges_created: usize,
    pub skipped_candidates: usize,
    pub items: Vec<MemoryReconsolidationItemReceipt>,
    #[serde(default)]
    pub activation: Option<MemoryActivation>,
    #[serde(default)]
    pub metadata: Value,
}

pub fn list_reconsolidation_candidates(
    store: &MemoryGraphStore,
    limit: Option<usize>,
    include_reprocessed: bool,
) -> MemoryResult<Vec<MemoryReconsolidationCandidate>> {
    store.list_reconsolidation_candidates(limit.unwrap_or(50), include_reprocessed)
}

pub fn finalize_reconsolidation_receipt(
    store: &MemoryGraphStore,
    request_id: Option<String>,
    root_query: String,
    items: Vec<MemoryReconsolidationItemReceipt>,
    scanned_candidates: usize,
) -> MemoryResult<MemoryReconsolidationReceipt> {
    let activated_node_ids = items
        .iter()
        .flat_map(|item| item.created_node_ids.iter().cloned())
        .take(40)
        .collect::<Vec<_>>();
    let activation = if activated_node_ids.is_empty() {
        None
    } else {
        Some(store.activate(MemoryActivationRequest {
            request_id,
            root_query,
            seed_node_ids: activated_node_ids,
            max_depth: 2,
            max_nodes: 40,
            metadata: json!({
                "source": "memory_reconsolidation",
                "metadata_only": true,
            }),
        })?)
    };
    let semantic_nodes_created = items.iter().map(|item| item.created_node_ids.len()).sum::<usize>();
    let semantic_edges_created = items.iter().map(|item| item.created_edge_ids.len()).sum::<usize>();
    let processed_candidates = items.iter().filter(|item| item.accepted).count();
    let skipped_candidates = items.len().saturating_sub(processed_candidates);
    Ok(MemoryReconsolidationReceipt {
        accepted: true,
        reason: "memory reconsolidation completed; semantic nodes remain advisory context and do not bypass governance".into(),
        scanned_candidates,
        processed_candidates,
        semantic_nodes_created,
        semantic_edges_created,
        skipped_candidates,
        items,
        activation,
        metadata: json!({
            "llm_first": true,
            "rust_governed": true,
            "source_of_truth": "sqlite_memory_graph",
            "metadata_only": true,
        }),
    })
}
