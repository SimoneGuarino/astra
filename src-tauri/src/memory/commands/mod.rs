use crate::memory::{
    consolidation::{
        ConversationMemoryBundle, ConversationMemoryConsolidationReceipt, ResearchMemoryBundle,
        ResearchMemoryConsolidationReceipt,
        MemoryReconsolidationCandidate, MemoryReconsolidationRequest, MemoryReconsolidationReceipt,
    },
    embeddings::{build_embedding_provider, EmbeddingProvider, EmbeddingRequest, StableHashEmbeddingProvider},
    store::MemoryGraphStore,
    types::{
        now_ms, CreateMemoryEdgeRequest, CreateMemoryNodeRequest, MemoryActivation,
        MemoryActivationRequest, MemoryCanonicalReviewCandidate, MemoryCanonicalReviewRequest, MemoryCanonicalReviewApplyRequest, MemoryDuplicateCandidate, MemoryDuplicateCandidateRequest, MemoryEdge, MemoryEmbeddingIndexStatus,
        MemoryEmbeddingMaintenanceReceipt, MemoryEmbeddingMaintenanceRequest,
        MemoryEmbeddingRebuildReceipt, MemoryEmbeddingRebuildRequest, MemoryEmbeddingRecord,
        MemoryGovernancePolicySnapshot, MemoryGraphSnapshot, MemoryHybridQueryRequest, MemoryMergeNodesReceipt, MemoryMergeNodesRequest,
        MemoryHybridQueryResponse, MemoryNode, MemoryQualityDashboard, MemoryNodeGovernanceUpdateReceipt,
        MemoryNodeGovernanceUpdateRequest, MemoryQueryRequest, MemoryQueryResponse, MemorySkillCandidate, MemorySkillCandidateExtractionReceipt, MemorySkillCandidateUpdateReceipt, MemorySkillCandidateUpdateRequest,
    },
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub fn status(store: &MemoryGraphStore) -> Value {
    store.status()
}


pub fn quality_dashboard(store: &MemoryGraphStore) -> Result<MemoryQualityDashboard, String> {
    store.quality_dashboard().map_err(|error| error.to_string())
}




pub fn list_canonical_review_candidates(
    store: &MemoryGraphStore,
    request: MemoryCanonicalReviewRequest,
) -> Result<Vec<MemoryCanonicalReviewCandidate>, String> {
    store
        .list_canonical_review_candidates(request)
        .map_err(|error| error.to_string())
}

pub fn apply_canonical_review(
    store: &MemoryGraphStore,
    request: MemoryCanonicalReviewApplyRequest,
) -> Result<MemoryMergeNodesReceipt, String> {
    let target_node_id = request.candidate.target_node.id.clone();
    let source_node_ids = request
        .candidate
        .candidate_nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<Vec<_>>();
    store
        .merge_nodes(MemoryMergeNodesRequest {
            target_node_id,
            source_node_ids,
            mark_sources_deprecated: request.mark_sources_deprecated,
            actor: request.actor.or_else(|| Some("user".into())),
            reason: request.reason.or_else(|| Some("user_approved_canonical_memory_review".into())),
            metadata: json!({
                "source": "memory_canonical_review",
                "review_candidate_id": request.candidate.id,
                "review_confidence": request.candidate.confidence,
                "review_reasons": request.candidate.reasons,
                "proposed_title": request.candidate.proposed_title,
                "proposed_summary": request.candidate.proposed_summary,
                "user_governed": true,
                "metadata_only": true,
                "request_metadata": request.metadata,
            }),
        })
        .map_err(|error| error.to_string())
}

pub fn list_duplicate_candidates(
    store: &MemoryGraphStore,
    request: MemoryDuplicateCandidateRequest,
) -> Result<Vec<MemoryDuplicateCandidate>, String> {
    store
        .list_duplicate_candidates(request)
        .map_err(|error| error.to_string())
}

pub fn merge_nodes(
    store: &MemoryGraphStore,
    request: MemoryMergeNodesRequest,
) -> Result<MemoryMergeNodesReceipt, String> {
    store.merge_nodes(request).map_err(|error| error.to_string())
}

pub fn governance_policy() -> MemoryGovernancePolicySnapshot {
    crate::memory::governance::governance_policy_snapshot()
}

pub fn update_node_governance(
    store: &MemoryGraphStore,
    request: MemoryNodeGovernanceUpdateRequest,
) -> Result<MemoryNodeGovernanceUpdateReceipt, String> {
    store
        .update_node_governance(request)
        .map_err(|error| error.to_string())
}

pub fn create_node(store: &MemoryGraphStore, request: CreateMemoryNodeRequest) -> Result<MemoryNode, String> {
    store.create_node(request).map_err(|error| error.to_string())
}

pub fn create_edge(store: &MemoryGraphStore, request: CreateMemoryEdgeRequest) -> Result<MemoryEdge, String> {
    store.create_edge(request).map_err(|error| error.to_string())
}

pub fn query(store: &MemoryGraphStore, request: MemoryQueryRequest) -> Result<MemoryQueryResponse, String> {
    store.query(request).map_err(|error| error.to_string())
}

pub fn hybrid_query(
    store: &MemoryGraphStore,
    request: MemoryHybridQueryRequest,
) -> Result<MemoryHybridQueryResponse, String> {
    let provider = build_embedding_provider();
    let mut query_embedding_error: Option<String> = None;
    let query_embedding = if request.query.trim().is_empty() {
        None
    } else {
        match provider.embed(EmbeddingRequest {
            text: request.query.clone(),
            model: None,
        }) {
            Ok(response) => Some(response.vector),
            Err(error) => {
                query_embedding_error = Some(error);
                None
            }
        }
    };
    let mut response = store
        .hybrid_query(request, query_embedding)
        .map_err(|error| error.to_string())?;
    if let Some(metadata) = response.metadata.as_object_mut() {
        if let Some(error) = query_embedding_error {
            metadata.insert("vector_query_embedding_error".into(), json!(error));
            metadata.insert("vector_query_fallback".into(), json!("lexical_graph_only"));
        }
        metadata.insert("query_embedding_provider_kind".into(), json!(provider.provider_kind()));
        metadata.insert("query_embedding_model".into(), json!(provider.default_model()));
    }
    Ok(response)
}

pub fn activate(store: &MemoryGraphStore, request: MemoryActivationRequest) -> Result<MemoryActivation, String> {
    store.activate(request).map_err(|error| error.to_string())
}

pub fn recent_activations(store: &MemoryGraphStore, limit: usize) -> Result<Vec<MemoryActivation>, String> {
    store.recent_activations(limit).map_err(|error| error.to_string())
}

pub fn snapshot(store: &MemoryGraphStore, limit: usize) -> Result<MemoryGraphSnapshot, String> {
    store.snapshot(limit).map_err(|error| error.to_string())
}

pub fn embedding_status(store: &MemoryGraphStore) -> Result<MemoryEmbeddingIndexStatus, String> {
    store.embedding_status().map_err(|error| error.to_string())
}

pub fn rebuild_embedding_index(
    store: &MemoryGraphStore,
    request: MemoryEmbeddingRebuildRequest,
) -> Result<MemoryEmbeddingRebuildReceipt, String> {
    let provider = build_embedding_provider();
    let stable_fallback = StableHashEmbeddingProvider::default();
    let fallback_enabled = embedding_fallback_enabled();
    let limit = request.limit.unwrap_or(500).clamp(1, 5_000);
    let chunks = store
        .list_chunks_for_embedding(limit, request.force)
        .map_err(|error| error.to_string())?;
    if chunks.is_empty() {
        return Ok(MemoryEmbeddingRebuildReceipt {
            accepted: true,
            reason: "no memory chunks pending embedding".into(),
            indexed_chunks: 0,
            skipped_chunks: 0,
            failed_chunks: 0,
            model: request
                .model
                .unwrap_or_else(|| provider.default_model()),
            dimensions: provider.dimensions_hint().unwrap_or(0),
            sample_node_ids: Vec::new(),
            metadata: json!({
                "backend": "sqlite_vector_cache",
                "metadata_only": true,
            }),
        });
    }

    let mut indexed = 0usize;
    let mut failed = 0usize;
    let mut model_used = request
        .model
        .clone()
        .unwrap_or_else(|| provider.default_model());
    let mut provider_kind_used = provider.provider_kind().to_string();
    let mut dimensions_used = provider.dimensions_hint().unwrap_or(0);
    let mut sample_node_ids = Vec::new();

    for chunk in chunks {
        let text = chunk.text.trim();
        if text.is_empty() {
            failed += 1;
            continue;
        }
        let primary = provider.embed(EmbeddingRequest {
            text: text.to_string(),
            model: request.model.clone(),
        });
        let response = match primary {
            Ok(response) => {
                provider_kind_used = provider.provider_kind().to_string();
                response
            }
            Err(primary_error) if fallback_enabled && provider.provider_kind() != stable_fallback.provider_kind() => {
                match stable_fallback.embed(EmbeddingRequest {
                    text: text.to_string(),
                    model: None,
                }) {
                    Ok(response) => {
                        provider_kind_used = format!("{}_fallback_after_{}", stable_fallback.provider_kind(), provider.provider_kind());
                        response
                    }
                    Err(fallback_error) => {
                        failed += 1;
                        let _ = store.append_memory_note("embedding_failed", json!({
                            "chunk_id": chunk.id,
                            "node_id": chunk.node_id,
                            "provider_kind": provider.provider_kind(),
                            "primary_error": primary_error,
                            "fallback_error": fallback_error,
                            "metadata_only": true,
                        }));
                        continue;
                    }
                }
            }
            Err(error) => {
                failed += 1;
                let _ = store.append_memory_note("embedding_failed", json!({
                    "chunk_id": chunk.id,
                    "node_id": chunk.node_id,
                    "provider_kind": provider.provider_kind(),
                    "error": error,
                    "metadata_only": true,
                }));
                continue;
            }
        };

        model_used = response.model.clone();
        dimensions_used = response.vector.len();
        let now = now_ms();
        let record = MemoryEmbeddingRecord {
            chunk_id: chunk.id.clone(),
            node_id: chunk.node_id.clone(),
            model: response.model,
            dimensions: response.vector.len(),
            vector: response.vector,
            created_at: now,
            updated_at: now,
            metadata: json!({
                "embedding_source": "memory_rebuild_embedding_index",
                "chunk_text_hash": sha256_hex(text),
                "provider_kind": provider_kind_used.clone(),
                "fallback_enabled": fallback_enabled,
                "metadata_only": true,
            }),
        };
        if store.upsert_embedding_record(record).is_ok() {
            indexed += 1;
            if sample_node_ids.len() < 12 && !sample_node_ids.contains(&chunk.node_id) {
                sample_node_ids.push(chunk.node_id);
            }
        } else {
            failed += 1;
        }
    }

    Ok(MemoryEmbeddingRebuildReceipt {
        accepted: true,
        reason: "memory vector index rebuilt through governed local embedding adapter".into(),
        indexed_chunks: indexed,
        skipped_chunks: 0,
        failed_chunks: failed,
        model: model_used,
        dimensions: dimensions_used,
        sample_node_ids,
        metadata: json!({
            "backend": "sqlite_vector_cache",
            "provider_kind": provider_kind_used.clone(),
            "fallback_enabled": fallback_enabled,
            "source_of_truth": "sqlite_memory_graph",
            "metadata_only": true,
        }),
    })
}





pub fn run_embedding_maintenance(
    store: &MemoryGraphStore,
    request: MemoryEmbeddingMaintenanceRequest,
) -> Result<MemoryEmbeddingMaintenanceReceipt, String> {
    if !memory_embedding_auto_index_enabled() && !request.force {
        let status = store.embedding_status().map_err(|error| error.to_string())?;
        return Ok(MemoryEmbeddingMaintenanceReceipt {
            accepted: true,
            reason: "automatic memory embedding maintenance is disabled".into(),
            ran: false,
            indexed_chunks: 0,
            skipped_chunks: 0,
            failed_chunks: 0,
            pending_before: status.pending_chunks,
            pending_after: status.pending_chunks,
            model: status.provider,
            dimensions: status.dimensions,
            sample_node_ids: Vec::new(),
            metadata: json!({
                "auto_index_enabled": false,
                "metadata_only": true,
            }),
        });
    }

    let before = store.embedding_status().map_err(|error| error.to_string())?;
    if before.pending_chunks == 0 && !request.force {
        return Ok(MemoryEmbeddingMaintenanceReceipt {
            accepted: true,
            reason: "no pending memory embeddings".into(),
            ran: false,
            indexed_chunks: 0,
            skipped_chunks: 0,
            failed_chunks: 0,
            pending_before: 0,
            pending_after: 0,
            model: before.provider,
            dimensions: before.dimensions,
            sample_node_ids: Vec::new(),
            metadata: json!({
                "auto_index_enabled": true,
                "metadata_only": true,
            }),
        });
    }

    let limit = request
        .limit
        .unwrap_or_else(memory_embedding_auto_index_batch_size)
        .clamp(1, 256);
    let receipt = rebuild_embedding_index(
        store,
        MemoryEmbeddingRebuildRequest {
            limit: Some(limit),
            force: request.force,
            model: request.model.clone(),
        },
    )?;
    let after = store.embedding_status().map_err(|error| error.to_string())?;
    let _ = store.append_memory_note("embedding_maintenance_ran", json!({
        "reason": request.reason.unwrap_or_else(|| "manual_or_runtime_maintenance".into()),
        "limit": limit,
        "force": request.force,
        "pending_before": before.pending_chunks,
        "pending_after": after.pending_chunks,
        "indexed_chunks": receipt.indexed_chunks,
        "failed_chunks": receipt.failed_chunks,
        "model": receipt.model,
        "dimensions": receipt.dimensions,
        "metadata_only": true,
    }));

    Ok(MemoryEmbeddingMaintenanceReceipt {
        accepted: true,
        reason: "memory embedding maintenance completed".into(),
        ran: true,
        indexed_chunks: receipt.indexed_chunks,
        skipped_chunks: receipt.skipped_chunks,
        failed_chunks: receipt.failed_chunks,
        pending_before: before.pending_chunks,
        pending_after: after.pending_chunks,
        model: receipt.model,
        dimensions: receipt.dimensions,
        sample_node_ids: receipt.sample_node_ids,
        metadata: json!({
            "backend": "sqlite_vector_cache",
            "auto_index_enabled": memory_embedding_auto_index_enabled(),
            "batch_limit": limit,
            "metadata_only": true,
        }),
    })
}

pub fn extract_skill_candidates(
    store: &MemoryGraphStore,
    limit: Option<usize>,
) -> Result<MemorySkillCandidateExtractionReceipt, String> {
    crate::memory::skills::extract_skill_candidates(store, limit).map_err(|error| error.to_string())
}

pub fn list_skill_candidates(
    store: &MemoryGraphStore,
    include_disabled: bool,
    limit: Option<usize>,
) -> Result<Vec<MemorySkillCandidate>, String> {
    crate::memory::skills::list_skill_candidates(store, include_disabled, limit)
        .map_err(|error| error.to_string())
}

pub fn update_skill_candidate(
    store: &MemoryGraphStore,
    request: MemorySkillCandidateUpdateRequest,
) -> Result<MemorySkillCandidateUpdateReceipt, String> {
    crate::memory::skills::update_skill_candidate(store, request).map_err(|error| error.to_string())
}

pub fn consolidate_research_bundle(
    store: &MemoryGraphStore,
    bundle: ResearchMemoryBundle,
) -> Result<ResearchMemoryConsolidationReceipt, String> {
    crate::memory::consolidation::consolidate_research_bundle(store, bundle)
        .map_err(|error| error.to_string())
}

pub fn consolidate_conversation_bundle(
    store: &MemoryGraphStore,
    bundle: ConversationMemoryBundle,
) -> Result<ConversationMemoryConsolidationReceipt, String> {
    crate::memory::consolidation::consolidate_conversation_bundle(store, bundle)
        .map_err(|error| error.to_string())
}


pub fn list_reconsolidation_candidates(
    store: &MemoryGraphStore,
    limit: Option<usize>,
    include_reprocessed: bool,
) -> Result<Vec<MemoryReconsolidationCandidate>, String> {
    crate::memory::consolidation::list_reconsolidation_candidates(store, limit, include_reprocessed)
        .map_err(|error| error.to_string())
}

pub fn reconsolidation_status(store: &MemoryGraphStore, limit: Option<usize>) -> Result<serde_json::Value, String> {
    let candidates = crate::memory::consolidation::list_reconsolidation_candidates(store, limit, false)
        .map_err(|error| error.to_string())?;
    Ok(json!({
        "pending_candidates": candidates.len(),
        "sample_node_ids": candidates.iter().take(12).map(|candidate| candidate.node.id.clone()).collect::<Vec<_>>(),
        "metadata_only": true,
    }))
}


fn memory_embedding_auto_index_enabled() -> bool {
    !matches!(
        std::env::var("ASTRA_MEMORY_EMBEDDING_AUTO_INDEX")
            .unwrap_or_else(|_| "true".to_string())
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

fn memory_embedding_auto_index_batch_size() -> usize {
    std::env::var("ASTRA_MEMORY_EMBEDDING_BATCH_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(24)
        .clamp(1, 256)
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn embedding_fallback_enabled() -> bool {
    !matches!(
        std::env::var("ASTRA_MEMORY_EMBEDDING_DISABLE_FALLBACK")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}
