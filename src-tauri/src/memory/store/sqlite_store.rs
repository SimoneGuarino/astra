use crate::memory::{
    config::MemoryConfig,
    consolidation::reconsolidation::MemoryReconsolidationCandidate,
    embeddings::cosine_similarity,
    errors::{MemoryError, MemoryResult},
    governance::is_retrieval_enabled_status,
    ids::new_memory_id,
    retrieval::vector::{MemoryVectorFilter, MemoryVectorHit},
    types::{
        now_ms, CreateMemoryEdgeRequest, CreateMemoryNodeRequest, MemoryActivation,
        MemoryActivationRequest, MemoryCanonicalReviewCandidate, MemoryCanonicalReviewRequest, MemoryChunk, MemoryDuplicateCandidate, MemoryDuplicateCandidateRequest, MemoryEdge, MemoryEmbeddingIndexStatus,
        MemoryEmbeddingRecord, MemoryGraphSnapshot, MemoryHybridQueryRequest, MemoryMergeNodesReceipt, MemoryMergeNodesRequest,
        MemoryHybridQueryResponse, MemoryNode, MemoryNodeGovernanceUpdateReceipt,
        MemoryHealthRepairAction, MemoryHealthRepairPlan, MemoryQualityDashboard, MemoryQualityGovernanceStats, MemoryQualityReconsolidationStats,
        MemoryQualityRetrievalStats, MemoryQualitySemanticStats, MemoryQualityTotals,
        MemoryNodeGovernanceUpdateRequest, MemoryQueryHit, MemoryQueryRequest,
        MemoryQueryResponse, MemorySkillCandidate, MemorySkillCandidateStatus, MemorySkillCandidateUpdateReceipt, MemorySkillCandidateUpdateRequest, MemoryVerificationStatus,
    },
};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::{collections::{HashMap, HashSet, VecDeque}, fs, path::PathBuf, sync::{Arc, Mutex}};

#[derive(Clone)]
pub struct MemoryGraphStore {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
    config: MemoryConfig,
}

impl MemoryGraphStore {
    pub fn new(config: MemoryConfig) -> Self {
        if let Some(parent) = config.sqlite_path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::create_dir_all(&config.journal_dir);
        let store = Self {
            path: config.sqlite_path.clone(),
            lock: Arc::new(Mutex::new(())),
            config,
        };
        if let Err(error) = store.initialize() {
            eprintln!("Astra memory store initialization failed: {error}");
        }
        store
    }

    pub fn status(&self) -> Value {
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let Ok(conn) = self.open_connection() else {
            return json!({"available": false, "reason": "connection_failed"});
        };
        let nodes = count_table(&conn, "memory_nodes").unwrap_or(0);
        let edges = count_table(&conn, "memory_edges").unwrap_or(0);
        let activations = count_table(&conn, "memory_activations").unwrap_or(0);
        let chunks = count_table(&conn, "memory_chunks").unwrap_or(0);
        let embeddings = count_table(&conn, "memory_embeddings").unwrap_or(0);
        let skill_candidates = count_table(&conn, "memory_skill_candidates").unwrap_or(0);
        json!({
            "available": true,
            "backend": "sqlite",
            "path": self.path.to_string_lossy(),
            "nodes": nodes,
            "edges": edges,
            "activations": activations,
            "chunks": chunks,
            "embeddings": embeddings,
            "skill_candidates": skill_candidates,
            "pending_embeddings": (chunks - embeddings).max(0),
            "vector_backend": "sqlite_vector_cache",
            "fts_enabled": true,
            "max_query_limit": self.config.max_query_limit,
            "max_activation_depth": self.config.max_activation_depth,
            "max_activation_nodes": self.config.max_activation_nodes,
        })
    }

    pub fn create_node(&self, request: CreateMemoryNodeRequest) -> MemoryResult<MemoryNode> {
        validate_non_empty("title", &request.title)?;
        validate_non_empty("summary", &request.summary)?;
        let now = now_ms();
        let node = MemoryNode {
            id: new_memory_id("node"),
            kind: request.kind,
            title: cap_text(request.title, 512),
            summary: cap_text(request.summary, 4096),
            content: request.content.map(|content| cap_text(content, 128 * 1024)),
            tags: request.tags.into_iter().map(|tag| cap_text(tag, 64)).take(32).collect(),
            source: request.source.map(|source| cap_text(source, 512)),
            confidence: clamp01(request.confidence),
            verification_status: request.verification_status,
            salience: clamp01(request.salience),
            created_at: now,
            updated_at: now,
            metadata: request.metadata,
        };
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        insert_node(&conn, &node)?;
        let chunk = MemoryChunk {
            id: new_memory_id("chunk"),
            node_id: node.id.clone(),
            text: node_search_text(&node),
            ordinal: 0,
            created_at: now,
            metadata: json!({"source": "node_text"}),
        };
        insert_chunk(&conn, &chunk)?;
        self.append_journal_locked("node_created", &node);
        Ok(node)
    }

    pub fn find_node_by_source(&self, source: &str) -> MemoryResult<Option<MemoryNode>> {
        validate_non_empty("source", source)?;
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut stmt = conn.prepare(
            "SELECT id, kind, title, summary, content, tags_json, source, confidence, verification_status, salience, created_at, updated_at, metadata_json
             FROM memory_nodes WHERE source = ?1 ORDER BY updated_at DESC LIMIT 1",
        )?;
        let node = stmt
            .query_row(params![source], row_to_node)
            .optional()?;
        Ok(node)
    }

    pub fn create_node_once_by_source(&self, request: CreateMemoryNodeRequest) -> MemoryResult<MemoryNode> {
        if let Some(source) = request.source.as_deref().map(str::trim).filter(|value| !value.is_empty()) {
            if let Some(existing) = self.find_node_by_source(source)? {
                return Ok(existing);
            }
        }
        self.create_node(request)
    }

    pub fn create_edge(&self, request: CreateMemoryEdgeRequest) -> MemoryResult<MemoryEdge> {
        validate_non_empty("from_node_id", &request.from_node_id)?;
        validate_non_empty("to_node_id", &request.to_node_id)?;
        if request.from_node_id == request.to_node_id {
            return Err(MemoryError::Validation("self edges are not allowed".into()));
        }
        let now = now_ms();
        let edge = MemoryEdge {
            id: new_memory_id("edge"),
            from_node_id: request.from_node_id,
            to_node_id: request.to_node_id,
            relation: request.relation,
            weight: clamp01(request.weight),
            confidence: clamp01(request.confidence),
            created_at: now,
            last_activated_at: None,
            activation_count: 0,
            metadata: request.metadata,
        };
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        ensure_node_exists(&conn, &edge.from_node_id)?;
        ensure_node_exists(&conn, &edge.to_node_id)?;
        insert_edge(&conn, &edge)?;
        self.append_journal_locked("edge_created", &edge);
        Ok(edge)
    }


    pub fn list_duplicate_candidates(
        &self,
        request: MemoryDuplicateCandidateRequest,
    ) -> MemoryResult<Vec<MemoryDuplicateCandidate>> {
        let limit = request.limit.clamp(1, 200);
        let min_score = clamp01(request.min_score).max(0.35);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut nodes = latest_nodes(&conn, 900)?;
        if !request.include_deprecated {
            nodes.retain(|node| is_retrieval_enabled_status(&node.verification_status));
        }
        if !request.kinds.is_empty() {
            let kinds = request
                .kinds
                .iter()
                .map(|kind| kind.as_str().to_string())
                .collect::<HashSet<_>>();
            nodes.retain(|node| kinds.contains(node.kind.as_str()));
        }

        let mut candidates = Vec::new();
        for left_index in 0..nodes.len() {
            for right_index in (left_index + 1)..nodes.len() {
                let left = &nodes[left_index];
                let right = &nodes[right_index];
                if left.kind != right.kind {
                    continue;
                }
                if merged_into(left).is_some() || merged_into(right).is_some() {
                    continue;
                }
                let scored = duplicate_score(left, right);
                if scored.score < min_score {
                    continue;
                }
                let (canonical_node, duplicate_node) = choose_canonical_node(left, right);
                candidates.push(MemoryDuplicateCandidate {
                    canonical_node,
                    duplicate_node,
                    score: scored.score,
                    reasons: scored.reasons,
                    shared_tags: scored.shared_tags,
                    metadata: json!({
                        "candidate_kind": "structural_duplicate_or_semantic_overlap",
                        "llm_advisory": false,
                        "requires_user_confirmation": true,
                        "metadata_only": true,
                    }),
                });
            }
        }
        candidates.sort_by(|left, right| right.score.partial_cmp(&left.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);
        Ok(candidates)
    }


    pub fn list_canonical_review_candidates(
        &self,
        request: MemoryCanonicalReviewRequest,
    ) -> MemoryResult<Vec<MemoryCanonicalReviewCandidate>> {
        let duplicate_request = MemoryDuplicateCandidateRequest {
            limit: request.limit.saturating_mul(3).clamp(1, 300),
            min_score: request.min_score.max(0.35),
            include_deprecated: request.include_deprecated,
            kinds: request.kinds.clone(),
        };
        let duplicates = self.list_duplicate_candidates(duplicate_request)?;
        let mut grouped: HashMap<String, Vec<MemoryDuplicateCandidate>> = HashMap::new();
        for candidate in duplicates {
            grouped
                .entry(candidate.canonical_node.id.clone())
                .or_default()
                .push(candidate);
        }

        let mut reviews = Vec::new();
        for (_, mut group) in grouped {
            group.sort_by(|left, right| right.score.partial_cmp(&left.score).unwrap_or(std::cmp::Ordering::Equal));
            let Some(first) = group.first() else { continue; };
            let target = first.canonical_node.clone();
            let candidate_nodes = group
                .iter()
                .take(8)
                .map(|candidate| candidate.duplicate_node.clone())
                .collect::<Vec<_>>();
            if candidate_nodes.is_empty() {
                continue;
            }
            let mut reasons = Vec::new();
            let mut shared_tags = HashSet::new();
            let mut weighted_score = 0.0f32;
            for candidate in group.iter().take(8) {
                weighted_score += candidate.score;
                for reason in candidate.reasons.iter() {
                    if !reasons.iter().any(|existing: &String| existing == reason) {
                        reasons.push(reason.clone());
                    }
                }
                for tag in candidate.shared_tags.iter() {
                    shared_tags.insert(tag.clone());
                }
            }
            let confidence = (weighted_score / candidate_nodes.len().max(1) as f32).clamp(0.0, 1.0);
            let rationale = if request.llm_assist {
                "Canonical review candidate prepared for LLM-assisted/user-governed review; structural grouping found semantically overlapping memories.".to_string()
            } else {
                "Canonical review candidate prepared from structural duplicate detection; user confirmation is required before merge.".to_string()
            };
            reviews.push(MemoryCanonicalReviewCandidate {
                id: new_memory_id("canon_review"),
                target_node: target.clone(),
                candidate_nodes,
                confidence,
                rationale,
                proposed_title: target.title.clone(),
                proposed_summary: target.summary.clone(),
                reasons,
                shared_tags: shared_tags.into_iter().collect(),
                metadata: json!({
                    "candidate_kind": "canonical_memory_review",
                    "llm_assist_requested": request.llm_assist,
                    "llm_advisory": false,
                    "requires_user_confirmation": true,
                    "merge_mode": "soft_canonicalization",
                    "metadata_only": true,
                }),
            });
        }
        reviews.sort_by(|left, right| right.confidence.partial_cmp(&left.confidence).unwrap_or(std::cmp::Ordering::Equal));
        reviews.truncate(request.limit.clamp(1, 120));
        Ok(reviews)
    }

    pub fn merge_nodes(&self, request: MemoryMergeNodesRequest) -> MemoryResult<MemoryMergeNodesReceipt> {
        validate_non_empty("target_node_id", &request.target_node_id)?;
        if request.source_node_ids.is_empty() {
            return Err(MemoryError::Validation("source_node_ids cannot be empty".into()));
        }
        let now = now_ms();
        let actor = request.actor.clone().unwrap_or_else(|| "user".into());
        let reason = request
            .reason
            .clone()
            .unwrap_or_else(|| "user_confirmed_memory_duplicate_merge".into());
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut target = node_by_id(&conn, &request.target_node_id)?;
        let mut merged = Vec::new();
        let mut created_edge_ids = Vec::new();
        let mut target_tags = target.tags.clone();
        let mut merged_sources = target
            .metadata
            .get("merged_from_node_ids")
            .and_then(Value::as_array)
            .map(|items| items.iter().filter_map(Value::as_str).map(ToOwned::to_owned).collect::<Vec<_>>())
            .unwrap_or_default();

        for source_id in request.source_node_ids.iter().map(|value| value.trim()).filter(|value| !value.is_empty()) {
            if source_id == target.id {
                continue;
            }
            let mut source = node_by_id(&conn, source_id)?;
            if source.kind != target.kind {
                return Err(MemoryError::Validation(format!(
                    "cannot merge nodes with different kinds: {} -> {}",
                    source.kind.as_str(),
                    target.kind.as_str()
                )));
            }
            if !merged_sources.iter().any(|existing| existing == &source.id) {
                merged_sources.push(source.id.clone());
            }
            for tag in source.tags.iter() {
                if !target_tags.iter().any(|existing| existing.eq_ignore_ascii_case(tag)) {
                    target_tags.push(cap_text(tag.clone(), 48));
                }
            }
            let edge = MemoryEdge {
                id: new_memory_id("edge"),
                from_node_id: source.id.clone(),
                to_node_id: target.id.clone(),
                relation: crate::memory::types::MemoryRelationKind::SameTopicAs,
                weight: 0.94,
                confidence: duplicate_score(&target, &source).score.max(0.7),
                created_at: now,
                last_activated_at: None,
                activation_count: 0,
                metadata: json!({
                    "semantic_relation": "merged_into",
                    "merge_actor": actor.clone(),
                    "merge_reason": reason.clone(),
                    "governance": "soft_merge_no_hard_delete",
                    "metadata_only": true,
                }),
            };
            insert_edge(&conn, &edge)?;
            created_edge_ids.push(edge.id.clone());

            let mut source_tags = source.tags.clone();
            for tag in ["merged_alias", "canonicalized"].iter() {
                if !source_tags.iter().any(|existing| existing == tag) {
                    source_tags.push((*tag).to_string());
                }
            }
            source_tags.truncate(32);
            let mut source_metadata = source.metadata.as_object().cloned().unwrap_or_default();
            source_metadata.insert("merged_into_node_id".into(), json!(target.id.clone()));
            source_metadata.insert("merged_at".into(), json!(now));
            source_metadata.insert("merge_actor".into(), json!(actor.clone()));
            source_metadata.insert("merge_reason".into(), json!(reason.clone()));
            source_metadata.insert("soft_merge".into(), json!(true));
            source.metadata = Value::Object(source_metadata);
            source.tags = source_tags;
            source.updated_at = now;
            if request.mark_sources_deprecated {
                source.verification_status = MemoryVerificationStatus::Deprecated;
                source.salience = (source.salience * 0.25).min(0.25);
            }
            conn.execute(
                "UPDATE memory_nodes
                 SET verification_status = ?1,
                     salience = ?2,
                     tags_json = ?3,
                     updated_at = ?4,
                     metadata_json = ?5
                 WHERE id = ?6",
                params![
                    source.verification_status.as_str(),
                    source.salience as f64,
                    serde_json::to_string(&source.tags)?,
                    source.updated_at,
                    serde_json::to_string(&source.metadata)?,
                    source.id,
                ],
            )?;
            merged.push(source.id.clone());
        }

        target_tags.truncate(32);
        let mut target_metadata = target.metadata.as_object().cloned().unwrap_or_default();
        target_metadata.insert("merged_from_node_ids".into(), json!(merged_sources));
        target_metadata.insert("last_merge_at".into(), json!(now));
        target_metadata.insert("last_merge_actor".into(), json!(actor.clone()));
        target_metadata.insert("last_merge_reason".into(), json!(reason.clone()));
        target_metadata.insert("canonical_memory_node".into(), json!(true));
        target.metadata = Value::Object(target_metadata);
        target.tags = target_tags;
        target.salience = (target.salience + 0.08).min(1.0);
        target.updated_at = now;
        conn.execute(
            "UPDATE memory_nodes
             SET salience = ?1,
                 tags_json = ?2,
                 updated_at = ?3,
                 metadata_json = ?4
             WHERE id = ?5",
            params![
                target.salience as f64,
                serde_json::to_string(&target.tags)?,
                target.updated_at,
                serde_json::to_string(&target.metadata)?,
                target.id,
            ],
        )?;

        let activation = MemoryActivation {
            id: new_memory_id("activation"),
            request_id: None,
            root_query: "memory_merge_duplicate_resolution".into(),
            activated_node_ids: std::iter::once(target.id.clone()).chain(merged.iter().cloned()).collect(),
            activated_edge_ids: created_edge_ids.clone(),
            intensity: json!({"target": target.id.clone(), "merged": merged.clone()}),
            created_at: now,
            metadata: json!({"source": "memory_merge", "metadata_only": true}),
        };
        insert_activation(&conn, &activation)?;
        self.append_journal_locked("memory_nodes_merged", &json!({
            "target_node_id": target.id.clone(),
            "merged_node_ids": merged.clone(),
            "created_edge_ids": created_edge_ids.clone(),
            "mark_sources_deprecated": request.mark_sources_deprecated,
            "metadata_only": true,
        }));
        Ok(MemoryMergeNodesReceipt {
            accepted: true,
            reason: "memory nodes soft-merged; original evidence nodes retained and linked to canonical target".into(),
            target_node: target,
            merged_node_ids: activation.activated_node_ids.iter().skip(1).cloned().collect(),
            created_edge_ids,
            activation: Some(activation),
            metadata: json!({
                "hard_delete_performed": false,
                "merge_mode": "soft_canonicalization",
                "requires_embeddings_maintenance": true,
                "metadata_only": true,
            }),
        })
    }

    pub fn update_node_governance(
        &self,
        request: MemoryNodeGovernanceUpdateRequest,
    ) -> MemoryResult<MemoryNodeGovernanceUpdateReceipt> {
        validate_non_empty("node_id", &request.node_id)?;
        let now = now_ms();
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut node = node_by_id(&conn, &request.node_id)?;

        if let Some(status) = request.verification_status.clone() {
            node.verification_status = status;
        }
        if let Some(confidence) = request.confidence {
            node.confidence = clamp01(confidence);
        }
        if let Some(salience) = request.salience {
            node.salience = clamp01(salience);
        }

        let mut tags = node.tags.clone();
        for tag in request.add_tags.iter().map(|tag| tag.trim()).filter(|tag| !tag.is_empty()) {
            let value = cap_text(tag.to_string(), 48);
            if !tags.iter().any(|existing| existing.eq_ignore_ascii_case(&value)) {
                tags.push(value);
            }
        }
        if !request.remove_tags.is_empty() {
            let remove = request
                .remove_tags
                .iter()
                .map(|tag| tag.trim().to_ascii_lowercase())
                .collect::<HashSet<_>>();
            tags.retain(|tag| !remove.contains(&tag.to_ascii_lowercase()));
        }
        tags.truncate(24);
        node.tags = tags;

        let mut metadata = node.metadata.as_object().cloned().unwrap_or_default();
        metadata.insert("governance_updated_at".into(), json!(now));
        metadata.insert(
            "governance_actor".into(),
            json!(request.actor.unwrap_or_else(|| "user".into())),
        );
        if let Some(reason) = request.reason.as_ref().map(|value| value.trim()).filter(|value| !value.is_empty()) {
            metadata.insert("governance_reason".into(), json!(cap_text(reason.to_string(), 512)));
        }
        if request.metadata.is_object() {
            metadata.insert("governance_metadata".into(), request.metadata.clone());
        }
        node.metadata = Value::Object(metadata);
        node.updated_at = now;

        conn.execute(
            "UPDATE memory_nodes
             SET confidence = ?1,
                 verification_status = ?2,
                 salience = ?3,
                 tags_json = ?4,
                 updated_at = ?5,
                 metadata_json = ?6
             WHERE id = ?7",
            params![
                node.confidence as f64,
                node.verification_status.as_str(),
                node.salience as f64,
                serde_json::to_string(&node.tags)?,
                node.updated_at,
                serde_json::to_string(&node.metadata)?,
                node.id,
            ],
        )?;
        self.append_journal_locked("node_governance_updated", &json!({
            "node_id": node.id,
            "verification_status": node.verification_status.as_str(),
            "confidence": node.confidence,
            "salience": node.salience,
            "reason": request.reason,
            "metadata_only": true,
        }));
        Ok(MemoryNodeGovernanceUpdateReceipt {
            accepted: true,
            reason: "memory node governance state updated through user-visible governed control".into(),
            node: node.clone(),
            metadata: json!({
                "hard_delete_performed": false,
                "retrieval_enabled": is_retrieval_enabled_status(&node.verification_status),
                "governance": "deprecated_and_contradicted_nodes_excluded_unless_requested",
                "metadata_only": true,
            }),
        })
    }

    pub fn query(&self, request: MemoryQueryRequest) -> MemoryResult<MemoryQueryResponse> {
        let limit = request.limit.clamp(1, self.config.max_query_limit);
        let query = request.query.trim().to_string();
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut hits = if query.is_empty() {
            latest_nodes(&conn, limit)?
                .into_iter()
                .map(|node| MemoryQueryHit { node, score: 0.35, reasons: vec!["latest".into()] })
                .collect::<Vec<_>>()
        } else {
            lexical_query(&conn, &query, limit * 2)?
        };
        if !request.kinds.is_empty() {
            let accepted = request.kinds.iter().map(|kind| kind.as_str().to_string()).collect::<HashSet<_>>();
            hits.retain(|hit| accepted.contains(hit.node.kind.as_str()));
        }
        if !request.include_deprecated {
            hits.retain(|hit| is_retrieval_enabled_status(&hit.node.verification_status));
        }
        hits.sort_by(|left, right| right.score.partial_cmp(&left.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(limit);
        let related_edges = if request.include_edges {
            let ids = hits.iter().map(|hit| hit.node.id.clone()).collect::<Vec<_>>();
            edges_for_nodes(&conn, &ids, limit * 3)?
        } else {
            Vec::new()
        };
        Ok(MemoryQueryResponse { hits, related_edges })
    }

    pub fn activate(&self, request: MemoryActivationRequest) -> MemoryResult<MemoryActivation> {
        let max_depth = request.max_depth.min(self.config.max_activation_depth);
        let max_nodes = request.max_nodes.clamp(1, self.config.max_activation_nodes);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let (activated_node_ids, activated_edge_ids, intensity) = propagate_activation(
            &conn,
            &request.seed_node_ids,
            max_depth,
            max_nodes,
        )?;
        let now = now_ms();
        for edge_id in &activated_edge_ids {
            let _ = conn.execute(
                "UPDATE memory_edges SET last_activated_at = ?1, activation_count = activation_count + 1 WHERE id = ?2",
                params![now, edge_id],
            );
        }
        let activation = MemoryActivation {
            id: new_memory_id("activation"),
            request_id: request.request_id,
            root_query: cap_text(request.root_query, 1024),
            activated_node_ids,
            activated_edge_ids,
            intensity,
            created_at: now,
            metadata: request.metadata,
        };
        insert_activation(&conn, &activation)?;
        self.append_journal_locked("activation_recorded", &activation);
        Ok(activation)
    }

    pub fn recent_activations(&self, limit: usize) -> MemoryResult<Vec<MemoryActivation>> {
        let limit = limit.clamp(1, 100);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut stmt = conn.prepare(
            "SELECT id, request_id, root_query, activated_node_ids, activated_edge_ids, intensity_json, created_at, metadata_json
             FROM memory_activations ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], row_to_activation)?;
        let mut activations = Vec::new();
        for row in rows {
            activations.push(row?);
        }
        Ok(activations)
    }

    pub fn snapshot(&self, limit: usize) -> MemoryResult<MemoryGraphSnapshot> {
        let limit = limit.clamp(1, 500);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let nodes = latest_nodes(&conn, limit)?;
        let ids = nodes.iter().map(|node| node.id.clone()).collect::<Vec<_>>();
        let edges = edges_for_nodes(&conn, &ids, limit * 4)?;
        let activations = {
            let mut stmt = conn.prepare(
                "SELECT id, request_id, root_query, activated_node_ids, activated_edge_ids, intensity_json, created_at, metadata_json
                 FROM memory_activations ORDER BY created_at DESC LIMIT 25",
            )?;
            let rows = stmt.query_map([], row_to_activation)?;
            let mut values = Vec::new();
            for row in rows { values.push(row?); }
            values
        };
        Ok(MemoryGraphSnapshot { nodes, edges, activations })
    }


    pub fn list_chunks_for_embedding(&self, limit: usize, force: bool) -> MemoryResult<Vec<MemoryChunk>> {
        let limit = limit.clamp(1, 2_000);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let sql = if force {
            "SELECT id, node_id, text, ordinal, created_at, metadata_json FROM memory_chunks ORDER BY created_at DESC LIMIT ?1"
        } else {
            "SELECT c.id, c.node_id, c.text, c.ordinal, c.created_at, c.metadata_json
             FROM memory_chunks c
             LEFT JOIN memory_embeddings e ON e.chunk_id = c.id
             WHERE e.chunk_id IS NULL
             ORDER BY c.created_at DESC
             LIMIT ?1"
        };
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![limit as i64], row_to_chunk)?;
        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row?);
        }
        Ok(chunks)
    }

    pub fn upsert_embedding_record(&self, record: MemoryEmbeddingRecord) -> MemoryResult<()> {
        if record.vector.is_empty() {
            return Err(MemoryError::Validation("embedding vector is empty".into()));
        }
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        ensure_node_exists(&conn, &record.node_id)?;
        let chunk_exists: Option<String> = conn
            .query_row(
                "SELECT id FROM memory_chunks WHERE id = ?1 AND node_id = ?2",
                params![&record.chunk_id, &record.node_id],
                |row| row.get(0),
            )
            .optional()?;
        if chunk_exists.is_none() {
            return Err(MemoryError::Validation(format!(
                "memory chunk not found for embedding: {}",
                record.chunk_id
            )));
        }
        conn.execute(
            "INSERT INTO memory_embeddings (chunk_id, node_id, model, dimensions, vector_json, created_at, updated_at, metadata_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
             ON CONFLICT(chunk_id) DO UPDATE SET
                node_id = excluded.node_id,
                model = excluded.model,
                dimensions = excluded.dimensions,
                vector_json = excluded.vector_json,
                updated_at = excluded.updated_at,
                metadata_json = excluded.metadata_json",
            params![
                &record.chunk_id,
                &record.node_id,
                &record.model,
                record.dimensions as i64,
                serde_json::to_string(&record.vector)?,
                record.created_at,
                record.updated_at,
                serde_json::to_string(&record.metadata)?,
            ],
        )?;
        self.append_journal_locked("embedding_upserted", &serde_json::json!({
            "chunk_id": record.chunk_id,
            "node_id": record.node_id,
            "model": record.model,
            "dimensions": record.dimensions,
            "metadata_only": true,
        }));
        Ok(())
    }

    pub fn embedding_status(&self) -> MemoryResult<MemoryEmbeddingIndexStatus> {
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let total_chunks = count_table(&conn, "memory_chunks")? as usize;
        let embedded_chunks = count_table(&conn, "memory_embeddings")? as usize;
        let last_indexed_at = conn
            .query_row("SELECT MAX(updated_at) FROM memory_embeddings", [], |row| row.get::<_, Option<i64>>(0))
            .optional()?
            .flatten();
        let (provider, dimensions) = conn
            .query_row(
                "SELECT model, dimensions FROM memory_embeddings ORDER BY updated_at DESC LIMIT 1",
                [],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize)),
            )
            .optional()?
            .unwrap_or_else(|| ("stable-local-hash-v1".into(), 384));
        Ok(MemoryEmbeddingIndexStatus {
            backend: "sqlite_vector_cache".into(),
            provider,
            dimensions,
            embedded_chunks,
            total_chunks,
            pending_chunks: total_chunks.saturating_sub(embedded_chunks),
            last_indexed_at,
            metadata: json!({
                "source_of_truth": "sqlite_memory_graph",
                "vector_index_role": "advisory_retrieval_index",
                "metadata_only": true,
            }),
        })
    }

    pub fn vector_search(
        &self,
        query_embedding: &[f32],
        filter: MemoryVectorFilter,
        limit: usize,
    ) -> MemoryResult<Vec<MemoryVectorHit>> {
        if query_embedding.is_empty() {
            return Ok(Vec::new());
        }
        let limit = limit.clamp(1, self.config.max_query_limit);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut stmt = conn.prepare(
            "SELECT e.chunk_id, e.node_id, e.model, e.vector_json, n.kind, n.confidence, n.verification_status
             FROM memory_embeddings e
             JOIN memory_nodes n ON n.id = e.node_id
             ORDER BY e.updated_at DESC",
        )?;
        let accepted_kinds = filter
            .node_kinds
            .iter()
            .map(|kind| kind.as_str().to_string())
            .collect::<HashSet<_>>();
        let rows = stmt.query_map([], |row| {
            let vector_json: String = row.get(3)?;
            let vector = serde_json::from_str::<Vec<f32>>(&vector_json).unwrap_or_default();
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                vector,
                row.get::<_, String>(4)?,
                row.get::<_, f64>(5)? as f32,
                row.get::<_, String>(6)?,
            ))
        })?;
        let mut hits = Vec::new();
        for row in rows {
            let (chunk_id, node_id, model, vector, kind, confidence, verification_status) = row?;
            let status = serde_json::from_value::<MemoryVerificationStatus>(json!(verification_status)).unwrap_or_default();
            if !is_retrieval_enabled_status(&status) {
                continue;
            }
            if !accepted_kinds.is_empty() && !accepted_kinds.contains(&kind) {
                continue;
            }
            if let Some(min_confidence) = filter.min_confidence {
                if confidence < min_confidence {
                    continue;
                }
            }
            let score = cosine_similarity(query_embedding, &vector);
            if score <= 0.0 {
                continue;
            }
            hits.push(MemoryVectorHit {
                node_id,
                chunk_id,
                score,
                model: Some(model),
            });
        }
        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(hits)
    }

    pub fn hybrid_query(&self, request: MemoryHybridQueryRequest, query_embedding: Option<Vec<f32>>) -> MemoryResult<MemoryHybridQueryResponse> {
        let limit = request.limit.clamp(1, self.config.max_query_limit);
        let lexical = self.query(MemoryQueryRequest {
            query: request.query.clone(),
            kinds: request.kinds.clone(),
            limit: limit * 2,
            include_edges: true,
            include_deprecated: request.include_deprecated,
        })?;
        let vector_hits = if let Some(embedding) = query_embedding.as_ref() {
            self.vector_search(
                embedding,
                MemoryVectorFilter {
                    node_kinds: request.kinds.clone(),
                    min_confidence: Some(0.15),
                },
                limit * 3,
            )?
        } else {
            Vec::new()
        };

        let mut ranked: HashMap<String, MemoryQueryHit> = HashMap::new();
        for hit in lexical.hits {
            let entry = ranked.entry(hit.node.id.clone()).or_insert(MemoryQueryHit {
                node: hit.node,
                score: 0.0,
                reasons: Vec::new(),
            });
            entry.score += hit.score.max(0.0) * request.lexical_weight.max(0.0);
            entry.reasons.extend(hit.reasons);
            entry.reasons.push("hybrid_lexical_component".into());
        }

        if !vector_hits.is_empty() {
            let node_ids = vector_hits.iter().map(|hit| hit.node_id.clone()).collect::<Vec<_>>();
            let nodes = nodes_by_ids(&self.open_connection()?, &node_ids)?;
            for vector_hit in vector_hits {
                let Some(node) = nodes.get(&vector_hit.node_id).cloned() else { continue; };
                if !request.include_deprecated && !is_retrieval_enabled_status(&node.verification_status) {
                    continue;
                }
                let entry = ranked.entry(node.id.clone()).or_insert(MemoryQueryHit {
                    node,
                    score: 0.0,
                    reasons: Vec::new(),
                });
                entry.score += vector_hit.score.max(0.0) * request.vector_weight.max(0.0);
                entry.reasons.push(format!("vector_similarity:{:.3}", vector_hit.score));
                entry.reasons.push("hybrid_vector_component".into());
            }
        }

        for hit in ranked.values_mut() {
            hit.score += hit.node.salience * request.graph_weight.max(0.0);
            hit.score += hit.node.confidence * 0.08;
            match hit.node.verification_status.as_str() {
                "system_verified" | "user_confirmed" => {
                    hit.score += 0.08;
                    hit.reasons.push("verified_memory_boost".into());
                }
                "llm_inferred" => {
                    hit.score *= 0.92;
                    hit.reasons.push("llm_inferred_memory_weighted".into());
                }
                "unverified" => {
                    hit.score *= 0.84;
                    hit.reasons.push("unverified_memory_weighted".into());
                }
                "deprecated" | "contradicted" => {
                    hit.score *= 0.05;
                    hit.reasons.push("governance_suppressed_memory".into());
                }
                _ => {}
            }
        }

        let mut hits = ranked.into_values().collect::<Vec<_>>();
        hits.sort_by(|left, right| right.score.partial_cmp(&left.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(limit);
        let ids = hits.iter().map(|hit| hit.node.id.clone()).collect::<Vec<_>>();
        let related_edges = {
            let _guard = self.lock.lock().expect("memory graph mutex poisoned");
            let conn = self.open_connection()?;
            edges_for_nodes(&conn, &ids, limit * 4)?
        };
        let embedding_status = self.embedding_status()?;
        Ok(MemoryHybridQueryResponse {
            hits,
            related_edges,
            embedding_status,
            metadata: json!({
                "retrieval_mode": "hybrid_lexical_vector_graph",
                "vector_backend": "sqlite_vector_cache",
                "vector_available": query_embedding.is_some(),
                "metadata_only": true,
            }),
        })
    }


    pub fn extract_skill_candidates_from_memory(&self, limit: usize) -> MemoryResult<Vec<MemorySkillCandidate>> {
        let limit = limit.clamp(1, 250);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut stmt = conn.prepare(
            "SELECT id, kind, title, summary, content, tags_json, source, confidence, verification_status, salience, created_at, updated_at, metadata_json
             FROM memory_nodes
             WHERE kind IN ('procedure', 'workflow', 'code_pattern')
               AND verification_status NOT IN ('deprecated', 'contradicted')
             ORDER BY salience DESC, confidence DESC, updated_at DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![limit as i64], row_to_node)?;
        let mut candidates = Vec::new();
        for row in rows {
            let node = row?;
            let already_exists: Option<String> = conn
                .query_row(
                    "SELECT id FROM memory_skill_candidates WHERE source_node_id = ?1 LIMIT 1",
                    params![&node.id],
                    |row| row.get(0),
                )
                .optional()?;
            if already_exists.is_some() {
                continue;
            }
            let candidate = crate::memory::skills::candidate_from_node(&node, candidates.len());
            insert_skill_candidate(&conn, &candidate)?;
            self.append_journal_locked("skill_candidate_created", &candidate);
            candidates.push(candidate);
        }
        Ok(candidates)
    }

    pub fn list_skill_candidates(&self, include_disabled: bool, limit: usize) -> MemoryResult<Vec<MemorySkillCandidate>> {
        let limit = limit.clamp(1, 250);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let sql = if include_disabled {
            "SELECT id, title, summary, source_node_id, status, confidence, salience, trigger_hints_json, required_tools_json, risk_level, created_at, updated_at, approved_by, approved_at, metadata_json
             FROM memory_skill_candidates ORDER BY updated_at DESC LIMIT ?1"
        } else {
            "SELECT id, title, summary, source_node_id, status, confidence, salience, trigger_hints_json, required_tools_json, risk_level, created_at, updated_at, approved_by, approved_at, metadata_json
             FROM memory_skill_candidates WHERE status NOT IN ('disabled', 'deprecated') ORDER BY updated_at DESC LIMIT ?1"
        };
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params![limit as i64], row_to_skill_candidate)?;
        let mut values = Vec::new();
        for row in rows {
            values.push(row?);
        }
        Ok(values)
    }

    pub fn update_skill_candidate(
        &self,
        request: MemorySkillCandidateUpdateRequest,
    ) -> MemoryResult<MemorySkillCandidateUpdateReceipt> {
        validate_non_empty("candidate_id", &request.candidate_id)?;
        let now = now_ms();
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut candidate = skill_candidate_by_id(&conn, &request.candidate_id)?;

        if let Some(status) = request.status.clone() {
            candidate.status = status;
            if matches!(candidate.status, MemorySkillCandidateStatus::Approved) {
                candidate.approved_by = Some(request.approved_by.clone().unwrap_or_else(|| "user".into()));
                candidate.approved_at = Some(now);
            }
        }
        if let Some(confidence) = request.confidence {
            candidate.confidence = clamp01(confidence);
        }
        if let Some(salience) = request.salience {
            candidate.salience = clamp01(salience);
        }
        if let Some(required_tools) = request.required_tools.clone() {
            candidate.required_tools = normalize_small_list(required_tools, 12, 64);
        }
        if let Some(risk_level) = request.risk_level.as_ref().map(|value| value.trim()).filter(|value| !value.is_empty()) {
            candidate.risk_level = cap_text(risk_level.to_string(), 64);
        }

        for hint in normalize_small_list(request.add_trigger_hints.clone(), 16, 120) {
            if !candidate.trigger_hints.iter().any(|existing| existing.eq_ignore_ascii_case(&hint)) {
                candidate.trigger_hints.push(hint);
            }
        }
        if !request.remove_trigger_hints.is_empty() {
            let remove = request.remove_trigger_hints.iter().map(|value| value.trim().to_ascii_lowercase()).collect::<HashSet<_>>();
            candidate.trigger_hints.retain(|hint| !remove.contains(&hint.to_ascii_lowercase()));
        }
        candidate.trigger_hints.truncate(16);

        let mut metadata = candidate.metadata.as_object().cloned().unwrap_or_default();
        metadata.insert("governance_updated_at".into(), json!(now));
        if let Some(reason) = request.reason.as_ref().map(|value| value.trim()).filter(|value| !value.is_empty()) {
            metadata.insert("governance_reason".into(), json!(cap_text(reason.to_string(), 512)));
        }
        if request.metadata.is_object() {
            metadata.insert("governance_metadata".into(), request.metadata.clone());
        }
        candidate.metadata = Value::Object(metadata);
        candidate.updated_at = now;

        conn.execute(
            "UPDATE memory_skill_candidates
             SET status = ?1,
                 confidence = ?2,
                 salience = ?3,
                 trigger_hints_json = ?4,
                 required_tools_json = ?5,
                 risk_level = ?6,
                 updated_at = ?7,
                 approved_by = ?8,
                 approved_at = ?9,
                 metadata_json = ?10
             WHERE id = ?11",
            params![
                candidate.status.as_str(),
                candidate.confidence as f64,
                candidate.salience as f64,
                serde_json::to_string(&candidate.trigger_hints)?,
                serde_json::to_string(&candidate.required_tools)?,
                candidate.risk_level,
                candidate.updated_at,
                candidate.approved_by,
                candidate.approved_at,
                serde_json::to_string(&candidate.metadata)?,
                candidate.id,
            ],
        )?;
        self.append_journal_locked("skill_candidate_governance_updated", &json!({
            "candidate_id": candidate.id,
            "status": candidate.status.as_str(),
            "confidence": candidate.confidence,
            "salience": candidate.salience,
            "metadata_only": true,
        }));

        Ok(MemorySkillCandidateUpdateReceipt {
            accepted: true,
            reason: "procedural skill candidate governance updated; candidate remains bounded by runtime policy and does not grant autonomous execution".into(),
            candidate,
            activation: None,
            metadata: json!({
                "execution_enabled": false,
                "approval_required_before_execution": true,
                "metadata_only": true,
            }),
        })
    }


    pub fn list_reconsolidation_candidates(
        &self,
        limit: usize,
        include_reprocessed: bool,
    ) -> MemoryResult<Vec<MemoryReconsolidationCandidate>> {
        let limit = limit.clamp(1, 250);
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut stmt = conn.prepare(
            "SELECT id, kind, title, summary, content, tags_json, source, confidence, verification_status, salience, created_at, updated_at, metadata_json
             FROM memory_nodes
             WHERE kind = 'conversation_turn'
               AND verification_status NOT IN ('deprecated', 'contradicted')
             ORDER BY updated_at DESC
             LIMIT ?1",
        )?;
        let rows = stmt.query_map(params![(limit * 4) as i64], row_to_node)?;
        let mut out = Vec::new();
        for row in rows {
            let node = row?;
            if !include_reprocessed && node.metadata.get("reconsolidated_at").is_some() {
                continue;
            }
            let semantic_count = node
                .metadata
                .get("semantic_atom_count")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let has_episode_tag = node.tags.iter().any(|tag| tag.eq_ignore_ascii_case("episode_only"));
            let extractor = node
                .metadata
                .get("extractor")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if semantic_count > 0 && !has_episode_tag && extractor != "fallback_episode_only" {
                continue;
            }
            let (user_message, assistant_answer) = extract_turn_messages_from_node(&node);
            if user_message.trim().is_empty() {
                continue;
            }
            out.push(MemoryReconsolidationCandidate {
                node,
                reason: if has_episode_tag || semantic_count == 0 {
                    "conversation turn has no semantic atoms or was captured as episode_only".into()
                } else {
                    "conversation turn selected for semantic reconsolidation".into()
                },
                user_message,
                assistant_answer,
            });
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    pub fn mark_node_reconsolidated(
        &self,
        node_id: &str,
        semantic_atom_count: usize,
        created_node_ids: &[String],
        reason: &str,
    ) -> MemoryResult<()> {
        validate_non_empty("node_id", node_id)?;
        let now = now_ms();
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let mut node = node_by_id(&conn, node_id)?;
        let mut metadata = node.metadata.as_object().cloned().unwrap_or_default();
        metadata.insert("reconsolidated_at".into(), json!(now));
        metadata.insert("reconsolidation_semantic_atom_count".into(), json!(semantic_atom_count));
        metadata.insert("reconsolidation_created_node_ids".into(), json!(created_node_ids));
        metadata.insert("reconsolidation_reason".into(), json!(reason));
        metadata.insert("reconsolidation_version".into(), json!(1));
        node.metadata = Value::Object(metadata);
        node.updated_at = now;
        conn.execute(
            "UPDATE memory_nodes SET updated_at = ?1, metadata_json = ?2 WHERE id = ?3",
            params![node.updated_at, serde_json::to_string(&node.metadata)?, node.id],
        )?;
        self.append_journal_locked("node_reconsolidated", &json!({
            "node_id": node_id,
            "semantic_atom_count": semantic_atom_count,
            "created_node_ids": created_node_ids,
            "reason": reason,
            "metadata_only": true,
        }));
        Ok(())
    }


    pub fn quality_dashboard(&self) -> MemoryResult<MemoryQualityDashboard> {
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        let nodes = count_table(&conn, "memory_nodes")? as usize;
        let edges = count_table(&conn, "memory_edges")? as usize;
        let chunks = count_table(&conn, "memory_chunks")? as usize;
        let activations = count_table(&conn, "memory_activations")? as usize;
        let skill_candidates = count_table(&conn, "memory_skill_candidates")? as usize;
        let embeddings = embedding_status_locked(&conn)?;

        let conversation_turn_nodes = count_where(&conn, "memory_nodes", "kind = 'conversation_turn'")? as usize;
        let episode_only_nodes = count_where(
            &conn,
            "memory_nodes",
            "kind = 'conversation_turn' AND tags_json LIKE '%episode_only%'",
        )? as usize;
        let semantic_nodes = count_where(
            &conn,
            "memory_nodes",
            "kind IN ('claim','entity','user_preference','procedure','decision','concept','workflow','research_finding','research_topic','code_pattern','fix')",
        )? as usize;
        let reconsolidated_nodes = count_where(&conn, "memory_nodes", "metadata_json LIKE '%reconsolidated_at%'")? as usize;
        let pending_reconsolidation = count_where(
            &conn,
            "memory_nodes",
            "kind = 'conversation_turn' AND verification_status NOT IN ('deprecated', 'contradicted') AND (metadata_json NOT LIKE '%reconsolidated_at%') AND (tags_json LIKE '%episode_only%' OR metadata_json LIKE '%\"semantic_atom_count\":0%' OR metadata_json LIKE '%fallback_episode_only%')",
        )? as usize;

        let unverified = count_where(&conn, "memory_nodes", "verification_status = 'unverified'")? as usize;
        let llm_inferred = count_where(&conn, "memory_nodes", "verification_status = 'llm_inferred'")? as usize;
        let user_confirmed = count_where(&conn, "memory_nodes", "verification_status = 'user_confirmed'")? as usize;
        let system_verified = count_where(&conn, "memory_nodes", "verification_status = 'system_verified'")? as usize;
        let contradicted = count_where(&conn, "memory_nodes", "verification_status = 'contradicted'")? as usize;
        let deprecated = count_where(&conn, "memory_nodes", "verification_status = 'deprecated'")? as usize;

        let (average_confidence, average_salience) = conn
            .query_row(
                "SELECT COALESCE(AVG(confidence), 0.0), COALESCE(AVG(salience), 0.0) FROM memory_nodes",
                [],
                |row| Ok((row.get::<_, f64>(0)? as f32, row.get::<_, f64>(1)? as f32)),
            )
            .unwrap_or((0.0, 0.0));

        let mut recent_activations = 0usize;
        let mut activation_nodes_total = 0usize;
        let last_activation_at = conn
            .query_row("SELECT MAX(created_at) FROM memory_activations", [], |row| row.get::<_, Option<i64>>(0))
            .optional()?
            .flatten();
        let mut stmt = conn.prepare(
            "SELECT activated_node_ids FROM memory_activations ORDER BY created_at DESC LIMIT 100",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            recent_activations += 1;
            let text = row?;
            let node_ids = serde_json::from_str::<Vec<String>>(&text).unwrap_or_default();
            activation_nodes_total += node_ids.len();
        }
        let average_activation_nodes = if recent_activations == 0 {
            0.0
        } else {
            activation_nodes_total as f32 / recent_activations as f32
        };

        let semantic_ratio = if nodes == 0 {
            0.0
        } else {
            semantic_nodes as f32 / nodes as f32
        };
        let embedding_ratio = if embeddings.total_chunks == 0 {
            1.0
        } else {
            embeddings.embedded_chunks as f32 / embeddings.total_chunks as f32
        };
        let reconsolidation_penalty = if conversation_turn_nodes == 0 {
            0.0
        } else {
            pending_reconsolidation as f32 / conversation_turn_nodes as f32
        };
        let deprecated_penalty = if nodes == 0 {
            0.0
        } else {
            (deprecated + contradicted) as f32 / nodes as f32
        };
        let verified_ratio = if nodes == 0 {
            0.0
        } else {
            (user_confirmed + system_verified) as f32 / nodes as f32
        };
        let score = clamp_quality_score(
            0.28 * semantic_ratio
                + 0.26 * embedding_ratio
                + 0.18 * (1.0 - reconsolidation_penalty).max(0.0)
                + 0.12 * (1.0 - deprecated_penalty).max(0.0)
                + 0.10 * average_salience.clamp(0.0, 1.0)
                + 0.06 * verified_ratio,
        );
        let status = if !self.path.exists() {
            "unavailable"
        } else if score >= 0.76 && embeddings.pending_chunks == 0 && pending_reconsolidation == 0 {
            "healthy"
        } else if score >= 0.48 {
            "degraded"
        } else {
            "needs_attention"
        }
        .to_string();

        let mut warnings = Vec::new();
        let mut recommendations = Vec::new();
        if embeddings.pending_chunks > 0 {
            warnings.push(format!("{} memory chunks are not embedded yet", embeddings.pending_chunks));
            recommendations.push("Run automatic embedding maintenance or keep ASTRA_MEMORY_EMBEDDING_AUTO_INDEX enabled".into());
        }
        if pending_reconsolidation > 0 {
            warnings.push(format!("{} conversation memories are still episode-only or semantically weak", pending_reconsolidation));
            recommendations.push("Run memory re-consolidation to distill semantic atoms from raw conversation turns".into());
        }
        if semantic_ratio < 0.25 && nodes > 10 {
            warnings.push("semantic memory ratio is low; retrieval may be noisy".into());
            recommendations.push("Prefer semantic consolidation and canonical nodes before increasing autonomy".into());
        }
        if contradicted + deprecated > 0 {
            warnings.push(format!("{} memories are contradicted or deprecated", contradicted + deprecated));
            recommendations.push("Review contradicted/deprecated nodes and merge or correct related memories".into());
        }
        if recent_activations == 0 && nodes > 0 {
            warnings.push("memory graph has nodes but no recent activations".into());
            recommendations.push("Verify that LLM-integrated memory retrieval is active in chat responses".into());
        }
        let embedding_provider = std::env::var("ASTRA_MEMORY_EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| "stable_hash".into())
            .trim()
            .to_ascii_lowercase();
        if !matches!(embedding_provider.as_str(), "ollama" | "ollama_embed" | "ollama_embeddings") {
            warnings.push("memory embeddings are using the deterministic stable-hash fallback, not a real semantic embedding model".into());
            recommendations.push("For production-like RAG quality, set ASTRA_MEMORY_EMBEDDING_PROVIDER=ollama and ASTRA_MEMORY_EMBEDDING_MODEL=nomic-embed-text or an equivalent local embedding model".into());
        }

        let summary = match status.as_str() {
            "healthy" => "Memory graph is healthy: semantic coverage, embeddings and reconsolidation are within expected bounds.",
            "degraded" => "Memory graph is usable but needs maintenance before increasing autonomy.",
            "needs_attention" => "Memory graph needs attention: retrieval quality may be noisy or incomplete.",
            _ => "Memory graph quality could not be fully evaluated.",
        }
        .to_string();

        let repair_plan = build_memory_health_repair_plan(
            nodes,
            semantic_nodes,
            semantic_ratio,
            episode_only_nodes,
            pending_reconsolidation,
            embeddings.pending_chunks,
            contradicted + deprecated,
            duplicate_review_pressure(&conn).unwrap_or(0),
            canonical_review_pressure(&conn).unwrap_or(0),
            score,
            &status,
        );

        Ok(MemoryQualityDashboard {
            schema_version: 1,
            generated_at: now_ms(),
            status,
            score,
            summary,
            totals: MemoryQualityTotals { nodes, edges, chunks, activations, skill_candidates },
            semantic: MemoryQualitySemanticStats {
                semantic_nodes,
                episode_only_nodes,
                conversation_turn_nodes,
                semantic_ratio,
                average_confidence,
                average_salience,
            },
            governance: MemoryQualityGovernanceStats {
                unverified,
                llm_inferred,
                user_confirmed,
                system_verified,
                contradicted,
                deprecated,
            },
            embeddings,
            reconsolidation: MemoryQualityReconsolidationStats {
                pending_candidates: pending_reconsolidation,
                reconsolidated_nodes,
            },
            retrieval: MemoryQualityRetrievalStats {
                recent_activations,
                average_activation_nodes,
                last_activation_at,
            },
            repair_plan: Some(repair_plan),
            warnings,
            recommendations,
            metadata: json!({
                "source": "sqlite_memory_graph_quality_dashboard",
                "source_of_truth": "sqlite_memory_graph",
                "vector_index_role": "advisory_retrieval_index",
                "embedding_provider": embedding_provider,
                "quality_score_components": {
                    "semantic_ratio": semantic_ratio,
                    "embedding_ratio": embedding_ratio,
                    "reconsolidation_penalty": reconsolidation_penalty,
                    "deprecated_penalty": deprecated_penalty,
                    "verified_ratio": verified_ratio,
                    "average_salience": average_salience
                },
                "metadata_only": true,
            }),
        })
    }

    pub fn append_memory_note(&self, event: &str, payload: serde_json::Value) -> MemoryResult<()> {
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        self.append_journal_locked(event, &payload);
        Ok(())
    }

    fn initialize(&self) -> MemoryResult<()> {
        let _guard = self.lock.lock().expect("memory graph mutex poisoned");
        let conn = self.open_connection()?;
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS memory_nodes (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT,
                tags_json TEXT NOT NULL,
                source TEXT,
                confidence REAL NOT NULL,
                verification_status TEXT NOT NULL,
                salience REAL NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS memory_edges (
                id TEXT PRIMARY KEY,
                from_node_id TEXT NOT NULL,
                to_node_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL NOT NULL,
                confidence REAL NOT NULL,
                created_at INTEGER NOT NULL,
                last_activated_at INTEGER,
                activation_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(from_node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY(to_node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_memory_nodes_kind ON memory_nodes(kind);
             CREATE INDEX IF NOT EXISTS idx_memory_nodes_created ON memory_nodes(created_at DESC);
             CREATE INDEX IF NOT EXISTS idx_memory_edges_from ON memory_edges(from_node_id);
             CREATE INDEX IF NOT EXISTS idx_memory_edges_to ON memory_edges(to_node_id);
             CREATE TABLE IF NOT EXISTS memory_chunks (
                id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                text TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
             );
             CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks_fts USING fts5(
                chunk_id UNINDEXED,
                node_id UNINDEXED,
                text
             );
             CREATE TABLE IF NOT EXISTS memory_embeddings (
                chunk_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES memory_chunks(id) ON DELETE CASCADE,
                FOREIGN KEY(node_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_memory_embeddings_node ON memory_embeddings(node_id);
             CREATE INDEX IF NOT EXISTS idx_memory_embeddings_updated ON memory_embeddings(updated_at DESC);
             CREATE TABLE IF NOT EXISTS memory_skill_candidates (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                source_node_id TEXT,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                salience REAL NOT NULL,
                trigger_hints_json TEXT NOT NULL,
                required_tools_json TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                approved_by TEXT,
                approved_at INTEGER,
                metadata_json TEXT NOT NULL,
                FOREIGN KEY(source_node_id) REFERENCES memory_nodes(id) ON DELETE SET NULL
             );
             CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_skill_candidates_source ON memory_skill_candidates(source_node_id) WHERE source_node_id IS NOT NULL;
             CREATE INDEX IF NOT EXISTS idx_memory_skill_candidates_status ON memory_skill_candidates(status);
             CREATE INDEX IF NOT EXISTS idx_memory_skill_candidates_updated ON memory_skill_candidates(updated_at DESC);
             CREATE TABLE IF NOT EXISTS memory_activations (
                id TEXT PRIMARY KEY,
                request_id TEXT,
                root_query TEXT NOT NULL,
                activated_node_ids TEXT NOT NULL,
                activated_edge_ids TEXT NOT NULL,
                intensity_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                metadata_json TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_memory_activations_created ON memory_activations(created_at DESC);"
        )?;
        Ok(())
    }

    fn open_connection(&self) -> MemoryResult<Connection> {
        Ok(Connection::open(&self.path)?)
    }

    fn append_journal_locked<T: serde::Serialize>(&self, event: &str, payload: &T) {
        let path = self.config.journal_dir.join("memory_events.jsonl");
        let record = json!({
            "schema_version": 1,
            "event": event,
            "timestamp_ms": now_ms(),
            "payload": payload,
        });
        if let Ok(line) = serde_json::to_string(&record) {
            let _ = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut file| std::io::Write::write_all(&mut file, format!("{line}\n").as_bytes()));
        }
    }
}


fn embedding_status_locked(conn: &Connection) -> MemoryResult<MemoryEmbeddingIndexStatus> {
    let total_chunks = count_table(conn, "memory_chunks")? as usize;
    let embedded_chunks = count_table(conn, "memory_embeddings")? as usize;
    let last_indexed_at = conn
        .query_row("SELECT MAX(updated_at) FROM memory_embeddings", [], |row| row.get::<_, Option<i64>>(0))
        .optional()?
        .flatten();
    let (provider, dimensions) = conn
        .query_row(
            "SELECT model, dimensions FROM memory_embeddings ORDER BY updated_at DESC LIMIT 1",
            [],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize)),
        )
        .optional()?
        .unwrap_or_else(|| ("stable-local-hash-v1".into(), 384));
    Ok(MemoryEmbeddingIndexStatus {
        backend: "sqlite_vector_cache".into(),
        provider,
        dimensions,
        embedded_chunks,
        total_chunks,
        pending_chunks: total_chunks.saturating_sub(embedded_chunks),
        last_indexed_at,
        metadata: json!({
            "source_of_truth": "sqlite_memory_graph",
            "vector_index_role": "advisory_retrieval_index",
            "metadata_only": true,
        }),
    })
}

fn count_where(conn: &Connection, table: &str, predicate: &str) -> MemoryResult<i64> {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE {predicate}");
    Ok(conn.query_row(&sql, [], |row| row.get::<_, i64>(0))?)
}

fn clamp_quality_score(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn insert_node(conn: &Connection, node: &MemoryNode) -> MemoryResult<()> {
    conn.execute(
        "INSERT INTO memory_nodes (id, kind, title, summary, content, tags_json, source, confidence, verification_status, salience, created_at, updated_at, metadata_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            &node.id,
            node.kind.as_str(),
            &node.title,
            &node.summary,
            &node.content,
            serde_json::to_string(&node.tags)?,
            &node.source,
            node.confidence,
            node.verification_status.as_str(),
            node.salience,
            node.created_at,
            node.updated_at,
            serde_json::to_string(&node.metadata)?,
        ],
    )?;
    Ok(())
}

fn insert_edge(conn: &Connection, edge: &MemoryEdge) -> MemoryResult<()> {
    conn.execute(
        "INSERT INTO memory_edges (id, from_node_id, to_node_id, relation, weight, confidence, created_at, last_activated_at, activation_count, metadata_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            &edge.id,
            &edge.from_node_id,
            &edge.to_node_id,
            edge.relation.as_str(),
            edge.weight,
            edge.confidence,
            edge.created_at,
            &edge.last_activated_at,
            edge.activation_count as i64,
            serde_json::to_string(&edge.metadata)?,
        ],
    )?;
    Ok(())
}

fn insert_chunk(conn: &Connection, chunk: &MemoryChunk) -> MemoryResult<()> {
    conn.execute(
        "INSERT INTO memory_chunks (id, node_id, text, ordinal, created_at, metadata_json) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![&chunk.id, &chunk.node_id, &chunk.text, chunk.ordinal as i64, chunk.created_at, serde_json::to_string(&chunk.metadata)?],
    )?;
    conn.execute(
        "INSERT INTO memory_chunks_fts (chunk_id, node_id, text) VALUES (?1, ?2, ?3)",
        params![&chunk.id, &chunk.node_id, &chunk.text],
    )?;
    Ok(())
}

fn insert_activation(conn: &Connection, activation: &MemoryActivation) -> MemoryResult<()> {
    conn.execute(
        "INSERT INTO memory_activations (id, request_id, root_query, activated_node_ids, activated_edge_ids, intensity_json, created_at, metadata_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            &activation.id,
            &activation.request_id,
            &activation.root_query,
            serde_json::to_string(&activation.activated_node_ids)?,
            serde_json::to_string(&activation.activated_edge_ids)?,
            serde_json::to_string(&activation.intensity)?,
            activation.created_at,
            serde_json::to_string(&activation.metadata)?,
        ],
    )?;
    Ok(())
}


fn insert_skill_candidate(conn: &Connection, candidate: &MemorySkillCandidate) -> MemoryResult<()> {
    conn.execute(
        "INSERT INTO memory_skill_candidates (id, title, summary, source_node_id, status, confidence, salience, trigger_hints_json, required_tools_json, risk_level, created_at, updated_at, approved_by, approved_at, metadata_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
         ON CONFLICT DO NOTHING",
        params![
            &candidate.id,
            &candidate.title,
            &candidate.summary,
            &candidate.source_node_id,
            candidate.status.as_str(),
            candidate.confidence as f64,
            candidate.salience as f64,
            serde_json::to_string(&candidate.trigger_hints)?,
            serde_json::to_string(&candidate.required_tools)?,
            &candidate.risk_level,
            candidate.created_at,
            candidate.updated_at,
            &candidate.approved_by,
            &candidate.approved_at,
            serde_json::to_string(&candidate.metadata)?,
        ],
    )?;
    Ok(())
}

fn skill_candidate_by_id(conn: &Connection, candidate_id: &str) -> MemoryResult<MemorySkillCandidate> {
    conn.query_row(
        "SELECT id, title, summary, source_node_id, status, confidence, salience, trigger_hints_json, required_tools_json, risk_level, created_at, updated_at, approved_by, approved_at, metadata_json
         FROM memory_skill_candidates WHERE id = ?1 LIMIT 1",
        params![candidate_id],
        row_to_skill_candidate,
    )
    .map_err(Into::into)
}

fn row_to_skill_candidate(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemorySkillCandidate> {
    let status: String = row.get(4)?;
    let trigger_hints_json: String = row.get(7)?;
    let required_tools_json: String = row.get(8)?;
    let metadata_json: String = row.get(14)?;
    Ok(MemorySkillCandidate {
        id: row.get(0)?,
        title: row.get(1)?,
        summary: row.get(2)?,
        source_node_id: row.get(3)?,
        status: MemorySkillCandidateStatus::from_str(&status),
        confidence: row.get::<_, f64>(5)? as f32,
        salience: row.get::<_, f64>(6)? as f32,
        trigger_hints: serde_json::from_str(&trigger_hints_json).unwrap_or_default(),
        required_tools: serde_json::from_str(&required_tools_json).unwrap_or_default(),
        risk_level: row.get(9)?,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
        approved_by: row.get(12)?,
        approved_at: row.get(13)?,
        metadata: serde_json::from_str(&metadata_json).unwrap_or_else(|_| json!({})),
    })
}

fn normalize_small_list(values: Vec<String>, max_items: usize, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for value in values.into_iter().map(|value| value.trim().to_string()).filter(|value| !value.is_empty()) {
        let capped = cap_text(value, max_chars);
        if !out.iter().any(|existing: &String| existing.eq_ignore_ascii_case(&capped)) {
            out.push(capped);
        }
        if out.len() >= max_items {
            break;
        }
    }
    out
}


fn extract_turn_messages_from_node(node: &MemoryNode) -> (String, String) {
    let content = node.content.as_deref().unwrap_or_default();
    if let Some((user, assistant)) = split_turn_content(content) {
        return (cap_text(user, 16_000), cap_text(assistant, 16_000));
    }
    let user = if node.title.trim().is_empty() {
        node.summary.clone()
    } else {
        node.title.clone()
    };
    let assistant = if !content.trim().is_empty() {
        content.to_string()
    } else {
        node.summary.clone()
    };
    (cap_text(user, 16_000), cap_text(assistant, 16_000))
}

fn split_turn_content(content: &str) -> Option<(String, String)> {
    let user_marker = "User message:";
    let assistant_marker = "Assistant answer:";
    let user_start = content.find(user_marker)? + user_marker.len();
    let assistant_start_marker = content.find(assistant_marker)?;
    let assistant_start = assistant_start_marker + assistant_marker.len();
    let user = content[user_start..assistant_start_marker].trim().to_string();
    let assistant = content[assistant_start..].trim().to_string();
    (!user.is_empty()).then_some((user, assistant))
}

fn lexical_query(conn: &Connection, query: &str, limit: usize) -> MemoryResult<Vec<MemoryQueryHit>> {
    let fts_query = sanitize_fts_query(query);
    let sql = if fts_query.is_empty() {
        "SELECT n.id, n.kind, n.title, n.summary, n.content, n.tags_json, n.source, n.confidence, n.verification_status, n.salience, n.created_at, n.updated_at, n.metadata_json, 0.25 as score
         FROM memory_nodes n ORDER BY n.updated_at DESC LIMIT ?1".to_string()
    } else {
        "SELECT n.id, n.kind, n.title, n.summary, n.content, n.tags_json, n.source, n.confidence, n.verification_status, n.salience, n.created_at, n.updated_at, n.metadata_json,
                bm25(memory_chunks_fts) * -1.0 + n.salience + n.confidence as score
         FROM memory_chunks_fts
         JOIN memory_nodes n ON n.id = memory_chunks_fts.node_id
         WHERE memory_chunks_fts MATCH ?1
         GROUP BY n.id
         ORDER BY score DESC, n.updated_at DESC
         LIMIT ?2".to_string()
    };
    let mut hits = Vec::new();
    if fts_query.is_empty() {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(MemoryQueryHit { node: row_to_node(row)?, score: row.get::<_, f64>(13)? as f32, reasons: vec!["recent_fallback".into()] })
        })?;
        for row in rows { hits.push(row?); }
    } else {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
            Ok(MemoryQueryHit { node: row_to_node(row)?, score: row.get::<_, f64>(13)? as f32, reasons: vec!["fts5_lexical_match".into()] })
        })?;
        for row in rows { hits.push(row?); }
    }
    Ok(hits)
}

fn latest_nodes(conn: &Connection, limit: usize) -> MemoryResult<Vec<MemoryNode>> {
    let mut stmt = conn.prepare(
        "SELECT id, kind, title, summary, content, tags_json, source, confidence, verification_status, salience, created_at, updated_at, metadata_json
         FROM memory_nodes ORDER BY updated_at DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], row_to_node)?;
    let mut nodes = Vec::new();
    for row in rows { nodes.push(row?); }
    Ok(nodes)
}

fn edges_for_nodes(conn: &Connection, ids: &[String], limit: usize) -> MemoryResult<Vec<MemoryEdge>> {
    if ids.is_empty() { return Ok(Vec::new()); }
    let id_set = ids.iter().cloned().collect::<HashSet<_>>();
    let mut stmt = conn.prepare(
        "SELECT id, from_node_id, to_node_id, relation, weight, confidence, created_at, last_activated_at, activation_count, metadata_json
         FROM memory_edges ORDER BY weight DESC, created_at DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![(limit * 5) as i64], row_to_edge)?;
    let mut edges = Vec::new();
    for row in rows {
        let edge = row?;
        if id_set.contains(&edge.from_node_id) || id_set.contains(&edge.to_node_id) {
            edges.push(edge);
            if edges.len() >= limit { break; }
        }
    }
    Ok(edges)
}

fn propagate_activation(conn: &Connection, seeds: &[String], max_depth: usize, max_nodes: usize) -> MemoryResult<(Vec<String>, Vec<String>, Value)> {
    let mut visited = HashSet::new();
    let mut edge_ids = HashSet::new();
    let mut intensities: HashMap<String, f32> = HashMap::new();
    let mut queue = VecDeque::new();
    for seed in seeds.iter().filter(|seed| !seed.trim().is_empty()) {
        if ensure_node_exists(conn, seed).is_ok() {
            visited.insert(seed.clone());
            intensities.insert(seed.clone(), 1.0);
            queue.push_back((seed.clone(), 0usize, 1.0f32));
        }
    }
    while let Some((node_id, depth, intensity)) = queue.pop_front() {
        if depth >= max_depth || visited.len() >= max_nodes { continue; }
        let mut stmt = conn.prepare(
            "SELECT id, from_node_id, to_node_id, relation, weight, confidence, created_at, last_activated_at, activation_count, metadata_json
             FROM memory_edges WHERE from_node_id = ?1 OR to_node_id = ?1 ORDER BY weight DESC LIMIT 32",
        )?;
        let rows = stmt.query_map(params![node_id], row_to_edge)?;
        for row in rows {
            let edge = row?;
            edge_ids.insert(edge.id.clone());
            let other = if edge.from_node_id == node_id { edge.to_node_id.clone() } else { edge.from_node_id.clone() };
            let next_intensity = intensity * edge.weight.max(0.05) * edge.confidence.max(0.05);
            intensities.entry(other.clone()).and_modify(|existing| *existing = existing.max(next_intensity)).or_insert(next_intensity);
            if visited.insert(other.clone()) && visited.len() < max_nodes {
                queue.push_back((other, depth + 1, next_intensity));
            }
        }
    }
    let intensity_json = Value::Object(intensities.into_iter().map(|(node, score)| (node, json!(score))).collect());
    Ok((visited.into_iter().collect(), edge_ids.into_iter().collect(), intensity_json))
}

fn ensure_node_exists(conn: &Connection, id: &str) -> MemoryResult<()> {
    let exists: Option<String> = conn.query_row("SELECT id FROM memory_nodes WHERE id = ?1", params![id], |row| row.get(0)).optional()?;
    exists.map(|_| ()).ok_or_else(|| MemoryError::Validation(format!("memory node not found: {id}")))
}


fn row_to_chunk(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryChunk> {
    let metadata_json: String = row.get(5)?;
    Ok(MemoryChunk {
        id: row.get(0)?,
        node_id: row.get(1)?,
        text: row.get(2)?,
        ordinal: row.get::<_, i64>(3)? as u32,
        created_at: row.get(4)?,
        metadata: serde_json::from_str(&metadata_json).unwrap_or_else(|_| json!({})),
    })
}

fn node_by_id(conn: &Connection, id: &str) -> MemoryResult<MemoryNode> {
    conn.query_row(
        "SELECT id, kind, title, summary, content, tags_json, source, confidence, verification_status, salience, created_at, updated_at, metadata_json
         FROM memory_nodes WHERE id = ?1",
        params![id],
        row_to_node,
    )
    .optional()?
    .ok_or_else(|| MemoryError::Validation(format!("memory node not found: {id}")))
}

fn nodes_by_ids(conn: &Connection, ids: &[String]) -> MemoryResult<HashMap<String, MemoryNode>> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let id_set = ids.iter().cloned().collect::<HashSet<_>>();
    let mut stmt = conn.prepare(
        "SELECT id, kind, title, summary, content, tags_json, source, confidence, verification_status, salience, created_at, updated_at, metadata_json
         FROM memory_nodes ORDER BY updated_at DESC",
    )?;
    let rows = stmt.query_map([], row_to_node)?;
    let mut nodes = HashMap::new();
    for row in rows {
        let node = row?;
        if id_set.contains(&node.id) {
            nodes.insert(node.id.clone(), node);
        }
    }
    Ok(nodes)
}

fn count_table(conn: &Connection, table: &str) -> MemoryResult<i64> {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    Ok(conn.query_row(&sql, [], |row| row.get(0))?)
}

fn row_to_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryNode> {
    let kind: String = row.get(1)?;
    let verification_status: String = row.get(8)?;
    let tags_json: String = row.get(5)?;
    let metadata_json: String = row.get(12)?;
    Ok(MemoryNode {
        id: row.get(0)?,
        kind: serde_json::from_value(json!(kind)).unwrap_or_default(),
        title: row.get(2)?,
        summary: row.get(3)?,
        content: row.get(4)?,
        tags: serde_json::from_str(&tags_json).unwrap_or_default(),
        source: row.get(6)?,
        confidence: row.get::<_, f64>(7)? as f32,
        verification_status: serde_json::from_value(json!(verification_status)).unwrap_or_default(),
        salience: row.get::<_, f64>(9)? as f32,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
        metadata: serde_json::from_str(&metadata_json).unwrap_or_else(|_| json!({})),
    })
}

fn row_to_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEdge> {
    let relation: String = row.get(3)?;
    let metadata_json: String = row.get(9)?;
    Ok(MemoryEdge {
        id: row.get(0)?,
        from_node_id: row.get(1)?,
        to_node_id: row.get(2)?,
        relation: serde_json::from_value(json!(relation)).unwrap_or_default(),
        weight: row.get::<_, f64>(4)? as f32,
        confidence: row.get::<_, f64>(5)? as f32,
        created_at: row.get(6)?,
        last_activated_at: row.get(7)?,
        activation_count: row.get::<_, i64>(8)? as u64,
        metadata: serde_json::from_str(&metadata_json).unwrap_or_else(|_| json!({})),
    })
}

fn row_to_activation(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryActivation> {
    let nodes_json: String = row.get(3)?;
    let edges_json: String = row.get(4)?;
    let intensity_json: String = row.get(5)?;
    let metadata_json: String = row.get(7)?;
    Ok(MemoryActivation {
        id: row.get(0)?,
        request_id: row.get(1)?,
        root_query: row.get(2)?,
        activated_node_ids: serde_json::from_str(&nodes_json).unwrap_or_default(),
        activated_edge_ids: serde_json::from_str(&edges_json).unwrap_or_default(),
        intensity: serde_json::from_str(&intensity_json).unwrap_or_else(|_| json!({})),
        created_at: row.get(6)?,
        metadata: serde_json::from_str(&metadata_json).unwrap_or_else(|_| json!({})),
    })
}

fn node_search_text(node: &MemoryNode) -> String {
    [
        node.title.as_str(),
        node.summary.as_str(),
        node.content.as_deref().unwrap_or(""),
        &node.tags.join(" "),
    ].join("\n")
}

fn sanitize_fts_query(query: &str) -> String {
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .map(str::trim)
        .filter(|token| token.chars().count() >= 2)
        .take(16)
        .map(|token| format!("\"{}\"", token.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" OR ")
}



struct DuplicateScore {
    score: f32,
    reasons: Vec<String>,
    shared_tags: Vec<String>,
}

fn choose_canonical_node(left: &MemoryNode, right: &MemoryNode) -> (MemoryNode, MemoryNode) {
    let left_score = canonical_node_score(left);
    let right_score = canonical_node_score(right);
    if left_score >= right_score {
        (left.clone(), right.clone())
    } else {
        (right.clone(), left.clone())
    }
}

fn canonical_node_score(node: &MemoryNode) -> f32 {
    let status_boost = match node.verification_status {
        MemoryVerificationStatus::UserConfirmed => 2.0,
        MemoryVerificationStatus::SystemVerified => 1.7,
        MemoryVerificationStatus::LlmInferred => 1.2,
        MemoryVerificationStatus::Unverified => 1.0,
        MemoryVerificationStatus::Contradicted | MemoryVerificationStatus::Deprecated => 0.1,
    };
    (node.salience * 1.8) + (node.confidence * 1.2) + status_boost + ((node.updated_at.max(0) as f32) / 10_000_000_000_000.0)
}

fn merged_into(node: &MemoryNode) -> Option<String> {
    node.metadata
        .get("merged_into_node_id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn duplicate_score(left: &MemoryNode, right: &MemoryNode) -> DuplicateScore {
    let title_similarity = token_jaccard(&left.title, &right.title);
    let summary_similarity = token_jaccard(&left.summary, &right.summary);
    let content_similarity = token_jaccard(
        left.content.as_deref().unwrap_or(""),
        right.content.as_deref().unwrap_or(""),
    );
    let left_tags = left.tags.iter().map(|tag| tag.to_ascii_lowercase()).collect::<HashSet<_>>();
    let right_tags = right.tags.iter().map(|tag| tag.to_ascii_lowercase()).collect::<HashSet<_>>();
    let shared_tags = left_tags
        .intersection(&right_tags)
        .cloned()
        .collect::<Vec<_>>();
    let tag_score = if left_tags.is_empty() && right_tags.is_empty() {
        0.0
    } else {
        let union = left_tags.union(&right_tags).count().max(1) as f32;
        shared_tags.len() as f32 / union
    };
    let source_score = match (left.source.as_deref(), right.source.as_deref()) {
        (Some(a), Some(b)) if a == b => 1.0,
        _ => 0.0,
    };
    let status_penalty = if matches!(left.verification_status, MemoryVerificationStatus::Contradicted | MemoryVerificationStatus::Deprecated)
        || matches!(right.verification_status, MemoryVerificationStatus::Contradicted | MemoryVerificationStatus::Deprecated)
    {
        0.2
    } else {
        0.0
    };
    let mut score = (title_similarity * 0.38)
        + (summary_similarity * 0.28)
        + (content_similarity * 0.14)
        + (tag_score * 0.14)
        + (source_score * 0.06)
        - status_penalty;
    score = score.clamp(0.0, 1.0);
    let mut reasons = Vec::new();
    if title_similarity >= 0.72 { reasons.push("similar_title".into()); }
    if summary_similarity >= 0.58 { reasons.push("similar_summary".into()); }
    if content_similarity >= 0.5 { reasons.push("similar_content".into()); }
    if tag_score >= 0.25 { reasons.push("shared_tags".into()); }
    if source_score > 0.0 { reasons.push("same_source".into()); }
    if reasons.is_empty() { reasons.push("combined_similarity".into()); }
    DuplicateScore { score, reasons, shared_tags }
}

fn token_jaccard(left: &str, right: &str) -> f32 {
    let left_tokens = semantic_tokens(left);
    let right_tokens = semantic_tokens(right);
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let intersection = left_tokens.intersection(&right_tokens).count() as f32;
    let union = left_tokens.union(&right_tokens).count().max(1) as f32;
    intersection / union
}

fn semantic_tokens(value: &str) -> HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "the", "and", "for", "with", "that", "this", "from", "una", "uno", "che", "per", "con", "del", "della", "sono", "essere", "utente", "user",
    ];
    value
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::trim)
        .filter(|token| token.chars().count() >= 2)
        .map(|token| token.to_ascii_lowercase())
        .filter(|token| !STOPWORDS.contains(&token.as_str()))
        .take(64)
        .collect()
}


fn duplicate_review_pressure(conn: &Connection) -> MemoryResult<usize> {
    let mut stmt = conn.prepare(
        "SELECT LOWER(TRIM(title)) AS title_key, COUNT(*)
         FROM memory_nodes
         WHERE verification_status NOT IN ('deprecated', 'contradicted')
           AND LENGTH(TRIM(title)) > 0
         GROUP BY title_key
         HAVING COUNT(*) > 1
         LIMIT 50",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, i64>(1))?;
    let mut pressure = 0usize;
    for row in rows {
        let count = row?.max(1) as usize;
        pressure = pressure.saturating_add(count.saturating_sub(1));
    }
    Ok(pressure)
}

fn canonical_review_pressure(conn: &Connection) -> MemoryResult<usize> {
    let profile_or_identity = count_where(
        conn,
        "memory_nodes",
        "verification_status NOT IN ('deprecated', 'contradicted') AND (tags_json LIKE '%identity%' OR tags_json LIKE '%profile%' OR tags_json LIKE '%user_profile%' OR title LIKE '%identity%' OR title LIKE '%Simone%')",
    )? as usize;
    let semantic_candidates = count_where(
        conn,
        "memory_nodes",
        "verification_status NOT IN ('deprecated', 'contradicted') AND kind IN ('claim','entity','user_preference','concept','procedure','decision','workflow')",
    )? as usize;
    Ok(profile_or_identity.min(semantic_candidates.saturating_add(profile_or_identity)))
}

fn build_memory_health_repair_plan(
    nodes: usize,
    semantic_nodes: usize,
    semantic_ratio: f32,
    episode_only_nodes: usize,
    pending_reconsolidation: usize,
    pending_embeddings: usize,
    deprecated_or_contradicted: usize,
    duplicate_pressure: usize,
    canonical_pressure: usize,
    score: f32,
    status: &str,
) -> MemoryHealthRepairPlan {
    let mut actions = Vec::new();
    let generated_at = now_ms();

    if pending_embeddings > 0 {
        actions.push(MemoryHealthRepairAction {
            id: "embedding-maintenance".into(),
            kind: "embedding_maintenance".into(),
            title: "Index pending memory vectors".into(),
            description: "Update the vector cache for newly learned memory chunks so Brain RAG can retrieve them semantically.".into(),
            priority: if pending_embeddings > 25 { "high" } else { "medium" }.into(),
            risk_level: "low".into(),
            requires_user_review: false,
            can_run_automatically: true,
            status: "ready".into(),
            affected_count: pending_embeddings,
            confidence: 0.98,
            rationale: "Vector indexing is non-destructive and does not change memory meaning.".into(),
            command_hint: Some("run_memory_embedding_maintenance".into()),
            metadata: json!({"metadata_only": true}),
        });
    }

    let weak_episode_debt = pending_reconsolidation.max(episode_only_nodes.saturating_sub(semantic_nodes));
    if weak_episode_debt > 0 || semantic_ratio < 0.25 && nodes > 8 {
        actions.push(MemoryHealthRepairAction {
            id: "semantic-repair".into(),
            kind: "semantic_reconsolidation".into(),
            title: "Repair weak episodic memories".into(),
            description: "Ask the LLM to re-read raw conversation memories and distill durable semantic atoms while preserving original evidence.".into(),
            priority: if semantic_ratio < 0.15 { "high" } else { "medium" }.into(),
            risk_level: "low".into(),
            requires_user_review: false,
            can_run_automatically: true,
            status: if pending_reconsolidation > 0 { "ready" } else { "needs_candidate_refresh" }.into(),
            affected_count: weak_episode_debt,
            confidence: 0.82,
            rationale: "Semantic nodes are too sparse compared with episodic memory; retrieval may be noisy until raw turns are distilled.".into(),
            command_hint: Some("reconsolidate_memory_candidates".into()),
            metadata: json!({"semantic_ratio": semantic_ratio, "metadata_only": true}),
        });
    }

    if duplicate_pressure > 0 {
        actions.push(MemoryHealthRepairAction {
            id: "duplicate-review".into(),
            kind: "duplicate_review".into(),
            title: "Review probable duplicate memories".into(),
            description: "Review duplicate candidates proposed by the Memory Graph and merge only after user approval.".into(),
            priority: "medium".into(),
            risk_level: "medium".into(),
            requires_user_review: true,
            can_run_automatically: false,
            status: "review_required".into(),
            affected_count: duplicate_pressure,
            confidence: 0.74,
            rationale: "Merging changes canonical structure, so Autopilot can propose but not apply it without governance.".into(),
            command_hint: Some("list_memory_duplicate_candidates".into()),
            metadata: json!({"metadata_only": true}),
        });
    }

    if canonical_pressure > 1 {
        actions.push(MemoryHealthRepairAction {
            id: "canonical-review".into(),
            kind: "canonical_review".into(),
            title: "Review canonical memory candidates".into(),
            description: "Let the LLM/RAG propose canonical entities or concepts, then approve safe soft-merges through the review queue.".into(),
            priority: if semantic_ratio < 0.25 { "high" } else { "medium" }.into(),
            risk_level: "medium".into(),
            requires_user_review: true,
            can_run_automatically: false,
            status: "review_required".into(),
            affected_count: canonical_pressure,
            confidence: 0.70,
            rationale: "Canonicalization improves recall quality but must remain user-governed because it changes semantic structure.".into(),
            command_hint: Some("list_memory_canonical_review_candidates".into()),
            metadata: json!({"metadata_only": true}),
        });
    }

    if deprecated_or_contradicted > 0 {
        actions.push(MemoryHealthRepairAction {
            id: "governance-cleanup".into(),
            kind: "governance_review".into(),
            title: "Review deprecated or contradicted memories".into(),
            description: "Inspect contradicted/deprecated nodes and decide whether to keep as evidence, merge, correct, or leave suppressed.".into(),
            priority: "medium".into(),
            risk_level: "medium".into(),
            requires_user_review: true,
            can_run_automatically: false,
            status: "review_required".into(),
            affected_count: deprecated_or_contradicted,
            confidence: 0.86,
            rationale: "Suppressed memories are excluded from ordinary retrieval, but too many of them indicate unresolved graph noise.".into(),
            command_hint: Some("memory_governance_review".into()),
            metadata: json!({"metadata_only": true}),
        });
    }

    if actions.is_empty() {
        actions.push(MemoryHealthRepairAction {
            id: "monitor-quality".into(),
            kind: "monitoring".into(),
            title: "Monitor memory quality".into(),
            description: "No urgent repair action is required. Keep Autopilot enabled and continue monitoring Brain RAG health.".into(),
            priority: "low".into(),
            risk_level: "low".into(),
            requires_user_review: false,
            can_run_automatically: true,
            status: "monitoring".into(),
            affected_count: nodes,
            confidence: 0.9,
            rationale: "Quality indicators are within acceptable bounds for the current graph size.".into(),
            command_hint: None,
            metadata: json!({"metadata_only": true}),
        });
    }

    actions.sort_by(|left, right| repair_priority_rank(&right.priority).cmp(&repair_priority_rank(&left.priority)));
    let automatic_action_count = actions.iter().filter(|action| action.can_run_automatically && !action.requires_user_review).count();
    let review_action_count = actions.iter().filter(|action| action.requires_user_review).count();
    let blocked_action_count = actions.iter().filter(|action| !action.can_run_automatically && !action.requires_user_review).count();
    let summary = if score >= 0.76 {
        "Brain RAG health is strong; Autopilot should continue bounded maintenance.".to_string()
    } else if automatic_action_count > 0 && review_action_count > 0 {
        "Brain RAG needs automatic semantic repair plus user-governed review for structural changes.".to_string()
    } else if automatic_action_count > 0 {
        "Brain RAG can improve through automatic low-risk maintenance.".to_string()
    } else if review_action_count > 0 {
        "Brain RAG requires user-governed review before quality can improve safely.".to_string()
    } else {
        "Brain RAG health is being monitored.".to_string()
    };

    MemoryHealthRepairPlan {
        schema_version: 1,
        generated_at,
        status: status.to_string(),
        summary,
        actions,
        automatic_action_count,
        review_action_count,
        blocked_action_count,
        metadata: json!({
            "source": "memory_health_repair_plan",
            "llm_first_runtime": true,
            "destructive_actions_user_governed": true,
            "semantic_ratio": semantic_ratio,
            "metadata_only": true,
        }),
    }
}

fn repair_priority_rank(priority: &str) -> u8 {
    match priority {
        "critical" => 4,
        "high" => 3,
        "medium" => 2,
        "low" => 1,
        _ => 0,
    }
}

fn validate_non_empty(field: &str, value: &str) -> MemoryResult<()> {
    if value.trim().is_empty() {
        Err(MemoryError::Validation(format!("{field} cannot be empty")))
    } else {
        Ok(())
    }
}

fn clamp01(value: f32) -> f32 {
    if !value.is_finite() { return 0.5; }
    value.clamp(0.0, 1.0)
}

fn cap_text(mut value: String, max_chars: usize) -> String {
    if value.chars().count() <= max_chars { return value; }
    value = value.chars().take(max_chars).collect();
    value.push_str("…");
    value
}
