export type MemoryNodeKind =
  | "conversation_turn"
  | "work_session"
  | "transcript_segment"
  | "summary"
  | "concept"
  | "entity"
  | "task"
  | "tool_use"
  | "error"
  | "fix"
  | "research_topic"
  | "research_finding"
  | "source_document"
  | "code_pattern"
  | "user_preference"
  | "workflow"
  | "claim"
  | "decision"
  | "procedure"
  | "unknown";

export type MemoryRelationKind =
  | "mentions"
  | "about"
  | "derived_from"
  | "supports"
  | "contradicts"
  | "caused"
  | "resolved_by"
  | "follows"
  | "part_of"
  | "same_topic_as"
  | "preferred_by_user"
  | "used_tool"
  | "verified_by"
  | "learned_from"
  | "triggered"
  | "implemented_in"
  | "related_to_codebase"
  | "depends_on"
  | "related_to";

export type MemoryVerificationStatus =
  | "unverified"
  | "llm_inferred"
  | "user_confirmed"
  | "system_verified"
  | "contradicted"
  | "deprecated";

export type MemoryNode = {
  id: string;
  kind: MemoryNodeKind;
  title: string;
  summary: string;
  content?: string | null;
  tags: string[];
  source?: string | null;
  confidence: number;
  verification_status: MemoryVerificationStatus;
  salience: number;
  created_at: number;
  updated_at: number;
  metadata?: Record<string, unknown>;
};

export type MemoryEdge = {
  id: string;
  from_node_id: string;
  to_node_id: string;
  relation: MemoryRelationKind;
  weight: number;
  confidence: number;
  created_at: number;
  last_activated_at?: number | null;
  activation_count: number;
  metadata?: Record<string, unknown>;
};

export type MemoryActivation = {
  id: string;
  request_id?: string | null;
  root_query: string;
  activated_node_ids: string[];
  activated_edge_ids: string[];
  intensity?: Record<string, number> | Record<string, unknown>;
  created_at: number;
  metadata?: Record<string, unknown>;
};

export type MemoryGraphSnapshot = {
  nodes: MemoryNode[];
  edges: MemoryEdge[];
  activations: MemoryActivation[];
};

export type MemoryGraphStatus = {
  available: boolean;
  backend?: string;
  path?: string;
  nodes?: number;
  edges?: number;
  activations?: number;
  chunks?: number;
  embeddings?: number;
  pending_embeddings?: number;
  skill_candidates?: number;
  vector_backend?: string;
  fts_enabled?: boolean;
  max_query_limit?: number;
  max_activation_depth?: number;
  max_activation_nodes?: number;
  reason?: string;
};


export type MemoryQualityTotals = {
  nodes: number;
  edges: number;
  chunks: number;
  activations: number;
  skill_candidates: number;
};

export type MemoryQualitySemanticStats = {
  semantic_nodes: number;
  episode_only_nodes: number;
  conversation_turn_nodes: number;
  semantic_ratio: number;
  average_confidence: number;
  average_salience: number;
};

export type MemoryQualityGovernanceStats = {
  unverified: number;
  llm_inferred: number;
  user_confirmed: number;
  system_verified: number;
  contradicted: number;
  deprecated: number;
};

export type MemoryQualityReconsolidationStats = {
  pending_candidates: number;
  reconsolidated_nodes: number;
};

export type MemoryQualityRetrievalStats = {
  recent_activations: number;
  average_activation_nodes: number;
  last_activation_at?: number | null;
};

export type MemoryHealthRepairAction = {
  id: string;
  kind: string;
  title: string;
  description: string;
  priority: "critical" | "high" | "medium" | "low" | string;
  risk_level: "low" | "medium" | "high" | string;
  requires_user_review: boolean;
  can_run_automatically: boolean;
  status: string;
  affected_count: number;
  confidence: number;
  rationale: string;
  command_hint?: string | null;
  metadata?: Record<string, unknown>;
};

export type MemoryHealthRepairPlan = {
  schema_version: number;
  generated_at: number;
  status: string;
  summary: string;
  actions: MemoryHealthRepairAction[];
  automatic_action_count: number;
  review_action_count: number;
  blocked_action_count: number;
  metadata?: Record<string, unknown>;
};

export type MemoryQualityDashboard = {
  schema_version: number;
  generated_at: number;
  status: "healthy" | "degraded" | "needs_attention" | "unavailable" | string;
  score: number;
  summary: string;
  totals: MemoryQualityTotals;
  semantic: MemoryQualitySemanticStats;
  governance: MemoryQualityGovernanceStats;
  embeddings: MemoryEmbeddingIndexStatus;
  reconsolidation: MemoryQualityReconsolidationStats;
  retrieval: MemoryQualityRetrievalStats;
  repair_plan?: MemoryHealthRepairPlan | null;
  warnings: string[];
  recommendations: string[];
  metadata?: Record<string, unknown>;
};


export type MemoryGovernancePolicySnapshot = {
  version: string;
  user_control_enabled: boolean;
  inferred_memory_default_weight: number;
  user_confirmed_weight: number;
  deprecated_memory_retrieval_enabled: boolean;
  hard_delete_enabled: boolean;
  allowed_statuses: MemoryVerificationStatus[];
  metadata?: Record<string, unknown>;
};

export type MemoryNodeGovernanceUpdateRequest = {
  node_id: string;
  verification_status?: MemoryVerificationStatus | null;
  confidence?: number | null;
  salience?: number | null;
  add_tags?: string[];
  remove_tags?: string[];
  reason?: string | null;
  actor?: string | null;
  metadata?: Record<string, unknown>;
};

export type MemoryNodeGovernanceUpdateReceipt = {
  accepted: boolean;
  reason: string;
  node: MemoryNode;
  metadata?: Record<string, unknown>;
};

export type MemoryQueryRequest = {
  query: string;
  kinds?: MemoryNodeKind[];
  limit?: number;
  include_edges?: boolean;
};

export type MemoryQueryHit = {
  node: MemoryNode;
  score: number;
  reasons: string[];
};

export type MemoryQueryResponse = {
  hits: MemoryQueryHit[];
  related_edges: MemoryEdge[];
};

export type MemoryActivationEventPayload = {
  request_id?: string | null;
  root_query?: string;
  activated_node_ids?: string[];
  activated_edge_ids?: string[];
  intensity?: Record<string, number> | Record<string, unknown>;
  node_count?: number;
  edge_count?: number;
  source?: string;
};

export type DeepSearchRequest = {
  topic: string;
  objective?: string | null;
  query?: string | null;
  seed_urls?: string[];
  enable_web_discovery?: boolean | null;
  search_providers?: string[];
  include_general_web?: boolean | null;
  include_academic_sources?: boolean | null;
  max_discovery_results_per_provider?: number | null;
  max_discovered_sources?: number | null;
  initial_query_count?: number | null;
  allowed_domains?: string[];
  blocked_domains?: string[];
  tags?: string[];
  max_sources?: number | null;
  min_sources_for_learning?: number | null;
  max_bytes_per_source?: number | null;
  timeout_ms?: number | null;
  require_cross_source_verification?: boolean;
  document_ingestion?: boolean | null;
  prefer_academic_landing_pages?: boolean | null;
  enable_pdf_text_extraction?: boolean | null;
  autonomous_loop?: boolean | null;
  max_research_passes?: number | null;
  min_research_passes?: number | null;
  max_sources_per_pass?: number | null;
  min_new_information_gain?: number | null;
  min_coverage_score?: number | null;
  min_supported_claim_ratio?: number | null;
  enable_claim_graph?: boolean | null;
  min_independent_sources_for_claim?: number | null;
  enable_contradiction_detection?: boolean | null;
  enable_memory_promotion_policy?: boolean | null;
  auto_promote_supported_claims?: boolean | null;
  require_user_confirmation_for_system_verified?: boolean | null;
  min_promotion_confidence?: number | null;
  min_promotion_independent_sources?: number | null;
  allow_http_localhost?: boolean;
  metadata?: Record<string, unknown>;
};


export type DeepSearchKnowledgeRefreshRequest = {
  enabled?: boolean;
  dry_run?: boolean;
  snapshot_limit?: number;
  max_candidates?: number;
  stale_after_days?: number;
  temporal_stale_after_days?: number;
  low_confidence_threshold?: number;
  include_low_confidence_candidates?: boolean;
  tag_candidates_for_refresh?: boolean;
  max_tags?: number;
  run_refresh_research?: boolean;
  max_refresh_topics?: number;
  max_refresh_runs?: number;
  max_sources_per_topic?: number;
  blocked_topics?: string[];
  search_providers?: string[];
  deep_search_defaults?: DeepSearchRequest | null;
  metadata?: Record<string, unknown>;
};

export type DeepSearchKnowledgeRefreshCandidate = {
  node: MemoryNode;
  topic: string;
  reason: string;
  priority: number;
  age_days: number;
  temporal: boolean;
  low_confidence: boolean;
  metadata?: Record<string, unknown>;
};

export type DeepSearchKnowledgeRefreshReceipt = {
  accepted: boolean;
  reason: string;
  started_at: number;
  completed_at: number;
  dry_run: boolean;
  candidates_scanned: number;
  stale_candidates: number;
  tagged_for_refresh: number;
  refresh_runs: number;
  sources_accepted: number;
  claims_promoted: number;
  candidate_claims: number;
  candidates: DeepSearchKnowledgeRefreshCandidate[];
  autopilot?: unknown | null;
  warnings: string[];
  recommendations: string[];
  metadata?: Record<string, unknown>;
};

export type DeepSearchRunSummary = {
  id: string;
  topic: string;
  objective?: string | null;
  started_at: number;
  completed_at: number;
  duration_ms: number;
  sources_seen: number;
  sources_accepted: number;
  sources_rejected: number;
  status: string;
};

export type DeepSearchAcceptedSource = {
  url: string;
  title: string;
  content_hash: string;
  fetched_at: number;
  content_type?: string | null;
  discovered_by?: string | null;
  source_type?: string | null;
  discovery_rank?: number | null;
};

export type DeepSearchRejectedSource = {
  url: string;
  reason: string;
};

export type DeepSearchConsolidationSummary = {
  accepted: boolean;
  reason: string;
  topic_node: MemoryNode;
  created_node_ids: string[];
  created_edge_ids: string[];
  summary?: Record<string, unknown>;
};

export type DeepSearchMemoryPromotionStage =
  | "ephemeral_research"
  | "candidate_memory"
  | "llm_inferred_memory"
  | "review_required"
  | "blocked_contradicted";

export type DeepSearchMemoryPromotionDecision = {
  claim_cluster_id: string;
  stage: DeepSearchMemoryPromotionStage;
  verification_status: MemoryVerificationStatus;
  confidence: number;
  salience: number;
  reason: string;
  source_refs: string[];
  metadata?: Record<string, unknown>;
};

export type DeepSearchPromotionReport = {
  enabled: boolean;
  promoted_claims: number;
  candidate_claims: number;
  review_required_claims: number;
  blocked_claims: number;
  decisions: DeepSearchMemoryPromotionDecision[];
  warnings: string[];
  metadata?: Record<string, unknown>;
};

export type DeepSearchReceipt = {
  accepted: boolean;
  reason: string;
  run: DeepSearchRunSummary;
  consolidated?: DeepSearchConsolidationSummary | null;
  accepted_sources: DeepSearchAcceptedSource[];
  rejected_sources: DeepSearchRejectedSource[];
  extracted_claims: number;
  extracted_findings: number;
  warnings: string[];
  passes?: Array<Record<string, unknown>>;
  coverage?: Record<string, unknown>;
  saturation?: Record<string, unknown>;
  claim_graph?: DeepSearchClaimGraphReport | null;
  promotion?: DeepSearchPromotionReport | null;
  metadata?: Record<string, unknown>;
};


export type ResearchSource = {
  title: string;
  uri?: string | null;
  source_type?: string | null;
  summary?: string | null;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type ResearchFinding = {
  title: string;
  summary: string;
  evidence?: string[];
  source_refs?: string[];
  confidence?: number | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
};

export type ResearchClaim = {
  claim: string;
  rationale?: string | null;
  source_refs?: string[];
  confidence?: number | null;
  verification_status?: MemoryVerificationStatus | null;
  metadata?: Record<string, unknown>;
};

export type ResearchProcedure = {
  title: string;
  steps?: string[];
  rationale?: string | null;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type ResearchRecommendation = {
  title: string;
  summary: string;
  actionability?: string | null;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type ResearchMemoryBundle = {
  topic: string;
  objective?: string | null;
  query?: string | null;
  summary?: string | null;
  confidence?: number | null;
  verification_status?: MemoryVerificationStatus | null;
  tags?: string[];
  sources?: ResearchSource[];
  findings?: ResearchFinding[];
  claims?: ResearchClaim[];
  procedures?: ResearchProcedure[];
  recommendations?: ResearchRecommendation[];
  metadata?: Record<string, unknown>;
};

export type ResearchMemoryConsolidationReceipt = {
  accepted: boolean;
  reason: string;
  topic_node: MemoryNode;
  created_node_ids: string[];
  created_edge_ids: string[];
  activation?: MemoryActivation | null;
  summary?: Record<string, unknown>;
};

export type ConversationImportantPoint = {
  title: string;
  summary: string;
  kind?: string | null;
  confidence?: number | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
};

export type ConversationEntity = {
  name: string;
  entity_type?: string | null;
  summary?: string | null;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type ConversationSemanticAtom = {
  title?: string | null;
  summary?: string | null;
  subject?: string | null;
  predicate?: string | null;
  object?: string | null;
  evidence?: string | null;
  kind?: string | null;
  confidence?: number | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
};

export type ConversationPreference = {
  preference: string;
  rationale?: string | null;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type ConversationProcedure = {
  title: string;
  steps?: string[];
  rationale?: string | null;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type ConversationDecision = {
  title: string;
  summary: string;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
};

export type ConversationMemoryBundle = {
  request_id?: string | null;
  source?: string | null;
  user_message: string;
  assistant_answer: string;
  topic?: string | null;
  summary?: string | null;
  importance?: number | null;
  confidence?: number | null;
  tags?: string[];
  semantic_atoms?: ConversationSemanticAtom[];
  important_points?: ConversationImportantPoint[];
  entities?: ConversationEntity[];
  preferences?: ConversationPreference[];
  procedures?: ConversationProcedure[];
  decisions?: ConversationDecision[];
  metadata?: Record<string, unknown>;
};

export type ConversationMemoryConsolidationReceipt = {
  accepted: boolean;
  reason: string;
  turn_node: MemoryNode;
  created_node_ids: string[];
  created_edge_ids: string[];
  activation?: MemoryActivation | null;
  summary?: Record<string, unknown>;
};

export type MemoryEmbeddingIndexStatus = {
  backend: string;
  provider: string;
  dimensions: number;
  embedded_chunks: number;
  total_chunks: number;
  pending_chunks: number;
  last_indexed_at?: number | null;
  metadata?: Record<string, unknown>;
};

export type MemoryEmbeddingRebuildRequest = {
  limit?: number | null;
  force?: boolean;
  model?: string | null;
};

export type MemoryEmbeddingRebuildReceipt = {
  accepted: boolean;
  reason: string;
  indexed_chunks: number;
  skipped_chunks: number;
  failed_chunks: number;
  model: string;
  dimensions: number;
  sample_node_ids: string[];
  metadata?: Record<string, unknown>;
};


export type MemoryEmbeddingMaintenanceRequest = {
  limit?: number | null;
  force?: boolean;
  model?: string | null;
  reason?: string | null;
};

export type MemoryEmbeddingMaintenanceReceipt = {
  accepted: boolean;
  reason: string;
  ran: boolean;
  indexed_chunks: number;
  skipped_chunks: number;
  failed_chunks: number;
  pending_before: number;
  pending_after: number;
  model: string;
  dimensions: number;
  sample_node_ids: string[];
  metadata?: Record<string, unknown>;
};




export type LegacyCanonicalMemoryCleanupRequest = {
  max_scan_nodes?: number | null;
  max_groups?: number | null;
  dry_run?: boolean | null;
  mark_aliases_deprecated?: boolean | null;
  reason?: string | null;
  metadata?: Record<string, unknown>;
};

export type LegacyCanonicalMemoryCleanupItem = {
  canonical_source: string;
  target_node_id?: string | null;
  created_canonical_node: boolean;
  merged_node_ids: string[];
  deprecated_node_ids: string[];
  linked_node_ids: string[];
  reason: string;
  metadata?: Record<string, unknown>;
};

export type LegacyCanonicalMemoryCleanupReceipt = {
  accepted: boolean;
  reason: string;
  started_at: number;
  completed_at: number;
  scanned_nodes: number;
  groups_processed: number;
  skipped_groups: number;
  canonical_nodes_created: number;
  canonical_nodes_existing: number;
  alias_nodes_merged: number;
  alias_nodes_deprecated: number;
  items: LegacyCanonicalMemoryCleanupItem[];
  warnings: string[];
  metadata?: Record<string, unknown>;
};

export type MemoryAutopilotRequest = {
  reconsolidation_limit?: number;
  embedding_limit?: number;
  run_skill_extraction?: boolean;
  run_candidate_discovery?: boolean;
  force_embeddings?: boolean;
  run_legacy_canonical_cleanup?: boolean;
  canonical_cleanup_scan_limit?: number;
  canonical_cleanup_group_limit?: number;
  canonical_cleanup_dry_run?: boolean;
  reason?: string | null;
};

export type MemoryAutopilotReceipt = {
  accepted: boolean;
  reason: string;
  started_at: number;
  completed_at: number;
  reconsolidated_candidates: number;
  semantic_nodes_created: number;
  embeddings_indexed: number;
  embeddings_failed: number;
  pending_embeddings_after: number;
  skill_candidates: number;
  duplicate_candidates: number;
  canonical_review_candidates: number;
  canonical_cleanup_groups?: number;
  canonical_cleanup_created?: number;
  canonical_cleanup_merged_aliases?: number;
  canonical_cleanup_deprecated_aliases?: number;
  canonical_cleanup_warnings?: string[];
  quality_score: number;
  quality_status: string;
  repair_plan?: MemoryHealthRepairPlan | null;
  recommendations: string[];
  warnings: string[];
  metadata?: Record<string, unknown>;
};

export type MemoryHybridQueryRequest = MemoryQueryRequest & {
  vector_weight?: number;
  lexical_weight?: number;
  graph_weight?: number;
};

export type MemoryHybridQueryResponse = MemoryQueryResponse & {
  embedding_status: MemoryEmbeddingIndexStatus;
  metadata?: Record<string, unknown>;
};

export type MemorySkillCandidateStatus = "candidate" | "approved" | "disabled" | "deprecated";

export type MemorySkillCandidate = {
  id: string;
  title: string;
  summary: string;
  source_node_id?: string | null;
  status: MemorySkillCandidateStatus;
  confidence: number;
  salience: number;
  trigger_hints: string[];
  required_tools: string[];
  risk_level: string;
  created_at: number;
  updated_at: number;
  approved_by?: string | null;
  approved_at?: number | null;
  metadata?: Record<string, unknown>;
};

export type MemorySkillCandidateExtractionReceipt = {
  accepted: boolean;
  reason: string;
  candidates: MemorySkillCandidate[];
  activation?: MemoryActivation | null;
  metadata?: Record<string, unknown>;
};

export type MemorySkillCandidateUpdateRequest = {
  candidate_id: string;
  status?: MemorySkillCandidateStatus | null;
  confidence?: number | null;
  salience?: number | null;
  add_trigger_hints?: string[];
  remove_trigger_hints?: string[];
  required_tools?: string[] | null;
  risk_level?: string | null;
  approved_by?: string | null;
  reason?: string | null;
  metadata?: Record<string, unknown>;
};

export type MemorySkillCandidateUpdateReceipt = {
  accepted: boolean;
  reason: string;
  candidate: MemorySkillCandidate;
  activation?: MemoryActivation | null;
  metadata?: Record<string, unknown>;
};


export type MemoryReconsolidationCandidate = {
  node: MemoryNode;
  reason: string;
  user_message: string;
  assistant_answer: string;
};

export type MemoryReconsolidationRequest = {
  limit?: number | null;
  include_reprocessed?: boolean;
  dry_run?: boolean;
};

export type MemoryReconsolidationItemReceipt = {
  source_node_id: string;
  accepted: boolean;
  reason: string;
  created_node_ids: string[];
  created_edge_ids: string[];
  semantic_atom_count?: number;
  metadata?: Record<string, unknown>;
};

export type MemoryReconsolidationReceipt = {
  accepted: boolean;
  reason: string;
  scanned_candidates: number;
  processed_candidates: number;
  semantic_nodes_created: number;
  semantic_edges_created: number;
  skipped_candidates: number;
  items: MemoryReconsolidationItemReceipt[];
  activation?: MemoryActivation | null;
  metadata?: Record<string, unknown>;
};

export type MemoryReconsolidationStatus = {
  pending_candidates: number;
  sample_node_ids?: string[];
  metadata_only?: boolean;
};

export type MemoryDuplicateCandidateRequest = {
  limit?: number;
  min_score?: number;
  include_deprecated?: boolean;
  kinds?: MemoryNodeKind[];
};

export type MemoryDuplicateCandidate = {
  canonical_node: MemoryNode;
  duplicate_node: MemoryNode;
  score: number;
  reasons: string[];
  shared_tags?: string[];
  metadata?: Record<string, unknown>;
};

export type MemoryMergeNodesRequest = {
  target_node_id: string;
  source_node_ids: string[];
  mark_sources_deprecated?: boolean;
  actor?: string | null;
  reason?: string | null;
  metadata?: Record<string, unknown>;
};

export type MemoryMergeNodesReceipt = {
  accepted: boolean;
  reason: string;
  target_node: MemoryNode;
  merged_node_ids: string[];
  created_edge_ids: string[];
  activation?: MemoryActivation | null;
  metadata?: Record<string, unknown>;
};

export type MemoryCanonicalReviewRequest = {
  limit?: number;
  min_score?: number;
  include_deprecated?: boolean;
  kinds?: MemoryNodeKind[];
  llm_assist?: boolean;
};

export type MemoryCanonicalReviewCandidate = {
  id: string;
  target_node: MemoryNode;
  candidate_nodes: MemoryNode[];
  confidence: number;
  rationale: string;
  proposed_title: string;
  proposed_summary: string;
  reasons: string[];
  shared_tags?: string[];
  metadata?: Record<string, unknown>;
};

export type MemoryCanonicalReviewApplyRequest = {
  candidate: MemoryCanonicalReviewCandidate;
  mark_sources_deprecated?: boolean;
  actor?: string | null;
  reason?: string | null;
  metadata?: Record<string, unknown>;
};

export type MemoryJobQueueRuntimeSnapshot = {
  job_id: string;
  kind: string;
  status: string;
  dedup_key?: string | null;
  enqueued_at: string;
  started_at?: string | null;
  age_ms: number;
  metadata?: Record<string, unknown>;
};

export type MemoryJobQueueEvent = {
  at: string;
  event: string;
  job_id?: string | null;
  kind?: string | null;
  dedup_key?: string | null;
  reason?: string | null;
  metadata?: Record<string, unknown>;
};

export type MemoryJobQueueSnapshot = {
  schema_version: number;
  status: "healthy" | "backpressured" | "saturated" | "degraded" | string;
  max_pending: number;
  max_concurrency: number;
  queued: number;
  running: number;
  pressure_ratio: number;
  concurrency_ratio: number;
  accepted_total: number;
  completed_total: number;
  failed_total: number;
  rejected_full_total: number;
  rejected_duplicate_total: number;
  failed_dispatch_total: number;
  last_event_at?: string | null;
  last_rejection_reason?: string | null;
  active_jobs: MemoryJobQueueRuntimeSnapshot[];
  recent_events: MemoryJobQueueEvent[];
};

export type MemoryJobSubmissionReceipt = {
  accepted: boolean;
  reason: string;
  job_id?: string | null;
  kind: string;
  dedup_key?: string | null;
  queued: number;
  running: number;
  max_pending: number;
  max_concurrency: number;
  submitted_at: string;
  metadata?: Record<string, unknown>;
};

export type MemoryRagIntegrityReport = {
  schema_version: number;
  generated_at: number;
  readiness: "enterprise_ready" | "ready_with_warnings" | "needs_hardening" | "blocked" | string;
  score: number;
  summary: string;
  checks?: Record<string, unknown>;
  blockers: string[];
  warnings: string[];
  strengths: string[];
  next_actions: string[];
  graph_status?: Record<string, unknown>;
  quality?: MemoryQualityDashboard | null;
  embedding_status?: MemoryEmbeddingIndexStatus | null;
  queue?: MemoryJobQueueSnapshot | null;
  governance_policy?: MemoryGovernancePolicySnapshot | null;
  metadata?: Record<string, unknown>;
};

export type MemoryRagCloseoutGate = {
  id: string;
  title: string;
  status: "pass" | "warn" | "block" | string;
  severity: "required" | "important" | string;
  summary: string;
  evidence?: Record<string, unknown>;
  next_action?: string | null;
  metadata_only?: boolean;
};

export type MemoryRagCloseoutSnapshotRequest = {
  allow_autopilot?: boolean;
  allow_skill_extraction?: boolean;
};

export type MemoryRagCloseoutSnapshot = {
  schema_version: number;
  generated_at: number;
  status: "closeout_ready" | "ready_with_warnings" | "needs_maintenance" | "blocked" | string;
  release_recommendation: string;
  summary: {
    quality_score?: number | null;
    quality_score_percent?: number | null;
    semantic_ratio?: number | null;
    embedding_coverage_ratio?: number | null;
    pending_embeddings?: number;
    pending_reconsolidation?: number;
    recent_activations?: number;
    embedding_provider?: string | null;
    queue_status?: string;
    graph_nodes?: number;
    graph_chunks?: number;
    [key: string]: unknown;
  };
  gate_counts: {
    pass: number;
    warn: number;
    block: number;
  };
  gates: MemoryRagCloseoutGate[];
  blockers: string[];
  warnings: string[];
  strengths: string[];
  next_actions: string[];
  recommended_queue_command: string;
  recommended_queue_request?: MemoryRagRecommendedMaintenanceRequest | Record<string, unknown>;
  control_center_commands?: Record<string, string>;
  quality?: MemoryQualityDashboard | null;
  embedding_status?: MemoryEmbeddingIndexStatus | null;
  queue?: MemoryJobQueueSnapshot | null;
  graph_status?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
};

export type MemoryRagRecommendedMaintenanceRequest = {
  dry_run?: boolean;
  max_actions?: number;
  allow_autopilot?: boolean;
  allow_skill_extraction?: boolean;
  reason?: string | null;
};

export type MemoryRagPlannedMaintenanceAction = {
  kind: string;
  priority: "critical" | "high" | "medium" | "low" | string;
  risk_level: "low" | "medium" | "high" | string;
  limit?: number;
  affected_count?: number;
  reason?: string;
  queued_command?: string;
  metadata_only?: boolean;
  [key: string]: unknown;
};

export type MemoryRagRecommendedMaintenanceReceipt = {
  schema_version: number;
  generated_at: number;
  status: "planned" | "queued" | "blocked" | "no_action_needed" | "not_queued" | string;
  dry_run: boolean;
  max_actions: number;
  planned_actions: MemoryRagPlannedMaintenanceAction[];
  submissions: MemoryJobSubmissionReceipt[];
  accepted_count: number;
  blockers: string[];
  queue_before: MemoryJobQueueSnapshot;
  queue_after: MemoryJobQueueSnapshot;
  quality?: MemoryQualityDashboard | null;
  embedding_status?: MemoryEmbeddingIndexStatus | null;
  metadata?: Record<string, unknown>;
};

