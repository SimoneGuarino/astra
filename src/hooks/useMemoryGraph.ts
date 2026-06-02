import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type {
  MemoryActivation,
  MemoryActivationEventPayload,
  MemoryGraphSnapshot,
  MemoryGraphStatus,
  MemoryQualityDashboard,
  MemoryQueryRequest,
  MemoryQueryResponse,
  MemoryEmbeddingIndexStatus,
  MemoryEmbeddingRebuildReceipt,
  MemoryEmbeddingRebuildRequest,
  MemoryEmbeddingMaintenanceReceipt,
  MemoryEmbeddingMaintenanceRequest,
  MemoryAutopilotRequest,
  MemoryAutopilotReceipt,
  MemoryHybridQueryRequest,
  MemoryHybridQueryResponse,
  ConversationMemoryBundle,
  ConversationMemoryConsolidationReceipt,
  ResearchMemoryBundle,
  ResearchMemoryConsolidationReceipt,
  MemorySkillCandidate,
  MemorySkillCandidateExtractionReceipt,
  MemorySkillCandidateUpdateRequest,
  MemorySkillCandidateUpdateReceipt,
  MemoryReconsolidationCandidate,
  MemoryReconsolidationReceipt,
  MemoryReconsolidationRequest,
  MemoryReconsolidationStatus,
  MemoryGovernancePolicySnapshot,
  MemoryNodeGovernanceUpdateRequest,
  MemoryNodeGovernanceUpdateReceipt,
  MemoryDuplicateCandidate,
  MemoryDuplicateCandidateRequest,
  MemoryCanonicalReviewCandidate,
  MemoryCanonicalReviewRequest,
  MemoryCanonicalReviewApplyRequest,
  MemoryMergeNodesRequest,
  MemoryMergeNodesReceipt,
  MemoryJobQueueSnapshot,
  MemoryRagIntegrityReport,
  MemoryRagCloseoutSnapshot,
  MemoryRagCloseoutSnapshotRequest,
  MemoryRagRecommendedMaintenanceRequest,
  MemoryRagRecommendedMaintenanceReceipt,
  DeepSearchRequest,
  DeepSearchReceipt,
  DeepSearchKnowledgeRefreshRequest,
  DeepSearchKnowledgeRefreshReceipt,
} from "../types/memory";

export function useMemoryGraph() {
  const [lastActivationEvent, setLastActivationEvent] = useState<MemoryActivationEventPayload | null>(null);

  const getStatus = useCallback(() => invoke<MemoryGraphStatus>("get_memory_graph_status"), []);

  const getGovernancePolicy = useCallback(
    () => invoke<MemoryGovernancePolicySnapshot>("get_memory_governance_policy"),
    []
  );

  const updateNodeGovernance = useCallback(
    (request: MemoryNodeGovernanceUpdateRequest) =>
      invoke<MemoryNodeGovernanceUpdateReceipt>("update_memory_node_governance", { request }),
    []
  );




  const listCanonicalReviewCandidates = useCallback(
    (request: MemoryCanonicalReviewRequest = { limit: 40, min_score: 0.62, include_deprecated: false, llm_assist: true }) =>
      invoke<MemoryCanonicalReviewCandidate[]>("list_memory_canonical_review_candidates", { request }),
    []
  );

  const applyCanonicalReview = useCallback(
    (request: MemoryCanonicalReviewApplyRequest) =>
      invoke<MemoryMergeNodesReceipt>("apply_memory_canonical_review", { request }),
    []
  );

  const listDuplicateCandidates = useCallback(
    (request: MemoryDuplicateCandidateRequest = { limit: 80, min_score: 0.72, include_deprecated: false }) =>
      invoke<MemoryDuplicateCandidate[]>("list_memory_duplicate_candidates", { request }),
    []
  );

  const mergeMemoryNodes = useCallback(
    (request: MemoryMergeNodesRequest) =>
      invoke<MemoryMergeNodesReceipt>("merge_memory_nodes", { request }),
    []
  );

  const exportSnapshot = useCallback(
    (limit = 180) => invoke<MemoryGraphSnapshot>("export_memory_graph_snapshot", { limit }),
    []
  );

  const query = useCallback(
    (request: MemoryQueryRequest) => invoke<MemoryQueryResponse>("query_memory_graph", { request }),
    []
  );

  const queryHybrid = useCallback(
    (request: MemoryHybridQueryRequest) => invoke<MemoryHybridQueryResponse>("query_memory_graph_hybrid", { request }),
    []
  );


  const getQualityDashboard = useCallback(
    () => invoke<MemoryQualityDashboard>("get_memory_quality_dashboard"),
    []
  );

  const getEmbeddingStatus = useCallback(
    () => invoke<MemoryEmbeddingIndexStatus>("get_memory_embedding_status"),
    []
  );


  const getMemoryJobQueueStatus = useCallback(
    () => invoke<MemoryJobQueueSnapshot>("get_memory_job_queue_status"),
    []
  );

  const getMemoryRagIntegrityReport = useCallback(
    () => invoke<MemoryRagIntegrityReport>("get_memory_rag_integrity_report"),
    []
  );

  const getMemoryRagCloseoutSnapshot = useCallback(
    (request: MemoryRagCloseoutSnapshotRequest = { allow_autopilot: false, allow_skill_extraction: false }) =>
      invoke<MemoryRagCloseoutSnapshot>("get_memory_rag_closeout_snapshot", { request }),
    []
  );

  const queueRecommendedMemoryMaintenance = useCallback(
    (request: MemoryRagRecommendedMaintenanceRequest = {
      dry_run: false,
      max_actions: 1,
      allow_autopilot: false,
      allow_skill_extraction: false,
      reason: "memory_graph_control_center_recommended_maintenance",
    }) => invoke<MemoryRagRecommendedMaintenanceReceipt>("queue_memory_rag_recommended_maintenance", { request }),
    []
  );

  const rebuildEmbeddingIndex = useCallback(
    (request: MemoryEmbeddingRebuildRequest = {}) =>
      invoke<MemoryEmbeddingRebuildReceipt>("rebuild_memory_embedding_index", { request }),
    []
  );


  const runMemoryAutopilot = useCallback(
    (request: MemoryAutopilotRequest = {
      reconsolidation_limit: 12,
      embedding_limit: 48,
      run_skill_extraction: true,
      run_candidate_discovery: true,
      force_embeddings: false,
      run_legacy_canonical_cleanup: true,
      canonical_cleanup_scan_limit: 1600,
      canonical_cleanup_group_limit: 36,
      canonical_cleanup_dry_run: false,
      auto_apply_safe_review_proposals: true,
      auto_apply_review_limit: 16,
      duplicate_auto_apply_min_score: 0.9,
      canonical_auto_apply_min_score: 0.88,
      reason: "memory_graph_user_autopilot",
    }) => invoke<MemoryAutopilotReceipt>("run_memory_autopilot", { request }),
    []
  );

  const runEmbeddingMaintenance = useCallback(
    (request: MemoryEmbeddingMaintenanceRequest = { limit: 24, force: false, reason: "memory_graph_ui" }) =>
      invoke<MemoryEmbeddingMaintenanceReceipt>("run_memory_embedding_maintenance", { request }),
    []
  );

  const runDeepSearch = useCallback(
    (request: DeepSearchRequest) =>
      invoke<DeepSearchReceipt>("run_memory_deep_search", { request }),
    []
  );

  const runDeepSearchKnowledgeRefresh = useCallback(
    (request: DeepSearchKnowledgeRefreshRequest = {
      enabled: true,
      dry_run: false,
      snapshot_limit: 320,
      max_candidates: 24,
      stale_after_days: 45,
      temporal_stale_after_days: 7,
      include_low_confidence_candidates: true,
      tag_candidates_for_refresh: true,
      run_refresh_research: true,
      max_refresh_topics: 8,
      max_refresh_runs: 3,
      max_sources_per_topic: 8,
    }) =>
      invoke<DeepSearchKnowledgeRefreshReceipt>("run_deep_search_knowledge_refresh", { request }),
    []
  );

  const getRecentActivations = useCallback(
    (limit = 25) => invoke<MemoryActivation[]>("get_recent_memory_activations", { limit }),
    []
  );

  const consolidateResearchBundle = useCallback(
    (bundle: ResearchMemoryBundle) =>
      invoke<ResearchMemoryConsolidationReceipt>("consolidate_research_memory_bundle", { bundle }),
    []
  );

  const extractSkillCandidates = useCallback(
    (limit = 80) => invoke<MemorySkillCandidateExtractionReceipt>("extract_memory_skill_candidates", { limit }),
    []
  );

  const listSkillCandidates = useCallback(
    (includeDisabled = false, limit = 80) =>
      invoke<MemorySkillCandidate[]>("list_memory_skill_candidates", { includeDisabled, limit }),
    []
  );

  const updateSkillCandidate = useCallback(
    (request: MemorySkillCandidateUpdateRequest) =>
      invoke<MemorySkillCandidateUpdateReceipt>("update_memory_skill_candidate", { request }),
    []
  );

  const consolidateConversationBundle = useCallback(
    (bundle: ConversationMemoryBundle) =>
      invoke<ConversationMemoryConsolidationReceipt>("consolidate_conversation_memory_bundle", { bundle }),
    []
  );

  const getReconsolidationStatus = useCallback(
    (limit = 80) => invoke<MemoryReconsolidationStatus>("get_memory_reconsolidation_status", { limit }),
    []
  );

  const listReconsolidationCandidates = useCallback(
    (limit = 80, includeReprocessed = false) =>
      invoke<MemoryReconsolidationCandidate[]>("list_memory_reconsolidation_candidates", { limit, includeReprocessed }),
    []
  );

  const reconsolidateMemoryCandidates = useCallback(
    (request: MemoryReconsolidationRequest = { limit: 80, include_reprocessed: false, dry_run: false }) =>
      invoke<MemoryReconsolidationReceipt>("reconsolidate_memory_candidates", { request }),
    []
  );

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;

    listen<MemoryActivationEventPayload>("memory-activation", (event) => {
      setLastActivationEvent(event.payload);
    })
      .then((dispose) => {
        if (cancelled) {
          dispose();
          return;
        }
        unlisten = dispose;
      })
      .catch((error) => {
        console.debug("Astra memory activation listener failed:", error);
      });

    return () => {
      cancelled = true;
      if (unlisten) unlisten();
    };
  }, []);

  return useMemo(
    () => ({
      consolidateConversationBundle,
      listCanonicalReviewCandidates,
      applyCanonicalReview,
      listDuplicateCandidates,
      mergeMemoryNodes,
      consolidateResearchBundle,
      exportSnapshot,
      getRecentActivations,
      getQualityDashboard,
      getStatus,
      getGovernancePolicy,
      updateNodeGovernance,
      getReconsolidationStatus,
      listReconsolidationCandidates,
      reconsolidateMemoryCandidates,
      extractSkillCandidates,
      listSkillCandidates,
      updateSkillCandidate,
      lastActivationEvent,
      query,
      queryHybrid,
      getEmbeddingStatus,
      getMemoryJobQueueStatus,
      getMemoryRagIntegrityReport,
      getMemoryRagCloseoutSnapshot,
      queueRecommendedMemoryMaintenance,
      rebuildEmbeddingIndex,
      runMemoryAutopilot,
      runEmbeddingMaintenance,
      runDeepSearch,
      runDeepSearchKnowledgeRefresh,
    }),
    [
      consolidateConversationBundle,
      listCanonicalReviewCandidates,
      applyCanonicalReview,
      listDuplicateCandidates,
      mergeMemoryNodes,
      consolidateResearchBundle,
      exportSnapshot,
      extractSkillCandidates,
      getEmbeddingStatus,
      getMemoryJobQueueStatus,
      getMemoryRagIntegrityReport,
      getMemoryRagCloseoutSnapshot,
      queueRecommendedMemoryMaintenance,
      runMemoryAutopilot,
      getRecentActivations,
      getQualityDashboard,
      getStatus,
      getGovernancePolicy,
      updateNodeGovernance,
      getReconsolidationStatus,
      listReconsolidationCandidates,
      reconsolidateMemoryCandidates,
      extractSkillCandidates,
      listSkillCandidates,
      updateSkillCandidate,
      lastActivationEvent,
      query,
      queryHybrid,
      rebuildEmbeddingIndex,
      runEmbeddingMaintenance,
      runDeepSearch,
      runDeepSearchKnowledgeRefresh,
    ]
  );
}
