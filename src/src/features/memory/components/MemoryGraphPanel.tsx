import { useCallback, useEffect, useMemo, useState } from "react";
import { IoClose } from "react-icons/io5";
import { useMemoryGraph } from "../../../hooks/useMemoryGraph";
import {
  MemoryGraphControlsOverlay,
  DEFAULT_MEMORY_GRAPH_LAYOUT_SETTINGS,
  DEFAULT_MEMORY_GRAPH_VIEW_SETTINGS,
  MEMORY_GRAPH_LAYOUT_PRESETS,
  type MemoryGraphLabelMode,
  type MemoryGraphLayoutSettings,
  type MemoryGraphViewSettings,
} from "./MemoryGraphControlsOverlay";
import { MemoryGraphSearchOverlay } from "./MemoryGraphSearchOverlay";
import { MemoryGraphStatusBar } from "./MemoryGraphStatusBar";
import { MemoryGraphToolbar } from "./MemoryGraphToolbar";
import { MemoryQualityOverlay } from "./MemoryQualityOverlay";
import type {
  MemoryActivation,
  MemoryActivationEventPayload,
  MemoryEdge,
  MemoryEmbeddingIndexStatus,
  MemoryEmbeddingRebuildReceipt,
  MemoryGraphSnapshot,
  MemoryGraphStatus,
  MemoryNode,
  MemoryQualityDashboard,
  MemoryJobQueueSnapshot,
  MemoryRagCloseoutSnapshot,
  MemoryRagIntegrityReport,
  MemoryRagRecommendedMaintenanceReceipt,
  MemoryDuplicateCandidate,
  MemoryCanonicalReviewCandidate,
  MemoryQueryHit,
  MemorySkillCandidate,
  MemoryVerificationStatus,
} from "../../../types/memory";
import {
  MAX_VISIBLE_GRAPH_EDGES,
  MAX_VISIBLE_GRAPH_NODES,
  NODE_KIND_LABELS,
} from "../layout/memoryGraphLayoutTypes";
import {
  activationToPayload,
  clampNumber,
  computeLocalGraphNodeIds,
  memoryNodeVisualWeight,
} from "../layout/memoryGraphLayoutEngine";
import { MemoryGraphCanvas } from "./MemoryGraphCanvas";
import { MemoryActivationSummary } from "./MemoryActivationSummary";
import { MemoryActivationTimeline } from "./MemoryActivationTimeline";
import { MemoryEmbeddingSummary } from "./MemoryEmbeddingSummary";
import { MemoryNodeInspector } from "./MemoryNodeInspector";
import { MemorySearchResults } from "./MemorySearchResults";
import { MemorySkillCandidatePanel } from "./MemorySkillCandidatePanel";
import { MemoryRagControlCenter } from "./MemoryRagControlCenter";

type MemoryGraphPanelProps = {
  mode?: "embedded" | "immersive";
  onClose?: () => void;
};

export function MemoryGraphPanel({ mode = "embedded", onClose }: MemoryGraphPanelProps) {
  const immersive = mode === "immersive";
  const memory = useMemoryGraph();
  const [status, setStatus] = useState<MemoryGraphStatus | null>(null);
  const [snapshot, setSnapshot] = useState<MemoryGraphSnapshot | null>(null);
  const [activations, setActivations] = useState<MemoryActivation[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [queryText, setQueryText] = useState("");
  const [queryExpanded, setQueryExpanded] = useState(false);
  const [qualityExpanded, setQualityExpanded] = useState(false);
  const [reviewExpanded, setReviewExpanded] = useState(false);
  const [controlsExpanded, setControlsExpanded] = useState(false);
  const [layoutSettings, setLayoutSettings] = useState<MemoryGraphLayoutSettings>(() => readMemoryGraphLayoutSettings());
  const [viewSettings, setViewSettings] = useState<MemoryGraphViewSettings>(() => readMemoryGraphViewSettings());
  const [qualityDashboard, setQualityDashboard] = useState<MemoryQualityDashboard | null>(null);
  const [memoryAutopilotStatus, setMemoryAutopilotStatus] = useState<string | null>(null);
  const [memoryRagCloseout, setMemoryRagCloseout] = useState<MemoryRagCloseoutSnapshot | null>(null);
  const [memoryRagIntegrity, setMemoryRagIntegrity] = useState<MemoryRagIntegrityReport | null>(null);
  const [memoryJobQueue, setMemoryJobQueue] = useState<MemoryJobQueueSnapshot | null>(null);
  const [recommendedMaintenanceReceipt, setRecommendedMaintenanceReceipt] = useState<MemoryRagRecommendedMaintenanceReceipt | null>(null);
  const [memoryControlCenterStatus, setMemoryControlCenterStatus] = useState<string | null>(null);
  const [labelsVisible, setLabelsVisible] = useState(() => readMemoryGraphLabelsPreference());

  useEffect(() => {
    writeMemoryGraphLabelsPreference(labelsVisible);
  }, [labelsVisible]);

  useEffect(() => {
    writeMemoryGraphLayoutSettings(layoutSettings);
    setLabelsVisible(layoutSettings.labelMode !== "hidden");
  }, [layoutSettings]);

  useEffect(() => {
    writeMemoryGraphViewSettings(viewSettings);
  }, [viewSettings]);

  const handleToggleLabels = useCallback(() => {
    setLayoutSettings((current) => ({
      ...current,
      labelMode: current.labelMode === "hidden" ? "important" : "hidden",
    }));
  }, []);

  const handleLayoutSettingsChange = useCallback((next: MemoryGraphLayoutSettings) => {
    setLayoutSettings(sanitizeMemoryGraphLayoutSettings(next));
  }, []);

  const handleViewSettingsChange = useCallback((next: MemoryGraphViewSettings) => {
    setViewSettings(sanitizeMemoryGraphViewSettings(next));
  }, []);
  const [queryHits, setQueryHits] = useState<MemoryQueryHit[]>([]);
  const [embeddingStatus, setEmbeddingStatus] = useState<MemoryEmbeddingIndexStatus | null>(null);
  const [embeddingReceipt, setEmbeddingReceipt] = useState<MemoryEmbeddingRebuildReceipt | null>(null);
  const [embeddingMaintenanceStatus, setEmbeddingMaintenanceStatus] = useState<string | null>(null);
  const [skillCandidates, setSkillCandidates] = useState<MemorySkillCandidate[]>([]);
  const [duplicateCandidates, setDuplicateCandidates] = useState<MemoryDuplicateCandidate[]>([]);
  const [canonicalReviewCandidates, setCanonicalReviewCandidates] = useState<MemoryCanonicalReviewCandidate[]>([]);
  const [mergeStatus, setMergeStatus] = useState<string | null>(null);
  const [canonicalReviewStatus, setCanonicalReviewStatus] = useState<string | null>(null);
  const [skillStatus, setSkillStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [governanceStatus, setGovernanceStatus] = useState<string | null>(null);
  const [isUpdatingGovernance, setIsUpdatingGovernance] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const [
        nextStatus,
        nextSnapshot,
        nextActivations,
        nextEmbeddingStatus,
        nextSkills,
        nextQuality,
        nextQueue,
        nextIntegrity,
        nextCloseout,
      ] = await Promise.all([
        memory.getStatus(),
        memory.exportSnapshot(800),
        memory.getRecentActivations(40),
        memory.getEmbeddingStatus().catch(() => null),
        memory.listSkillCandidates(false, 120).catch(() => []),
        memory.getQualityDashboard().catch(() => null),
        memory.getMemoryJobQueueStatus().catch(() => null),
        memory.getMemoryRagIntegrityReport().catch(() => null),
        memory.getMemoryRagCloseoutSnapshot({ allow_autopilot: false, allow_skill_extraction: false }).catch(() => null),
      ]);
      setStatus(nextStatus);
      setSnapshot(nextSnapshot);
      setActivations(nextActivations);
      setEmbeddingStatus(nextEmbeddingStatus);
      setSkillCandidates(nextSkills);
      setQualityDashboard(nextQuality);
      setMemoryJobQueue(nextQueue);
      setMemoryRagIntegrity(nextIntegrity);
      setMemoryRagCloseout(nextCloseout);
      if (nextEmbeddingStatus && nextEmbeddingStatus.pending_chunks > 0) {
        setEmbeddingMaintenanceStatus(`${nextEmbeddingStatus.pending_chunks} chunks pending; use governed recommended maintenance to queue indexing.`);
      }
      setSelectedNodeId((current) => {
        if (!current) return null;
        return nextSnapshot.nodes.some((node) => node.id === current) ? current : null;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }, [memory]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!memory.lastActivationEvent) return;
    void refresh();
  }, [memory.lastActivationEvent, refresh]);

  const activePayload = useMemo(() => {
    if (memory.lastActivationEvent) return memory.lastActivationEvent;
    const latest = activations[0];
    if (!latest) return null;
    return activationToPayload(latest);
  }, [activations, memory.lastActivationEvent]);

  const nodeById = useMemo(() => {
    const map = new Map<string, MemoryNode>();
    for (const node of snapshot?.nodes ?? []) map.set(node.id, node);
    return map;
  }, [snapshot?.nodes]);

  const selectedNode = selectedNodeId ? nodeById.get(selectedNodeId) ?? null : null;

  const availableNodeKinds = useMemo(() => {
    const kinds = new Set<string>();
    for (const node of snapshot?.nodes ?? []) kinds.add(node.kind);
    return [...kinds].sort((left, right) => (NODE_KIND_LABELS[left] ?? left).localeCompare(NODE_KIND_LABELS[right] ?? right));
  }, [snapshot?.nodes]);

  const graphDegreeByNodeId = useMemo(() => {
    const map = new Map<string, number>();
    for (const edge of snapshot?.edges ?? []) {
      map.set(edge.from_node_id, (map.get(edge.from_node_id) ?? 0) + 1);
      map.set(edge.to_node_id, (map.get(edge.to_node_id) ?? 0) + 1);
    }
    return map;
  }, [snapshot?.edges]);

  const visibleNodes = useMemo(() => {
    const allNodes = snapshot?.nodes ?? [];
    const allEdges = snapshot?.edges ?? [];
    const activeIds = new Set(activePayload?.activated_node_ids ?? []);
    const selectedId = selectedNodeId;
    const kindVisible = (node: MemoryNode) => viewSettings.visibleKinds[node.kind] !== false || node.id === selectedId || activeIds.has(node.id);
    const orphanVisible = (node: MemoryNode) => viewSettings.showIsolatedNodes || (graphDegreeByNodeId.get(node.id) ?? 0) > 0 || node.id === selectedId || activeIds.has(node.id);

    let candidates = allNodes.filter((node) => kindVisible(node) && orphanVisible(node));

    if (viewSettings.mode === "local") {
      const rootIds = selectedId
        ? [selectedId]
        : activeIds.size > 0
          ? [...activeIds].slice(0, 8)
          : candidates
              .slice()
              .sort((left, right) => memoryNodeVisualWeight(right) - memoryNodeVisualWeight(left))
              .slice(0, 1)
              .map((node) => node.id);
      const localIds = computeLocalGraphNodeIds(rootIds, allEdges, Math.max(1, Math.round(viewSettings.localDepth)));
      if (localIds.size > 0) {
        candidates = candidates.filter((node) => localIds.has(node.id));
      }
    }

    if (candidates.length <= MAX_VISIBLE_GRAPH_NODES) return candidates;
    return [...candidates]
      .sort((left, right) => {
        const leftBoost = (activeIds.has(left.id) ? 1000 : 0) + (left.id === selectedId ? 800 : 0);
        const rightBoost = (activeIds.has(right.id) ? 1000 : 0) + (right.id === selectedId ? 800 : 0);
        const leftScore = leftBoost + memoryNodeVisualWeight(left);
        const rightScore = rightBoost + memoryNodeVisualWeight(right);
        return rightScore - leftScore;
      })
      .slice(0, MAX_VISIBLE_GRAPH_NODES);
  }, [activePayload?.activated_node_ids, graphDegreeByNodeId, selectedNodeId, snapshot?.edges, snapshot?.nodes, viewSettings]);

  const visibleNodeIds = useMemo(() => new Set(visibleNodes.map((node) => node.id)), [visibleNodes]);
  const visibleEdges = useMemo(
    () => (snapshot?.edges ?? [])
      .filter((edge) => visibleNodeIds.has(edge.from_node_id) && visibleNodeIds.has(edge.to_node_id))
      .sort((left, right) => right.weight - left.weight)
      .slice(0, MAX_VISIBLE_GRAPH_EDGES),
    [snapshot?.edges, visibleNodeIds]
  );

  const handleSearch = useCallback(async () => {
    const trimmed = queryText.trim();
    if (!trimmed) {
      setQueryHits([]);
      return;
    }
    try {
      setError(null);
      const response = await memory.queryHybrid({
        query: trimmed,
        include_edges: true,
        limit: 18,
        vector_weight: 0.48,
        lexical_weight: 0.32,
        graph_weight: 0.2,
      });
      setEmbeddingStatus(response.embedding_status);
      setQueryHits(response.hits);
      setSelectedNodeId(response.hits[0]?.node.id ?? selectedNodeId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [memory, queryText, selectedNodeId]);

  const loadMemoryReviewQueue = useCallback(async (open = false) => {
    const [duplicates, canonicalReviews] = await Promise.all([
      memory.listDuplicateCandidates({ limit: 80, min_score: 0.7, include_deprecated: false }).catch((err) => {
        setMergeStatus(err instanceof Error ? err.message : String(err));
        return [] as MemoryDuplicateCandidate[];
      }),
      memory.listCanonicalReviewCandidates({
        limit: 40,
        min_score: 0.62,
        include_deprecated: false,
        llm_assist: true,
      }).catch((err) => {
        setCanonicalReviewStatus(err instanceof Error ? err.message : String(err));
        return [] as MemoryCanonicalReviewCandidate[];
      }),
    ]);
    setDuplicateCandidates(duplicates);
    setCanonicalReviewCandidates(canonicalReviews);
    setMergeStatus(`${duplicates.length} duplicate candidate${duplicates.length === 1 ? "" : "s"}`);
    setCanonicalReviewStatus(`${canonicalReviews.length} canonical review candidate${canonicalReviews.length === 1 ? "" : "s"}`);
    if (open) setReviewExpanded(true);
    return { duplicates, canonicalReviews };
  }, [memory]);

  const handleToggleReviewQueue = useCallback(async () => {
    const next = !reviewExpanded;
    setReviewExpanded(next);
    if (next) {
      try {
        setError(null);
        setIsLoading(true);
        await loadMemoryReviewQueue(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setIsLoading(false);
      }
    }
  }, [loadMemoryReviewQueue, reviewExpanded]);


  const refreshMemoryControlCenter = useCallback(async () => {
    const [nextQueue, nextIntegrity, nextCloseout, nextEmbeddingStatus, nextQuality] = await Promise.all([
      memory.getMemoryJobQueueStatus().catch(() => null),
      memory.getMemoryRagIntegrityReport().catch(() => null),
      memory.getMemoryRagCloseoutSnapshot({ allow_autopilot: false, allow_skill_extraction: false }).catch(() => null),
      memory.getEmbeddingStatus().catch(() => null),
      memory.getQualityDashboard().catch(() => null),
    ]);
    setMemoryJobQueue(nextQueue);
    setMemoryRagIntegrity(nextIntegrity);
    setMemoryRagCloseout(nextCloseout);
    setEmbeddingStatus(nextEmbeddingStatus);
    setQualityDashboard(nextQuality);
  }, [memory]);

  const handlePlanRecommendedMaintenance = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      setMemoryControlCenterStatus("Planning recommended memory/RAG maintenance…");
      const receipt = await memory.queueRecommendedMemoryMaintenance({
        dry_run: true,
        max_actions: 1,
        allow_autopilot: false,
        allow_skill_extraction: false,
        reason: "memory_graph_control_center_plan_only",
      });
      setRecommendedMaintenanceReceipt(receipt);
      setMemoryJobQueue(receipt.queue_after ?? receipt.queue_before ?? null);
      setMemoryControlCenterStatus(
        receipt.planned_actions.length
          ? `Plan ready: ${receipt.planned_actions.map((action) => action.kind).join(", ")}`
          : receipt.status === "blocked"
            ? `Planning blocked: ${receipt.blockers.join("; ")}`
            : "No recommended maintenance is currently needed"
      );
      await refreshMemoryControlCenter();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMemoryControlCenterStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [memory, refreshMemoryControlCenter]);

  const handleRunRecommendedMaintenance = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      setMemoryControlCenterStatus("Queueing governed recommended maintenance…");
      const receipt = await memory.queueRecommendedMemoryMaintenance({
        dry_run: false,
        max_actions: 1,
        allow_autopilot: false,
        allow_skill_extraction: false,
        reason: "memory_graph_control_center_run_recommended",
      });
      setRecommendedMaintenanceReceipt(receipt);
      setMemoryJobQueue(receipt.queue_after ?? receipt.queue_before ?? null);
      const accepted = receipt.submissions.filter((submission) => submission.accepted).length;
      setMemoryControlCenterStatus(
        receipt.status === "blocked"
          ? `Maintenance blocked: ${receipt.blockers.join("; ")}`
          : accepted > 0
            ? `Queued ${accepted} bounded memory job${accepted === 1 ? "" : "s"}`
            : receipt.status === "no_action_needed"
              ? "No recommended maintenance is currently needed"
              : `Maintenance status: ${receipt.status}`
      );
      await refreshMemoryControlCenter();
      await refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMemoryControlCenterStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh, refreshMemoryControlCenter]);

  const handleRunMemoryAutopilot = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      setMemoryAutopilotStatus("Autopilot running…");
      const receipt = await memory.runMemoryAutopilot({
        reconsolidation_limit: 12,
        embedding_limit: 64,
        run_skill_extraction: true,
        run_candidate_discovery: true,
        force_embeddings: false,
        run_legacy_canonical_cleanup: true,
        canonical_cleanup_scan_limit: 1200,
        canonical_cleanup_group_limit: 24,
        canonical_cleanup_dry_run: false,
        reason: "memory_graph_toolbar_autopilot",
      });
      const autoActions = receipt.repair_plan?.automatic_action_count ?? 0;
      const reviewActions = receipt.repair_plan?.review_action_count ?? 0;
      const cleanupMerged = receipt.canonical_cleanup_merged_aliases ?? 0;
      const cleanupCreated = receipt.canonical_cleanup_created ?? 0;
      setMemoryAutopilotStatus(
        `Autopilot: quality ${Math.round(receipt.quality_score * 100)}%, ${receipt.semantic_nodes_created} semantic, ${receipt.embeddings_indexed} vectors, ${cleanupCreated} canonical / ${cleanupMerged} aliases, ${autoActions} auto / ${reviewActions} review`
      );
      const reviewCount = receipt.duplicate_candidates + receipt.canonical_review_candidates;
      setSkillStatus(
        `Autopilot completed: ${receipt.skill_candidates} skills, ${receipt.duplicate_candidates} duplicate proposals, ${receipt.canonical_review_candidates} canonical proposals, ${receipt.canonical_cleanup_deprecated_aliases ?? 0} legacy aliases deprecated`
      );
      await refresh();
      if (reviewCount > 0) {
        await loadMemoryReviewQueue(true);
      }
      const [nextEmbeddingStatus, nextQuality, nextSkills] = await Promise.all([
        memory.getEmbeddingStatus().catch(() => null),
        memory.getQualityDashboard().catch(() => null),
        memory.listSkillCandidates(false, 120).catch(() => []),
      ]);
      setEmbeddingStatus(nextEmbeddingStatus);
      setQualityDashboard(nextQuality);
      setSkillCandidates(nextSkills);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMemoryAutopilotStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [loadMemoryReviewQueue, memory, refresh]);

  const handleRunKnowledgeRefresh = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      setMemoryControlCenterStatus("Refreshing stale knowledge…");
      const receipt = await memory.runDeepSearchKnowledgeRefresh({
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
      });
      setMemoryControlCenterStatus(
        `Knowledge refresh: ${receipt.stale_candidates} stale candidates, ${receipt.refresh_runs} refresh runs, ${receipt.claims_promoted} promoted claims`
      );
      await refresh();
      await refreshMemoryControlCenter();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMemoryControlCenterStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh, refreshMemoryControlCenter]);

  const handleExtractSkills = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const receipt = await memory.extractSkillCandidates(120);
      setSkillStatus(`${receipt.candidates.length} candidate skill estratte`);
      const nextSkills = await memory.listSkillCandidates(false, 120);
      setSkillCandidates(nextSkills);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh]);


  /*const handleReconsolidateMemory = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const receipt = await memory.reconsolidateMemoryCandidates({
        limit: 80,
        include_reprocessed: false,
        dry_run: false,
      });
      setSkillStatus(
        `Reconsolidation: ${receipt.processed_candidates} candidate, ${receipt.semantic_nodes_created} semantic node`
      );
      await refresh();
      const nextEmbeddingStatus = await memory.getEmbeddingStatus().catch(() => null);
      setEmbeddingStatus(nextEmbeddingStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh]);*/

  /*const handleFindCanonicalReviews = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const candidates = await memory.listCanonicalReviewCandidates({
        limit: 40,
        min_score: 0.62,
        include_deprecated: false,
        llm_assist: true,
      });
      setCanonicalReviewCandidates(candidates);
      setCanonicalReviewStatus(`${candidates.length} canonical review candidate found`);
      setReviewExpanded(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setCanonicalReviewStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [memory]);*/

  const handleApplyCanonicalReview = useCallback(async (candidate: MemoryCanonicalReviewCandidate) => {
    try {
      setError(null);
      setIsLoading(true);
      const receipt = await memory.applyCanonicalReview({
        candidate,
        mark_sources_deprecated: true,
        actor: "user",
        reason: "user_approved_canonical_memory_review_from_graph",
        metadata: { source: "memory_graph_canonical_review", confidence: candidate.confidence, reasons: candidate.reasons },
      });
      setCanonicalReviewStatus(receipt.reason);
      await refresh();
      const nextCandidates = await memory.listCanonicalReviewCandidates({ limit: 40, min_score: 0.62, include_deprecated: false, llm_assist: true });
      setCanonicalReviewCandidates(nextCandidates);
      setSelectedNodeId(receipt.target_node.id);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setCanonicalReviewStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh]);

  /*const handleFindDuplicates = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const candidates = await memory.listDuplicateCandidates({ limit: 80, min_score: 0.7, include_deprecated: false });
      setDuplicateCandidates(candidates);
      setMergeStatus(`${candidates.length} duplicate candidate found`);
      setReviewExpanded(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMergeStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [memory]);*/

  const handleMergeCandidate = useCallback(async (candidate: MemoryDuplicateCandidate) => {
    try {
      setError(null);
      setIsLoading(true);
      const receipt = await memory.mergeMemoryNodes({
        target_node_id: candidate.canonical_node.id,
        source_node_ids: [candidate.duplicate_node.id],
        mark_sources_deprecated: true,
        actor: "user",
        reason: "user_confirmed_duplicate_from_memory_graph",
        metadata: { source: "memory_graph_duplicate_resolution", score: candidate.score, reasons: candidate.reasons },
      });
      setMergeStatus(receipt.reason);
      await refresh();
      const candidates = await memory.listDuplicateCandidates({ limit: 80, min_score: 0.7, include_deprecated: false });
      setDuplicateCandidates(candidates);
      setSelectedNodeId(receipt.target_node.id);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setMergeStatus(message);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh]);

  const handleUpdateSkillStatus = useCallback(async (candidateId: string, status: MemorySkillCandidate["status"]) => {
    try {
      setError(null);
      await memory.updateSkillCandidate({
        candidate_id: candidateId,
        status,
        approved_by: status === "approved" ? "user" : null,
        reason: `user_set_${status}`,
        metadata: { source: "memory_graph_panel" },
      });
      const nextSkills = await memory.listSkillCandidates(false, 120);
      setSkillCandidates(nextSkills);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [memory]);

  /*const handleRebuildEmbeddings = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const receipt = await memory.rebuildEmbeddingIndex({ limit: 2000, force: false });
      setEmbeddingReceipt(receipt);
      const nextEmbeddingStatus = await memory.getEmbeddingStatus();
      setEmbeddingStatus(nextEmbeddingStatus);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh]);

  const handleRunEmbeddingMaintenance = useCallback(async () => {
    try {
      setError(null);
      setIsLoading(true);
      const receipt = await memory.runEmbeddingMaintenance({ limit: 64, force: false, reason: "memory_graph_user_action" });
      setEmbeddingMaintenanceStatus(receipt.ran ? `Indexed ${receipt.indexed_chunks}; pending ${receipt.pending_after}` : receipt.reason);
      const nextEmbeddingStatus = await memory.getEmbeddingStatus();
      setEmbeddingStatus(nextEmbeddingStatus);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }, [memory, refresh]);*/

  const handleUpdateNodeGovernance = useCallback(async (
    status: MemoryVerificationStatus,
    reason: string,
    salience?: number
  ) => {
    if (!selectedNodeId) return;
    try {
      setError(null);
      setIsUpdatingGovernance(true);
      const receipt = await memory.updateNodeGovernance({
        node_id: selectedNodeId,
        verification_status: status,
        salience: typeof salience === "number" ? salience : null,
        reason,
        actor: "user",
        metadata: { source: "memory_graph_node_inspector" },
      });
      setGovernanceStatus(receipt.accepted ? receipt.reason : `Update rejected: ${receipt.reason}`);
      await refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setGovernanceStatus(message);
      setError(message);
    } finally {
      setIsUpdatingGovernance(false);
    }
  }, [memory, refresh, selectedNodeId]);

  const handleUpdateNodeSalience = useCallback(async (salience: number, reason: string) => {
    if (!selectedNodeId) return;
    try {
      setError(null);
      setIsUpdatingGovernance(true);
      const receipt = await memory.updateNodeGovernance({
        node_id: selectedNodeId,
        salience,
        reason,
        actor: "user",
        metadata: { source: "memory_graph_node_inspector", action: "salience_adjustment" },
      });
      setGovernanceStatus(receipt.accepted ? receipt.reason : `Update rejected: ${receipt.reason}`);
      await refresh();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setGovernanceStatus(message);
      setError(message);
    } finally {
      setIsUpdatingGovernance(false);
    }
  }, [memory, refresh, selectedNodeId]);


  return (
    <div className={`memory-graph-panel ${immersive ? "memory-graph-panel--immersive" : ""}`}>
      {error ? <div className="desktop-agent-error memory-graph-error-overlay">{error}</div> : null}

      <MemoryGraphToolbar
        isBusy={isLoading}
        labelsVisible={labelsVisible}
        qualityExpanded={qualityExpanded}
        queryExpanded={queryExpanded}
        controlsExpanded={controlsExpanded}
        reviewExpanded={reviewExpanded}
        reviewCount={duplicateCandidates.length + canonicalReviewCandidates.length}
        graphMode={viewSettings.mode}
        autopilotStatus={memoryAutopilotStatus}
        onRunAutopilot={() => void handleRunRecommendedMaintenance()}
        onToggleGraphMode={() => setViewSettings((current) => ({ ...current, mode: current.mode === "local" ? "global" : "local" }))}
        onToggleControls={() => setControlsExpanded((value) => !value)}
        onToggleQuality={() => setQualityExpanded((value) => !value)}
        onToggleReview={() => void handleToggleReviewQueue()}
        onRefresh={() => void refresh()}
        onToggleLabels={handleToggleLabels}
        onToggleSearch={() => setQueryExpanded((value) => !value)}
      />

      <MemoryGraphStatusBar
        status={status}
        snapshot={snapshot}
        embeddingStatus={embeddingStatus}
        qualityDashboard={qualityDashboard}
      />

      {controlsExpanded ? (
        <MemoryGraphControlsOverlay
          settings={layoutSettings}
          viewSettings={viewSettings}
          availableKinds={availableNodeKinds}
          nodeKindLabels={NODE_KIND_LABELS}
          onChange={handleLayoutSettingsChange}
          onViewChange={handleViewSettingsChange}
          onClose={() => setControlsExpanded(false)}
        />
      ) : null}

      {qualityExpanded && qualityDashboard ? <MemoryQualityOverlay dashboard={qualityDashboard} /> : null}


      {reviewExpanded ? (
        <aside className="memory-merge-overlay memory-autopilot-review-overlay" aria-label="Governed Brain Review queue">
          <div className="memory-merge-overlay-header">
            <div>
              <strong>Brain Review Queue</strong>
              <span>Autopilot/LLM proposes duplicate and canonical-memory candidates. You govern the final merge.</span>
            </div>
            <button type="button" onClick={() => setReviewExpanded(false)}>Close</button>
          </div>
          <div className="memory-merge-overlay-actions">
            <button type="button" onClick={() => void loadMemoryReviewQueue(false)} disabled={isLoading}>Refresh proposals</button>
            <button type="button" onClick={() => void handleRunRecommendedMaintenance()} disabled={isLoading}>Run recommended maintenance</button>
            <button type="button" onClick={() => void handleRunMemoryAutopilot()} disabled={isLoading} title="Advanced direct autopilot fallback; recommended maintenance remains the default governed flow">Advanced autopilot</button>
            <span>{canonicalReviewStatus || mergeStatus || "No review run yet"}</span>
          </div>
          <div className="memory-review-section">
            <div className="memory-review-section-title">
              <strong>Canonical memory proposals</strong>
              <span>{canonicalReviewCandidates.length}</span>
            </div>
            <div className="memory-merge-candidate-list">
              {canonicalReviewCandidates.length === 0 ? (
                <p className="desktop-agent-muted">No canonical memory proposals. Run Autopilot to let the LLM/RAG review the graph.</p>
              ) : canonicalReviewCandidates.slice(0, 18).map((candidate) => (
                <article key={candidate.id} className="memory-merge-candidate memory-canonical-review-candidate">
                  <div className="memory-merge-candidate-score">{Math.round(candidate.confidence * 100)}%</div>
                  <div className="memory-merge-candidate-body">
                    <strong>{candidate.proposed_title || candidate.target_node.title}</strong>
                    <span>{candidate.candidate_nodes.length} node{candidate.candidate_nodes.length === 1 ? "" : "s"} → {candidate.target_node.title}</span>
                    <small>{candidate.rationale}</small>
                    <small>{candidate.reasons.slice(0, 4).join(", ")}{candidate.shared_tags?.length ? ` · tags: ${candidate.shared_tags.slice(0, 5).join(", ")}` : ""}</small>
                  </div>
                  <button type="button" onClick={() => void handleApplyCanonicalReview(candidate)} disabled={isLoading}>Approve</button>
                </article>
              ))}
            </div>
          </div>
          <div className="memory-review-section">
            <div className="memory-review-section-title">
              <strong>Duplicate proposals</strong>
              <span>{duplicateCandidates.length}</span>
            </div>
            <div className="memory-merge-candidate-list">
              {duplicateCandidates.length === 0 ? (
                <p className="desktop-agent-muted">No duplicate proposals. Autopilot keeps discovery advisory-only until you approve a merge.</p>
              ) : duplicateCandidates.slice(0, 18).map((candidate) => (
                <article key={`${candidate.canonical_node.id}:${candidate.duplicate_node.id}`} className="memory-merge-candidate">
                  <div className="memory-merge-candidate-score">{Math.round(candidate.score * 100)}%</div>
                  <div className="memory-merge-candidate-body">
                    <strong>{candidate.duplicate_node.title}</strong>
                    <span>→ {candidate.canonical_node.title}</span>
                    <small>{candidate.reasons.join(", ")}{candidate.shared_tags?.length ? ` · tags: ${candidate.shared_tags.slice(0, 4).join(", ")}` : ""}</small>
                  </div>
                  <button type="button" onClick={() => void handleMergeCandidate(candidate)} disabled={isLoading}>Approve</button>
                </article>
              ))}
            </div>
          </div>
        </aside>
      ) : null}

      {queryExpanded ? (
        <MemoryGraphSearchOverlay
          value={queryText}
          onChange={setQueryText}
          onSearch={() => void handleSearch()}
          onClose={() => setQueryExpanded(false)}
        />
      ) : null}

      <section className="memory-graph-content memory-graph-content--brain-first memory-graph-content--full-surface">
        <div className="memory-graph-canvas-card memory-graph-canvas-card--primary memory-graph-canvas-card--full-surface">
          <MemoryGraphCanvas
            edges={visibleEdges}
            nodes={visibleNodes}
            nodeById={nodeById}
            onSelectNode={setSelectedNodeId}
            selectedNodeId={selectedNodeId}
            activePayload={activePayload}
            labelsVisible={labelsVisible}
            layoutSettings={layoutSettings}
            graphMode={viewSettings.mode}
            localDepth={viewSettings.localDepth}
          />
        </div>
      </section>

      {selectedNode ? (
        <aside className="memory-graph-node-drawer" aria-label="Memory node details">
          <button type="button" className="memory-graph-drawer-close" onClick={() => setSelectedNodeId(null)} title="Chiudi dettagli nodo">
            <IoClose />
          </button>
          <MemoryNodeInspector
            node={selectedNode}
            isUpdatingGovernance={isUpdatingGovernance}
            governanceMessage={governanceStatus}
            onUpdateGovernance={(status, reason) => void handleUpdateNodeGovernance(status, reason)}
            onUpdateSalience={(salience, reason) => void handleUpdateNodeSalience(salience, reason)}
          />
          {queryHits.length ? <MemorySearchResults hits={queryHits} onSelectNode={setSelectedNodeId} /> : null}
        </aside>
      ) : null}

      {queryExpanded && queryHits.length ? (
        <aside className="memory-graph-search-results-overlay">
          <MemorySearchResults hits={queryHits} onSelectNode={(nodeId) => { setSelectedNodeId(nodeId); setQueryExpanded(false); }} />
        </aside>
      ) : null}

      <section className="memory-graph-utility-dock" aria-label="Memory utility status">
        <MemoryRagControlCenter
          closeout={memoryRagCloseout}
          integrity={memoryRagIntegrity}
          queue={memoryJobQueue}
          maintenanceReceipt={recommendedMaintenanceReceipt}
          status={memoryControlCenterStatus}
          isBusy={isLoading}
          onRefresh={() => void refreshMemoryControlCenter()}
          onDryRun={() => void handlePlanRecommendedMaintenance()}
          onRunRecommended={() => void handleRunRecommendedMaintenance()}
          onRunKnowledgeRefresh={() => void handleRunKnowledgeRefresh()}
        />
        <MemoryEmbeddingSummary status={embeddingStatus} receipt={embeddingReceipt} maintenanceStatus={embeddingMaintenanceStatus} />
        <MemoryActivationSummary payload={activePayload} />
        <MemorySkillCandidatePanel
          candidates={skillCandidates}
          status={skillStatus}
          onExtract={() => void handleExtractSkills()}
          onUpdateStatus={(candidateId, status) => void handleUpdateSkillStatus(candidateId, status)}
        />
        <MemoryActivationTimeline activations={activations} />
      </section>
    </div>
  );
}

const MEMORY_GRAPH_VIEW_PREF_KEY = "astra.memoryGraph.viewSettings.v1";

function readMemoryGraphViewSettings(): MemoryGraphViewSettings {
  if (typeof window === "undefined") return DEFAULT_MEMORY_GRAPH_VIEW_SETTINGS;
  try {
    const raw = window.localStorage.getItem(MEMORY_GRAPH_VIEW_PREF_KEY);
    if (!raw) return DEFAULT_MEMORY_GRAPH_VIEW_SETTINGS;
    return sanitizeMemoryGraphViewSettings(JSON.parse(raw));
  } catch {
    return DEFAULT_MEMORY_GRAPH_VIEW_SETTINGS;
  }
}

function writeMemoryGraphViewSettings(settings: MemoryGraphViewSettings) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(MEMORY_GRAPH_VIEW_PREF_KEY, JSON.stringify(sanitizeMemoryGraphViewSettings(settings)));
  } catch {
    // Non-blocking UI preference persistence.
  }
}

function sanitizeMemoryGraphViewSettings(value: Partial<MemoryGraphViewSettings> | null | undefined): MemoryGraphViewSettings {
  const mode = value?.mode === "local" ? "local" : "global";
  const visibleKinds = typeof value?.visibleKinds === "object" && value.visibleKinds !== null
    ? Object.fromEntries(Object.entries(value.visibleKinds).filter(([, flag]) => typeof flag === "boolean")) as Record<string, boolean>
    : {};
  return {
    mode,
    localDepth: Math.round(clampNumber(Number(value?.localDepth ?? DEFAULT_MEMORY_GRAPH_VIEW_SETTINGS.localDepth), 1, 4)),
    showIsolatedNodes: typeof value?.showIsolatedNodes === "boolean" ? value.showIsolatedNodes : DEFAULT_MEMORY_GRAPH_VIEW_SETTINGS.showIsolatedNodes,
    visibleKinds,
  };
}

const MEMORY_GRAPH_LABELS_PREF_KEY = "astra.memoryGraph.labelsVisible.v2";

function readMemoryGraphLabelsPreference(): boolean {
  if (typeof window === "undefined") return true;
  try {
    const value = window.localStorage.getItem(MEMORY_GRAPH_LABELS_PREF_KEY);
    if (value === "hidden") return false;
    if (value === "visible") return true;
    // v0.5.27 intentionally ignores the old boolean key once, because the previous
    // toggle could persist a hidden state that appeared unrecoverable after reload.
    return true;
  } catch {
    return true;
  }
}

function writeMemoryGraphLabelsPreference(value: boolean) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(MEMORY_GRAPH_LABELS_PREF_KEY, value ? "visible" : "hidden");
  } catch {
    // Non-blocking UI preference persistence.
  }
}

const MEMORY_GRAPH_LAYOUT_PREF_KEY = "astra.memoryGraph.layoutSettings.v1";

function readMemoryGraphLayoutSettings(): MemoryGraphLayoutSettings {
  if (typeof window === "undefined") return DEFAULT_MEMORY_GRAPH_LAYOUT_SETTINGS;
  try {
    const raw = window.localStorage.getItem(MEMORY_GRAPH_LAYOUT_PREF_KEY);
    if (!raw) return DEFAULT_MEMORY_GRAPH_LAYOUT_SETTINGS;
    return sanitizeMemoryGraphLayoutSettings(JSON.parse(raw));
  } catch {
    return DEFAULT_MEMORY_GRAPH_LAYOUT_SETTINGS;
  }
}

function writeMemoryGraphLayoutSettings(settings: MemoryGraphLayoutSettings) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(MEMORY_GRAPH_LAYOUT_PREF_KEY, JSON.stringify(sanitizeMemoryGraphLayoutSettings(settings)));
  } catch {
    // Non-blocking UI preference persistence.
  }
}

function sanitizeMemoryGraphLayoutSettings(value: Partial<MemoryGraphLayoutSettings> | null | undefined): MemoryGraphLayoutSettings {
  const preset = typeof value?.preset === "string" && value.preset in MEMORY_GRAPH_LAYOUT_PRESETS
    ? value.preset as keyof typeof MEMORY_GRAPH_LAYOUT_PRESETS
    : "vault";
  const base = MEMORY_GRAPH_LAYOUT_PRESETS[preset];
  const labelMode = isMemoryGraphLabelMode(value?.labelMode) ? value!.labelMode : base.labelMode;
  return {
    preset,
    labelMode,
    labelSize: clampNumber(Number(value?.labelSize ?? base.labelSize), 0.72, 1.65),
    repulsion: clampNumber(Number(value?.repulsion ?? base.repulsion), 0.45, 1.75),
    linkDistance: clampNumber(Number(value?.linkDistance ?? base.linkDistance), 0.55, 1.65),
    centerForce: clampNumber(Number(value?.centerForce ?? base.centerForce), 0.35, 1.9),
    clusterForce: clampNumber(Number(value?.clusterForce ?? base.clusterForce), 0.35, 1.9),
  };
}

function isMemoryGraphLabelMode(value: unknown): value is MemoryGraphLabelMode {
  return value === "hidden" || value === "selected" || value === "active" || value === "important" || value === "all";
}
