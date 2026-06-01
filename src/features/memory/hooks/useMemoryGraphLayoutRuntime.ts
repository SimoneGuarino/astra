import { useCallback, useEffect, useMemo, useRef, useState, type MutableRefObject } from "react";
import type { MemoryActivationEventPayload, MemoryEdge, MemoryNode } from "../../../types/memory";
import type { MemoryGraphLayoutSettings } from "../components/MemoryGraphControlsOverlay";
import {
  FORCE_STABLE_TICKS,
  FORCE_STABLE_VELOCITY,
  LARGE_GRAPH_NODE_THRESHOLD,
  type GraphPerformanceStats,
  type GraphPoint,
  type GraphPosition,
} from "../layout/memoryGraphLayoutTypes";
import {
  buildGraphClusters,
  initializeGraphPositions,
  maxGraphVelocity,
  nodeRadius,
  seededPosition,
  stepBrainForceLayout,
} from "../layout/memoryGraphLayoutEngine";
import { GRAPH_CENTER_X, GRAPH_CENTER_Y } from "../layout/memoryGraphLayoutTypes";

type DragState = { nodeId: string; pointerId: number; offsetX: number; offsetY: number } | null;

type UseMemoryGraphLayoutRuntimeArgs = {
  nodes: MemoryNode[];
  edges: MemoryEdge[];
  activePayload: MemoryActivationEventPayload | null;
  layoutSettings: MemoryGraphLayoutSettings;
  dragRef: MutableRefObject<DragState>;
  pinnedNodeIdsRef: MutableRefObject<Set<string>>;
  simulationPaused: boolean;
};

export function useMemoryGraphLayoutRuntime({
  activePayload,
  dragRef,
  edges,
  layoutSettings,
  nodes,
  pinnedNodeIdsRef,
  simulationPaused,
}: UseMemoryGraphLayoutRuntimeArgs) {
  const activeIds = useMemo(() => new Set(activePayload?.activated_node_ids ?? []), [activePayload?.activated_node_ids]);
  const intensities = activePayload?.intensity ?? {};
  const [positions, setPositions] = useState<Record<string, GraphPosition>>({});
  const clusters = useMemo(() => buildGraphClusters(nodes, edges), [edges, nodes]);
  const lastTickAtRef = useRef(0);
  const stableTickCountRef = useRef(0);
  const [simulationState, setSimulationState] = useState<GraphPerformanceStats["simulationState"]>("running");
  const [lastTickMs, setLastTickMs] = useState(0);

  useEffect(() => {
    stableTickCountRef.current = 0;
    setSimulationState(simulationPaused ? "paused" : "running");
    setPositions((previous) => initializeGraphPositions(nodes, previous, clusters));
  }, [clusters, nodes, simulationPaused]);

  useEffect(() => {
    let frame = 0;
    let stopped = false;
    const targetFrameMs = nodes.length > LARGE_GRAPH_NODE_THRESHOLD ? 32 : 16;

    const tick = (now: number) => {
      if (stopped) return;
      if (simulationPaused) {
        setSimulationState("paused");
        frame = window.requestAnimationFrame(tick);
        return;
      }
      if (lastTickAtRef.current && now - lastTickAtRef.current < targetFrameMs) {
        frame = window.requestAnimationFrame(tick);
        return;
      }
      lastTickAtRef.current = now;
      const startedAt = performance.now();
      setPositions((previous) => {
        const result = stepBrainForceLayout(
          previous,
          nodes,
          edges,
          clusters,
          dragRef.current?.nodeId ?? null,
          pinnedNodeIdsRef.current,
          layoutSettings,
        );
        const maxVelocity = maxGraphVelocity(result);
        if (!dragRef.current && maxVelocity < FORCE_STABLE_VELOCITY) {
          stableTickCountRef.current += 1;
        } else {
          stableTickCountRef.current = 0;
        }
        setSimulationState(stableTickCountRef.current >= FORCE_STABLE_TICKS ? "settled" : "running");
        return stableTickCountRef.current >= FORCE_STABLE_TICKS && !dragRef.current ? previous : result;
      });
      setLastTickMs(performance.now() - startedAt);
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => {
      stopped = true;
      window.cancelAnimationFrame(frame);
    };
  }, [clusters, dragRef, edges, layoutSettings, nodes, pinnedNodeIdsRef, simulationPaused]);

  const layout = useMemo(() => {
    return nodes.map((node) => {
      const position = positions[node.id] ?? seededPosition(node.id, clusters[0]?.x ?? GRAPH_CENTER_X, clusters[0]?.y ?? GRAPH_CENTER_Y, nodes.length || 1);
      const activated = activeIds.has(node.id);
      const rawIntensity = intensities[node.id];
      const intensity = typeof rawIntensity === "number" ? rawIntensity : activated ? 0.85 : 0;
      return {
        id: node.id,
        x: position.x,
        y: position.y,
        radius: nodeRadius(node, intensity),
        activated,
        intensity,
      };
    });
  }, [activeIds, clusters, intensities, nodes, positions]);

  const layoutById = useMemo(() => {
    const map = new Map<string, GraphPoint>();
    for (const point of layout) map.set(point.id, point);
    return map;
  }, [layout]);

  const resetPositions = useCallback(() => {
    stableTickCountRef.current = 0;
    const nextPositions = initializeGraphPositions(nodes, {}, clusters);
    setPositions(nextPositions);
    return nextPositions;
  }, [clusters, nodes]);

  const markRunning = useCallback(() => {
    stableTickCountRef.current = 0;
    setSimulationState("running");
  }, []);

  return {
    activeIds,
    clusters,
    layout,
    layoutById,
    lastTickMs,
    markRunning,
    positions,
    resetPositions,
    setPositions,
    simulationState,
  };
}

export function graphNodeFallbackPosition(nodeId: string, nodeCount: number): GraphPosition {
  return seededPosition(nodeId, GRAPH_CENTER_X, GRAPH_CENTER_Y, nodeCount || 1);
}

