import { useCallback, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import type { MemoryActivationEventPayload, MemoryEdge, MemoryNode } from "../../../types/memory";
import type { MemoryGraphLayoutSettings } from "./MemoryGraphControlsOverlay";
import {
  GRAPH_CENTER_X,
  GRAPH_CENTER_Y,
  GRAPH_HEIGHT,
  GRAPH_WIDTH,
  LARGE_GRAPH_NODE_THRESHOLD,
  NODE_KIND_LABELS,
  type GraphPerformanceStats,
  type GraphViewport,
} from "../layout/memoryGraphLayoutTypes";
import {
  clampNumber,
  graphBounds,
  memoryNodeVisualWeight,
  nodeRadius,
} from "../layout/memoryGraphLayoutEngine";
import { useMemoryGraphLayoutRuntime, graphNodeFallbackPosition } from "../hooks/useMemoryGraphLayoutRuntime";
import { useMemoryGraphSurfaceSize } from "../hooks/useMemoryGraphSurfaceSize";
import { useMemoryGraphSvgLabels } from "../hooks/useMemoryGraphSvgLabels";
import { MemoryGraphEdgeLayer } from "./MemoryGraphEdgeLayer";
import { MemoryGraphNodeLayer } from "./MemoryGraphNodeLayer";
import { MemoryGraphLabelLayer } from "./MemoryGraphLabelLayer";

export function MemoryGraphCanvas({
  activePayload,
  edges,
  nodes,
  nodeById,
  onSelectNode,
  selectedNodeId,
  labelsVisible,
  layoutSettings,
  graphMode,
  localDepth,
}: {
  activePayload: MemoryActivationEventPayload | null;
  edges: MemoryEdge[];
  nodes: MemoryNode[];
  nodeById: Map<string, MemoryNode>;
  onSelectNode: (nodeId: string) => void;
  selectedNodeId: string | null;
  labelsVisible: boolean;
  layoutSettings: MemoryGraphLayoutSettings;
  graphMode: "global" | "local";
  localDepth: number;
}) {
  const activeEdges = useMemo(() => new Set(activePayload?.activated_edge_ids ?? []), [activePayload?.activated_edge_ids]);
  const [viewport, setViewport] = useState<GraphViewport>({ scale: 1, tx: 0, ty: 0 });
  const dragRef = useRef<{ nodeId: string; pointerId: number; offsetX: number; offsetY: number } | null>(null);
  const panRef = useRef<{ pointerId: number; x: number; y: number; start: GraphViewport } | null>(null);
  const pinnedNodeIdsRef = useRef<Set<string>>(new Set());
  const svgRef = useRef<SVGSVGElement | null>(null);
  const { shellRef, surfaceSize } = useMemoryGraphSurfaceSize();
  const [simulationPaused, setSimulationPaused] = useState(false);

  const {
    clusters,
    layout,
    layoutById,
    lastTickMs,
    markRunning,
    positions,
    resetPositions,
    setPositions,
    simulationState,
  } = useMemoryGraphLayoutRuntime({
    activePayload,
    dragRef,
    edges,
    layoutSettings,
    nodes,
    pinnedNodeIdsRef,
    simulationPaused,
  });

  const svgLabels = useMemoryGraphSvgLabels({
    labelsVisible,
    layout,
    layoutSettings,
    nodeById,
    nodeCount: nodes.length,
    selectedNodeId,
    surfaceSize,
    viewport,
  });

  const transform = `translate(${viewport.tx} ${viewport.ty}) scale(${viewport.scale})`;
  const pointerToGraph = useCallback((event: PointerEvent<SVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return { x: GRAPH_CENTER_X, y: GRAPH_CENTER_Y };
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const matrix = svg.getScreenCTM();
    if (!matrix) return { x: GRAPH_CENTER_X, y: GRAPH_CENTER_Y };
    const svgPoint = point.matrixTransform(matrix.inverse());
    return {
      x: (svgPoint.x - viewport.tx) / viewport.scale,
      y: (svgPoint.y - viewport.ty) / viewport.scale,
    };
  }, [viewport.scale, viewport.tx, viewport.ty]);

  const handleNodePointerDown = useCallback((event: PointerEvent<SVGGElement>, nodeId: string) => {
    event.preventDefault();
    event.stopPropagation();
    svgRef.current?.setPointerCapture(event.pointerId);
    const pointer = pointerToGraph(event);
    const currentPosition = positions[nodeId] ?? graphNodeFallbackPosition(nodeId, nodes.length);
    pinnedNodeIdsRef.current.add(nodeId);
    dragRef.current = {
      nodeId,
      pointerId: event.pointerId,
      offsetX: currentPosition.x - pointer.x,
      offsetY: currentPosition.y - pointer.y,
    };
    onSelectNode(nodeId);
    setPositions((previous) => ({
      ...previous,
      [nodeId]: { ...currentPosition, vx: 0, vy: 0 },
    }));
  }, [nodes.length, onSelectNode, pointerToGraph, positions]);

  const handleCanvasPointerDown = useCallback((event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    panRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, start: viewport };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [viewport]);

  const handlePointerMove = useCallback((event: PointerEvent<SVGSVGElement>) => {
    const drag = dragRef.current;
    if (drag) {
      const point = pointerToGraph(event);
      setPositions((previous) => ({
        ...previous,
        [drag.nodeId]: { ...(previous[drag.nodeId] ?? graphNodeFallbackPosition(drag.nodeId, nodes.length)), x: point.x + drag.offsetX, y: point.y + drag.offsetY, vx: 0, vy: 0 },
      }));
      return;
    }

    const pan = panRef.current;
    if (!pan || pan.pointerId !== event.pointerId) return;
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const dx = ((event.clientX - pan.x) / Math.max(1, rect.width)) * GRAPH_WIDTH;
    const dy = ((event.clientY - pan.y) / Math.max(1, rect.height)) * GRAPH_HEIGHT;
    setViewport({ ...pan.start, tx: pan.start.tx + dx, ty: pan.start.ty + dy });
  }, [nodes.length, pointerToGraph]);

  const handlePointerUp = useCallback((event: PointerEvent<SVGSVGElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
    if (panRef.current?.pointerId === event.pointerId) panRef.current = null;
  }, []);

  const handleWheel = useCallback((event: WheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const sx = ((event.clientX - rect.left) / Math.max(1, rect.width)) * GRAPH_WIDTH;
    const sy = ((event.clientY - rect.top) / Math.max(1, rect.height)) * GRAPH_HEIGHT;
    const nextScale = clampNumber(viewport.scale * (event.deltaY < 0 ? 1.1 : 0.9), 0.42, 2.8);
    const graphX = (sx - viewport.tx) / viewport.scale;
    const graphY = (sy - viewport.ty) / viewport.scale;
    setViewport({
      scale: nextScale,
      tx: sx - graphX * nextScale,
      ty: sy - graphY * nextScale,
    });
  }, [viewport]);

  const resetLayout = useCallback(() => {
    setSimulationPaused(false);
    markRunning();
    pinnedNodeIdsRef.current.clear();
    const nextPositions = resetPositions();
    const nextLayout = nodes.map((node) => {
      const p = nextPositions[node.id] ?? graphNodeFallbackPosition(node.id, nodes.length);
      return { id: node.id, x: p.x, y: p.y, radius: nodeRadius(node, 0), activated: false, intensity: 0 };
    });
    const bounds = graphBounds(nextLayout);
    const scale = clampNumber(Math.min(
      GRAPH_WIDTH / Math.max(1, bounds.width + 360),
      GRAPH_HEIGHT / Math.max(1, bounds.height + 280)
    ), 0.62, 1.35);
    setViewport({
      scale,
      tx: GRAPH_CENTER_X - (bounds.x + bounds.width / 2) * scale,
      ty: GRAPH_CENTER_Y - (bounds.y + bounds.height / 2) * scale,
    });
  }, [markRunning, nodes, resetPositions]);

  const fitGraph = useCallback(() => {
    if (layout.length === 0) return;
    const bounds = graphBounds(layout);
    const padding = 120;
    const scale = clampNumber(Math.min(
      GRAPH_WIDTH / Math.max(1, bounds.width + padding * 2),
      GRAPH_HEIGHT / Math.max(1, bounds.height + padding * 2)
    ), 0.42, 1.8);
    setViewport({
      scale,
      tx: GRAPH_CENTER_X - (bounds.x + bounds.width / 2) * scale,
      ty: GRAPH_CENTER_Y - (bounds.y + bounds.height / 2) * scale,
    });
  }, [layout]);

  const performanceStats: GraphPerformanceStats = {
    visibleNodes: nodes.length,
    visibleEdges: edges.length,
    renderedLabels: svgLabels.length,
    culledLabels: Math.max(0, layout.length - svgLabels.length),
    simulationState,
    lastTickMs,
  };

  if (nodes.length === 0) {
    return <div className="desktop-agent-empty">Nessun nodo memoria ancora disponibile. Genera un recap, una conversazione o una Work Session per alimentare il grafo.</div>;
  }

  return (
    <div className={`memory-graph-canvas-shell ${labelsVisible ? "memory-graph-canvas-shell--labels-visible" : "memory-graph-canvas-shell--labels-hidden"}`} ref={shellRef}>
      <div className="memory-graph-canvas-controls">
        <span>{performanceStats.visibleNodes} nodes · {performanceStats.visibleEdges} edges · {graphMode === "local" ? `local depth ${localDepth}` : "global vault"} · {performanceStats.simulationState}</span>
        <button type="button" onClick={() => setSimulationPaused((value) => !value)} title={simulationPaused ? "Resume graph layout" : "Freeze graph layout"}>{simulationPaused ? "▶" : "Ⅱ"}</button>
        <button type="button" onClick={() => setViewport((current) => ({ ...current, scale: clampNumber(current.scale * 1.14, 0.42, 2.8) }))}>+</button>
        <button type="button" onClick={() => setViewport((current) => ({ ...current, scale: clampNumber(current.scale * 0.88, 0.42, 2.8) }))}>−</button>
        <button type="button" onClick={fitGraph}>Fit</button>
        <button type="button" onClick={resetLayout}>Reset</button>
      </div>
      <MemoryGraphPerformanceBadge stats={performanceStats} />
      <svg
        ref={svgRef}
        className="memory-graph-canvas memory-graph-canvas--brain"
        viewBox={`0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`}
        role="img"
        aria-label="Astra Memory Graph"
        onPointerDown={handleCanvasPointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={handleWheel}
      >
        <defs>
          <filter id="memoryGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="memoryNodeDepth" cx="35%" cy="30%" r="72%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.96)" />
            <stop offset="100%" stopColor="rgba(125,150,255,0.78)" />
          </radialGradient>
        </defs>
        <rect x="0" y="0" width={GRAPH_WIDTH} height={GRAPH_HEIGHT} className="memory-graph-background" />
        <g transform={transform}>
          <MemoryGraphEdgeLayer
            activeEdges={activeEdges}
            edges={edges}
            layoutById={layoutById}
            viewportScale={viewport.scale}
          />
          <MemoryGraphNodeLayer
            layout={layout}
            nodeById={nodeById}
            onNodePointerDown={handleNodePointerDown}
            selectedNodeId={selectedNodeId}
          />
          <MemoryGraphLabelLayer labels={svgLabels} />
        </g>
      </svg>
    </div>
  );
}


export function MemoryGraphPerformanceBadge({ stats }: { stats: GraphPerformanceStats }) {
  const degraded = stats.lastTickMs > 18 || stats.visibleNodes > LARGE_GRAPH_NODE_THRESHOLD;
  return (
    <div className={`memory-graph-performance-badge ${degraded ? "memory-graph-performance-badge--degraded" : ""}`} aria-label="Memory graph performance metrics">
      <span>{stats.simulationState}</span>
      <span>{stats.visibleNodes}N</span>
      <span>{stats.visibleEdges}E</span>
      <span>{stats.renderedLabels}L</span>
      <span>{stats.lastTickMs.toFixed(1)}ms</span>
    </div>
  );
}