import type { MemoryEdge, MemoryNode, MemoryActivation } from "../../../types/memory";
import type { MemoryActivationEventPayload } from "../../../types/memory";
import type { MemoryGraphLabelMode, MemoryGraphLayoutSettings } from "../components/MemoryGraphControlsOverlay";
import {
  FORCE_SPATIAL_CELL_SIZE,
  GRAPH_CENTER_X,
  GRAPH_CENTER_Y,
  GRAPH_HEIGHT,
  GRAPH_WIDTH,
  MAX_RENDERED_LABELS,
  NODE_KIND_PRIORITY,
  type GraphCluster,
  type GraphPerformanceStats,
  type GraphPoint,
  type GraphPosition,
  type GraphSurfaceProjection,
  type GraphSurfaceSize,
  type GraphViewport,
} from "./memoryGraphLayoutTypes";

export function shouldShowMemoryNodeLabel(
  mode: MemoryGraphLabelMode,
  node: { selected: boolean; activated: boolean; important: boolean; nodeCount: number; viewportScale: number },
): boolean {
  if (mode === "hidden") return false;
  if (mode === "selected") return node.selected;
  if (mode === "active") return node.selected || node.activated;
  if (mode === "important") return node.selected || node.activated || node.important || node.viewportScale >= 1.2 || node.nodeCount <= 50;
  return true;
}

export function initializeGraphPositions(
  nodes: MemoryNode[],
  previous: Record<string, GraphPosition>,
  clusters: GraphCluster[],
): Record<string, GraphPosition> {
  const next: Record<string, GraphPosition> = {};
  const clusterByNode = new Map<string, GraphCluster>();
  const clusterIndexByNode = new Map<string, number>();
  for (const cluster of clusters) {
    cluster.nodeIds.forEach((id, index) => {
      clusterByNode.set(id, cluster);
      clusterIndexByNode.set(id, index);
    });
  }
  for (const node of nodes) {
    const cluster = clusterByNode.get(node.id) ?? clusters[0];
    const previousPosition = previous[node.id];
    if (previousPosition) {
      next[node.id] = { ...previousPosition, cluster: cluster?.id ?? previousPosition.cluster };
      continue;
    }
    next[node.id] = seededPosition(
      node.id,
      cluster?.x ?? GRAPH_CENTER_X,
      cluster?.y ?? GRAPH_CENTER_Y,
      (cluster?.nodeIds.length ?? nodes.length) || 1,
      clusterIndexByNode.get(node.id),
      cluster?.id ?? 0,
    );
  }
  return next;
}

export function stepBrainForceLayout(
  previous: Record<string, GraphPosition>,
  nodes: MemoryNode[],
  edges: MemoryEdge[],
  clusters: GraphCluster[],
  draggedNodeId: string | null,
  pinnedNodeIds: Set<string>,
  settings: MemoryGraphLayoutSettings,
): Record<string, GraphPosition> {
  const next: Record<string, GraphPosition> = {};
  const linked = new Map<string, { id: string; weight: number }[]>();
  for (const edge of edges) {
    if (!linked.has(edge.from_node_id)) linked.set(edge.from_node_id, []);
    if (!linked.has(edge.to_node_id)) linked.set(edge.to_node_id, []);
    linked.get(edge.from_node_id)!.push({ id: edge.to_node_id, weight: edge.weight });
    linked.get(edge.to_node_id)!.push({ id: edge.from_node_id, weight: edge.weight });
  }
  const clusterById = new Map(clusters.map((cluster) => [cluster.id, cluster]));
  const buckets = buildGraphSpatialBuckets(previous);
  const nodeCount = Math.max(1, nodes.length);

  for (const node of nodes) {
    const id = node.id;
    const p = previous[id];
    if (!p) continue;
    if (id === draggedNodeId || pinnedNodeIds.has(id)) {
      next[id] = { ...p, vx: 0, vy: 0 };
      continue;
    }

    const cluster = clusterById.get(p.cluster);
    const anchorX = cluster?.x ?? GRAPH_CENTER_X;
    const anchorY = cluster?.y ?? GRAPH_CENTER_Y;
    let fx = (GRAPH_CENTER_X - p.x) * 0.00065 * settings.centerForce;
    let fy = (GRAPH_CENTER_Y - p.y) * 0.00065 * settings.centerForce;
    fx += (anchorX - p.x) * 0.0022 * settings.clusterForce;
    fy += (anchorY - p.y) * 0.0022 * settings.clusterForce;

    for (const otherId of nearbyGraphNodeIds(p, buckets)) {
      if (otherId === id) continue;
      const other = previous[otherId];
      if (!other) continue;
      const dx = p.x - other.x;
      const dy = p.y - other.y;
      const distanceSquared = Math.max(64, dx * dx + dy * dy);
      const distance = Math.sqrt(distanceSquared);
      const sameCluster = p.cluster === other.cluster;
      const range = sameCluster ? 240 : 150;
      if (distance > range) continue;
      const strength = ((sameCluster ? 34 : 12) * settings.repulsion) / Math.max(1, Math.sqrt(nodeCount));
      const force = strength / distanceSquared;
      fx += dx * force;
      fy += dy * force;
    }

    for (const link of linked.get(id) ?? []) {
      const other = previous[link.id];
      if (!other) continue;
      const dx = other.x - p.x;
      const dy = other.y - p.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const target = (draggedNodeId === link.id ? 128 : 72 + Math.max(0, 1 - link.weight) * 52) * settings.linkDistance;
      const force = (distance - target) * (0.0065 + link.weight * 0.006);
      fx += (dx / distance) * force;
      fy += (dy / distance) * force;
    }

    const soft = softBoundaryForce(p.x, p.y);
    fx += soft.fx;
    fy += soft.fy;

    const vx = clampNumber((p.vx + fx) * 0.74, -7, 7);
    const vy = clampNumber((p.vy + fy) * 0.74, -7, 7);
    next[id] = {
      x: p.x + vx,
      y: p.y + vy,
      vx,
      vy,
      cluster: p.cluster,
    };
  }
  return next;
}

type GraphSpatialBuckets = Map<string, string[]>;

function buildGraphSpatialBuckets(positions: Record<string, GraphPosition>): GraphSpatialBuckets {
  const buckets: GraphSpatialBuckets = new Map();
  for (const [id, position] of Object.entries(positions)) {
    const key = graphSpatialBucketKey(position.x, position.y);
    buckets.set(key, [...(buckets.get(key) ?? []), id]);
  }
  return buckets;
}

function nearbyGraphNodeIds(position: GraphPosition, buckets: GraphSpatialBuckets): string[] {
  const cx = Math.floor(position.x / FORCE_SPATIAL_CELL_SIZE);
  const cy = Math.floor(position.y / FORCE_SPATIAL_CELL_SIZE);
  const ids: string[] = [];
  for (let dx = -1; dx <= 1; dx += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      ids.push(...(buckets.get(`${cx + dx}:${cy + dy}`) ?? []));
    }
  }
  return ids;
}

function graphSpatialBucketKey(x: number, y: number): string {
  return `${Math.floor(x / FORCE_SPATIAL_CELL_SIZE)}:${Math.floor(y / FORCE_SPATIAL_CELL_SIZE)}`;
}

export function maxGraphVelocity(positions: Record<string, GraphPosition>): number {
  let max = 0;
  for (const position of Object.values(positions)) {
    max = Math.max(max, Math.abs(position.vx) + Math.abs(position.vy));
  }
  return max;
}

export function adaptiveLabelLimit(mode: MemoryGraphLabelMode, nodeCount: number, viewportScale: number): number {
  if (mode === "hidden") return 0;
  if (mode === "selected") return 12;
  if (mode === "active") return 36;
  const scaleBoost = viewportScale >= 1.5 ? 1.4 : viewportScale >= 1.1 ? 1.15 : viewportScale < 0.72 ? 0.42 : 0.78;
  const countPenalty = nodeCount > 320 ? 0.45 : nodeCount > 180 ? 0.62 : nodeCount > 100 ? 0.78 : 1;
  const base = mode === "all" ? MAX_RENDERED_LABELS : 88;
  return Math.max(8, Math.round(base * scaleBoost * countPenalty));
}

export function computeLocalGraphNodeIds(rootIds: string[], edges: MemoryEdge[], depth: number): Set<string> {
  const result = new Set<string>();
  const adjacency = new Map<string, string[]>();
  for (const edge of edges) {
    if (!adjacency.has(edge.from_node_id)) adjacency.set(edge.from_node_id, []);
    if (!adjacency.has(edge.to_node_id)) adjacency.set(edge.to_node_id, []);
    adjacency.get(edge.from_node_id)!.push(edge.to_node_id);
    adjacency.get(edge.to_node_id)!.push(edge.from_node_id);
  }
  const queue = rootIds
    .filter(Boolean)
    .map((id) => ({ id, distance: 0 }));
  for (const root of queue) result.add(root.id);
  while (queue.length) {
    const current = queue.shift()!;
    if (current.distance >= depth) continue;
    for (const next of adjacency.get(current.id) ?? []) {
      if (result.has(next)) continue;
      result.add(next);
      queue.push({ id: next, distance: current.distance + 1 });
    }
  }
  return result;
}

export function buildGraphClusters(nodes: MemoryNode[], edges: MemoryEdge[]): GraphCluster[] {
  const nodeIds = new Set(nodes.map((node) => node.id));
  const adjacency = new Map<string, string[]>();
  for (const node of nodes) adjacency.set(node.id, []);
  for (const edge of edges) {
    if (!nodeIds.has(edge.from_node_id) || !nodeIds.has(edge.to_node_id)) continue;
    adjacency.get(edge.from_node_id)?.push(edge.to_node_id);
    adjacency.get(edge.to_node_id)?.push(edge.from_node_id);
  }

  const seen = new Set<string>();
  const components: string[][] = [];
  for (const node of nodes) {
    if (seen.has(node.id)) continue;
    const stack = [node.id];
    const component: string[] = [];
    seen.add(node.id);
    while (stack.length) {
      const current = stack.pop()!;
      component.push(current);
      for (const next of adjacency.get(current) ?? []) {
        if (seen.has(next)) continue;
        seen.add(next);
        stack.push(next);
      }
    }
    components.push(component);
  }

  components.sort((left, right) => right.length - left.length);
  return components.map((nodeIds, index) => {
    if (index === 0) return { id: index, nodeIds, x: GRAPH_CENTER_X, y: GRAPH_CENTER_Y };
    const angle = index * 2.399963229728653;
    const radius = Math.min(260, 105 + Math.sqrt(index) * 72);
    return {
      id: index,
      nodeIds,
      x: GRAPH_CENTER_X + Math.cos(angle) * radius,
      y: GRAPH_CENTER_Y + Math.sin(angle) * radius * 0.72,
    };
  });
}

export function seededPosition(
  id: string,
  clusterX: number,
  clusterY: number,
  count: number,
  index = hashString(id) % Math.max(1, count),
  cluster = 0,
): GraphPosition {
  const angle = ((index / Math.max(1, count)) * Math.PI * 2) + ((hashString(id) % 97) / 97) * 0.75;
  const radius = 24 + Math.sqrt(Math.max(1, count)) * 10 + ((hashString(`${id}:r`) % 100) / 100) * 54;
  return {
    x: clusterX + Math.cos(angle) * radius,
    y: clusterY + Math.sin(angle) * radius * 0.82,
    vx: 0,
    vy: 0,
    cluster,
  };
}

export function memoryNodeVisualWeight(node: MemoryNode): number {
  return (NODE_KIND_PRIORITY[node.kind] ?? 8) + node.salience * 6 + node.confidence * 4 + Math.min(4, node.tags.length * 0.45);
}

export function nodeRadius(node: MemoryNode, intensity: number): number {
  return 5.5 + Math.max(node.salience, node.confidence) * 7.5 + Math.max(0, memoryNodeVisualWeight(node) - 11) * 0.45 + intensity * 4.5;
}

export function softBoundaryForce(x: number, y: number): { fx: number; fy: number } {
  const margin = 160;
  let fx = 0;
  let fy = 0;
  if (x < margin) fx += (margin - x) * 0.00055;
  if (x > GRAPH_WIDTH - margin) fx -= (x - (GRAPH_WIDTH - margin)) * 0.00055;
  if (y < margin) fy += (margin - y) * 0.00055;
  if (y > GRAPH_HEIGHT - margin) fy -= (y - (GRAPH_HEIGHT - margin)) * 0.00055;
  return { fx, fy };
}

export function graphBounds(points: GraphPoint[]): { x: number; y: number; width: number; height: number } {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    minX = Math.min(minX, point.x - point.radius);
    minY = Math.min(minY, point.y - point.radius);
    maxX = Math.max(maxX, point.x + point.radius);
    maxY = Math.max(maxY, point.y + point.radius);
  }
  return { x: minX, y: minY, width: Math.max(1, maxX - minX), height: Math.max(1, maxY - minY) };
}

export function hashString(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0);
}

export function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function activationToPayload(activation: MemoryActivation): MemoryActivationEventPayload {
  return {
    request_id: activation.request_id,
    root_query: activation.root_query,
    activated_node_ids: activation.activated_node_ids,
    activated_edge_ids: activation.activated_edge_ids,
    intensity: activation.intensity,
    source: "memory_graph_snapshot",
  };
}

export function safeClassName(value: string): string {
  return value.replace(/[^a-z0-9_-]/gi, "_");
}

export function graphSurfaceProjection(surfaceSize: GraphSurfaceSize, viewport: GraphViewport): GraphSurfaceProjection | null {
  if (surfaceSize.width <= 0 || surfaceSize.height <= 0) return null;
  const scale = Math.min(surfaceSize.width / GRAPH_WIDTH, surfaceSize.height / GRAPH_HEIGHT);
  return {
    scale,
    offsetX: (surfaceSize.width - GRAPH_WIDTH * scale) / 2,
    offsetY: (surfaceSize.height - GRAPH_HEIGHT * scale) / 2,
    viewport,
  };
}

export function projectGraphPointToScreen(x: number, y: number, surface: GraphSurfaceProjection): { x: number; y: number } {
  return {
    x: surface.offsetX + (x * surface.viewport.scale + surface.viewport.tx) * surface.scale,
    y: surface.offsetY + (y * surface.viewport.scale + surface.viewport.ty) * surface.scale,
  };
}

export function graphLabelWidth(label: string, prominent: boolean): number {
  const charWidth = prominent ? 9.8 : 8.7;
  return clampNumber(label.length * charWidth + 24, prominent ? 92 : 72, prominent ? 460 : 340);
}

export function truncate(value: string, max: number): string {
  if (value.length <= max) return value;
  return `${value.slice(0, Math.max(0, max - 1))}…`;
}
