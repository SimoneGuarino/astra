import type { MemoryEdge } from "../../../types/memory";
import type { GraphPoint } from "../layout/memoryGraphLayoutTypes";

export function MemoryGraphEdgeLayer({
  activeEdges,
  edges,
  layoutById,
  viewportScale,
}: {
  activeEdges: Set<string>;
  edges: MemoryEdge[];
  layoutById: Map<string, GraphPoint>;
  viewportScale: number;
}) {
  return (
    <>
      {edges.map((edge) => {
        const from = layoutById.get(edge.from_node_id);
        const to = layoutById.get(edge.to_node_id);
        if (!from || !to) return null;
        const activated = activeEdges.has(edge.id) || (from.activated && to.activated);
        return (
          <line
            key={edge.id}
            x1={from.x}
            y1={from.y}
            x2={to.x}
            y2={to.y}
            className={`memory-graph-edge ${activated ? "memory-graph-edge--active" : ""}`}
            strokeWidth={activated ? 2.3 / viewportScale : Math.max(0.7, edge.weight * 1.7) / viewportScale}
          />
        );
      })}
    </>
  );
}
