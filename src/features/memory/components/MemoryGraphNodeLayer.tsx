import type { PointerEvent } from "react";
import type { MemoryNode } from "../../../types/memory";
import { NODE_KIND_LABELS, type GraphPoint } from "../layout/memoryGraphLayoutTypes";
import { memoryNodeVisualWeight, safeClassName } from "../layout/memoryGraphLayoutEngine";

export function MemoryGraphNodeLayer({
  layout,
  nodeById,
  onNodePointerDown,
  selectedNodeId,
}: {
  layout: GraphPoint[];
  nodeById: Map<string, MemoryNode>;
  onNodePointerDown: (event: PointerEvent<SVGGElement>, nodeId: string) => void;
  selectedNodeId: string | null;
}) {
  return (
    <>
      {layout.map((point) => {
        const node = nodeById.get(point.id);
        if (!node) return null;
        const selected = selectedNodeId === node.id;
        const important = memoryNodeVisualWeight(node) > 14;
        return (
          <g
            key={node.id}
            className="memory-graph-node-group"
            onPointerDown={(event) => onNodePointerDown(event, node.id)}
            tabIndex={0}
            role="button"
          >
            {point.activated || selected ? <circle cx={point.x} cy={point.y} r={point.radius + 10} className="memory-graph-node-halo" /> : null}
            <circle
              cx={point.x}
              cy={point.y}
              r={point.radius}
              className={`memory-graph-node memory-graph-node--${safeClassName(node.kind)} ${point.activated ? "memory-graph-node--active" : ""} ${selected ? "memory-graph-node--selected" : ""} ${important ? "memory-graph-node--important" : ""}`}
              filter={point.activated || selected ? "url(#memoryGlow)" : undefined}
            />
            <title>{`${node.title}\n${NODE_KIND_LABELS[node.kind] ?? node.kind}\n${node.summary}`}</title>
          </g>
        );
      })}
    </>
  );
}
