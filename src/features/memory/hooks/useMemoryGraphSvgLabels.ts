import { useMemo } from "react";
import type { MemoryNode } from "../../../types/memory";
import type { MemoryGraphLayoutSettings } from "../components/MemoryGraphControlsOverlay";
import type { GraphPoint, GraphSurfaceSize, GraphViewport } from "../layout/memoryGraphLayoutTypes";
import {
  adaptiveLabelLimit,
  clampNumber,
  graphLabelWidth,
  graphSurfaceProjection,
  memoryNodeVisualWeight,
  shouldShowMemoryNodeLabel,
  truncate,
} from "../layout/memoryGraphLayoutEngine";

export type MemoryGraphSvgLabel = {
  id: string;
  label: string;
  kind: string;
  selected: boolean;
  important: boolean;
  activated: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
  fontSize: number;
  priority: number;
};

export function useMemoryGraphSvgLabels({
  labelsVisible,
  layout,
  layoutSettings,
  nodeById,
  nodeCount,
  selectedNodeId,
  surfaceSize,
  viewport,
}: {
  labelsVisible: boolean;
  layout: GraphPoint[];
  layoutSettings: MemoryGraphLayoutSettings;
  nodeById: Map<string, MemoryNode>;
  nodeCount: number;
  selectedNodeId: string | null;
  surfaceSize: GraphSurfaceSize;
  viewport: GraphViewport;
}) {
  return useMemo(() => {
    if (!labelsVisible) return [];
    const surface = graphSurfaceProjection(surfaceSize, viewport);
    const totalScale = Math.max(0.05, (surface?.scale ?? 0.25) * viewport.scale);
    const fontSize = clampNumber((12.5 * layoutSettings.labelSize) / totalScale, 16, 96);
    const height = clampNumber((20 * layoutSettings.labelSize) / totalScale, 28, 132);
    const yOffset = clampNumber((22 * layoutSettings.labelSize) / totalScale, 32, 146);

    const candidates = layout
      .map((point) => {
        const node = nodeById.get(point.id);
        if (!node) return null;
        const selected = selectedNodeId === node.id;
        const important = memoryNodeVisualWeight(node) > 14;
        const activated = point.activated;
        const shouldShow = shouldShowMemoryNodeLabel(layoutSettings.labelMode, { selected, activated, important, nodeCount, viewportScale: viewport.scale });
        if (!shouldShow) return null;

        const label = truncate(node.title, selected || important || activated ? 54 : 34);
        const width = graphLabelWidth(label, selected || important || activated) / totalScale;
        const priority = (selected ? 10000 : 0) + (activated ? 6000 : 0) + (important ? 2600 : 0) + memoryNodeVisualWeight(node) * 40;
        return {
          id: node.id,
          label,
          kind: String(node.kind),
          selected,
          important,
          activated,
          x: point.x,
          y: point.y + point.radius + yOffset,
          width,
          height,
          fontSize,
          priority,
        };
      })
      .filter((value): value is MemoryGraphSvgLabel => Boolean(value));

    const maxLabels = adaptiveLabelLimit(layoutSettings.labelMode, nodeCount, viewport.scale);
    return candidates.sort((left, right) => right.priority - left.priority).slice(0, maxLabels);
  }, [labelsVisible, layout, layoutSettings.labelMode, layoutSettings.labelSize, nodeById, nodeCount, selectedNodeId, surfaceSize, viewport]);
}

