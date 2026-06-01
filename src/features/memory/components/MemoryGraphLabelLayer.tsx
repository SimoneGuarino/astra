import type { MemoryGraphSvgLabel } from "../hooks/useMemoryGraphSvgLabels";
import { safeClassName } from "../layout/memoryGraphLayoutEngine";

export function MemoryGraphLabelLayer({ labels }: { labels: MemoryGraphSvgLabel[] }) {
  return (
    <>
      {labels.map((label) => (
        <g
          key={`label-${label.id}`}
          className={`memory-graph-svg-label memory-graph-svg-label--${safeClassName(label.kind)} ${label.selected ? "memory-graph-svg-label--selected" : ""} ${label.important ? "memory-graph-svg-label--important" : ""} ${label.activated ? "memory-graph-svg-label--active" : ""}`}
          transform={`translate(${label.x} ${label.y})`}
          aria-hidden="true"
        >
          <rect
            x={-label.width / 2}
            y={-label.height / 2}
            width={label.width}
            height={label.height}
            rx={label.height / 2}
            className="memory-graph-svg-label-plate"
          />
          <text
            textAnchor="middle"
            dominantBaseline="central"
            fontSize={label.fontSize}
            className="memory-graph-svg-label-text"
          >
            {label.label}
          </text>
        </g>
      ))}
    </>
  );
}
