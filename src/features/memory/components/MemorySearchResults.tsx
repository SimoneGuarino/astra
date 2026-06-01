import type { MemoryQueryHit } from "../../../types/memory";

type MemorySearchResultsProps = {
  hits: MemoryQueryHit[];
  onSelectNode: (nodeId: string) => void;
};

export function MemorySearchResults({ hits, onSelectNode }: MemorySearchResultsProps) {
  if (hits.length === 0) return null;
  return (
    <article className="memory-graph-card">
      <h4>Search results</h4>
      <div className="memory-graph-list">
        {hits.map((hit) => (
          <button key={hit.node.id} type="button" onClick={() => onSelectNode(hit.node.id)}>
            <strong>{hit.node.title}</strong>
            <span>{hit.score.toFixed(2)} · {hit.reasons.join(", ")}</span>
          </button>
        ))}
      </div>
    </article>
  );
}
