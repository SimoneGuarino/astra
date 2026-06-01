import type { MemoryActivationEventPayload } from "../../../types/memory";

type MemoryActivationSummaryProps = {
  payload: MemoryActivationEventPayload | null;
};

export function MemoryActivationSummary({ payload }: MemoryActivationSummaryProps) {
  if (!payload) {
    return (
      <article className="memory-graph-card">
        <h4>Current activation</h4>
        <p className="desktop-agent-muted">Nessuna attivazione recente. Quando Astra userà la memoria, i nodi coinvolti si illumineranno qui.</p>
      </article>
    );
  }

  return (
    <article className="memory-graph-card memory-graph-card--activation">
      <h4>Current activation</h4>
      <p><strong>Query:</strong> {payload.root_query || "latest retrieval"}</p>
      <p><strong>Nodes:</strong> {payload.activated_node_ids?.length ?? payload.node_count ?? 0}</p>
      <p><strong>Edges:</strong> {payload.activated_edge_ids?.length ?? payload.edge_count ?? 0}</p>
      <p><strong>Source:</strong> {payload.source ?? "memory_graph"}</p>
    </article>
  );
}
