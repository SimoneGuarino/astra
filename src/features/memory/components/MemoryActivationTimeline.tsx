import type { MemoryActivation } from "../../../types/memory";
import { truncate } from "../layout/memoryGraphLayoutEngine";

type MemoryActivationTimelineProps = {
  activations: MemoryActivation[];
};

export function MemoryActivationTimeline({ activations }: MemoryActivationTimelineProps) {
  return (
    <article className="memory-graph-card">
      <h4>Activation timeline</h4>
      {activations.length === 0 ? (
        <p className="desktop-agent-muted">Nessuna attivazione persistita.</p>
      ) : (
        <div className="memory-graph-list">
          {activations.slice(0, 10).map((activation) => (
            <div key={activation.id} className="memory-graph-timeline-item">
              <strong>{truncate(activation.root_query, 52)}</strong>
              <span>{activation.activated_node_ids.length} nodes · {new Date(activation.created_at).toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}
