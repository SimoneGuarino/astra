import type { MemoryEmbeddingIndexStatus, MemoryGraphSnapshot, MemoryGraphStatus, MemoryQualityDashboard } from "../../../types/memory";

export type MemoryGraphStatusBarProps = {
  status: MemoryGraphStatus | null;
  snapshot: MemoryGraphSnapshot | null;
  embeddingStatus: MemoryEmbeddingIndexStatus | null;
  qualityDashboard: MemoryQualityDashboard | null;
};

export function MemoryGraphStatusBar({ status, snapshot, embeddingStatus, qualityDashboard }: MemoryGraphStatusBarProps) {
  return (
    <section className="memory-graph-status-hud" aria-label="Memory graph status">
      <span>{status?.nodes ?? snapshot?.nodes.length ?? 0} nodes</span>
      <span>{status?.edges ?? snapshot?.edges.length ?? 0} edges</span>
      <span>{embeddingStatus?.embedded_chunks ?? status?.embeddings ?? 0}/{embeddingStatus?.total_chunks ?? status?.chunks ?? 0} vectors</span>
      <span>{embeddingStatus?.backend ?? status?.vector_backend ?? "vector"}</span>
      <span className={`memory-quality-status-pill memory-quality-status-pill--${safeClassName(qualityDashboard?.status ?? "unknown")}`}>
        quality {qualityDashboard ? Math.round(qualityDashboard.score * 100) : "—"}%
      </span>
    </section>
  );
}

function safeClassName(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
}
