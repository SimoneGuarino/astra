import type { MemoryEmbeddingIndexStatus, MemoryEmbeddingRebuildReceipt } from "../../../types/memory";

type MemoryEmbeddingSummaryProps = {
  receipt: MemoryEmbeddingRebuildReceipt | null;
  status: MemoryEmbeddingIndexStatus | null;
  maintenanceStatus: string | null;
};

export function MemoryEmbeddingSummary({ receipt, status, maintenanceStatus }: MemoryEmbeddingSummaryProps) {
  const providerKind = typeof receipt?.metadata?.provider_kind === "string"
    ? receipt.metadata.provider_kind
    : typeof status?.metadata?.provider_kind === "string"
      ? status.metadata.provider_kind
      : "configured provider";
  const fallbackEnabled = typeof receipt?.metadata?.fallback_enabled === "boolean"
    ? receipt.metadata.fallback_enabled
    : null;

  return (
    <article className="memory-graph-card memory-graph-card--activation">
      <h4>Brain RAG / Vector retrieval</h4>
      <p>Backend: {status?.backend ?? "not indexed yet"}</p>
      <p>Provider: {status?.provider ?? "stable-local-hash-v1"}</p>
      <p>Adapter: {providerKind}</p>
      <p>Chunks: {status?.embedded_chunks ?? 0}/{status?.total_chunks ?? 0} embedded · pending {status?.pending_chunks ?? 0}</p>
      {receipt ? <p>Last rebuild: {receipt.indexed_chunks} indexed · {receipt.failed_chunks} failed · {receipt.model}</p> : null}
      {maintenanceStatus ? <p>Maintenance: {maintenanceStatus}</p> : null}
      {fallbackEnabled !== null ? <p className="desktop-agent-muted">Fallback locale: {fallbackEnabled ? "enabled" : "disabled"}</p> : null}
    </article>
  );
}
