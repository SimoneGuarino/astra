import type { MemoryNode, MemoryVerificationStatus } from "../../../types/memory";
import { NODE_KIND_LABELS } from "../layout/memoryGraphLayoutTypes";

type MemoryNodeInspectorProps = {
  node: MemoryNode | null;
  isUpdatingGovernance?: boolean;
  governanceMessage?: string | null;
  onUpdateGovernance?: (status: MemoryVerificationStatus, reason: string) => void;
  onUpdateSalience?: (salience: number, reason: string) => void;
};

const GOVERNANCE_ACTIONS: Array<{
  status: MemoryVerificationStatus;
  label: string;
  title: string;
  reason: string;
  tone: "positive" | "neutral" | "warning" | "danger";
}> = [
  {
    status: "user_confirmed",
    label: "Confirm",
    title: "Conferma questa memoria come verificata dall'utente",
    reason: "user_confirmed_from_memory_node_inspector",
    tone: "positive",
  },
  {
    status: "unverified",
    label: "Unverify",
    title: "Riporta questa memoria allo stato non verificato",
    reason: "user_marked_unverified_from_memory_node_inspector",
    tone: "neutral",
  },
  {
    status: "contradicted",
    label: "Contradict",
    title: "Marca questa memoria come contraddetta / non più affidabile",
    reason: "user_marked_contradicted_from_memory_node_inspector",
    tone: "warning",
  },
  {
    status: "deprecated",
    label: "Deprecate",
    title: "Depreca questa memoria e riducine l'uso nel retrieval normale",
    reason: "user_deprecated_from_memory_node_inspector",
    tone: "danger",
  },
];

export function MemoryNodeInspector({
  node,
  isUpdatingGovernance = false,
  governanceMessage = null,
  onUpdateGovernance,
  onUpdateSalience,
}: MemoryNodeInspectorProps) {
  if (!node) {
    return (
      <article className="memory-graph-card">
        <h4>Node inspector</h4>
        <p className="desktop-agent-muted">Seleziona un nodo per vedere dettagli, source, confidenza e tags.</p>
      </article>
    );
  }

  const canGovern = Boolean(onUpdateGovernance);
  const saliencePercent = Math.round(Math.max(0, Math.min(1, node.salience)) * 100);

  return (
    <article className="memory-graph-card memory-graph-card--node-inspector">
      <div className="memory-graph-node-header">
        <h4>{node.title}</h4>
        <span>{NODE_KIND_LABELS[node.kind] ?? node.kind}</span>
      </div>
      <p>{node.summary}</p>
      <div className="memory-graph-meta-grid">
        <span>Confidence <strong>{node.confidence.toFixed(2)}</strong></span>
        <span>Salience <strong>{node.salience.toFixed(2)}</strong></span>
        <span>Verified <strong>{node.verification_status}</strong></span>
      </div>

      <section className="memory-node-governance-panel" aria-label="Memory governance actions">
        <div className="memory-node-governance-header">
          <div>
            <strong>Governance</strong>
            <span>Correggi peso e stato della memoria selezionata.</span>
          </div>
          {isUpdatingGovernance ? <span className="memory-node-governance-badge">saving</span> : null}
        </div>

        <div className="memory-node-governance-actions">
          {GOVERNANCE_ACTIONS.map((action) => (
            <button
              key={action.status}
              type="button"
              className={`memory-node-governance-action memory-node-governance-action--${action.tone} ${node.verification_status === action.status ? "memory-node-governance-action--active" : ""}`}
              disabled={!canGovern || isUpdatingGovernance || node.verification_status === action.status}
              title={action.title}
              onClick={() => onUpdateGovernance?.(action.status, action.reason)}
            >
              {action.label}
            </button>
          ))}
        </div>

        <label className="memory-node-salience-control">
          <span>Salience</span>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={saliencePercent}
            disabled={!onUpdateSalience || isUpdatingGovernance}
            onChange={(event) => onUpdateSalience?.(Number(event.currentTarget.value) / 100, "user_adjusted_salience_from_memory_node_inspector")}
          />
          <strong>{saliencePercent}%</strong>
        </label>

        {governanceMessage ? <p className="memory-node-governance-message">{governanceMessage}</p> : null}
      </section>

      {node.content ? (
        <details className="memory-node-content-details">
          <summary>Evidence / content</summary>
          <p>{node.content}</p>
        </details>
      ) : null}

      {node.tags.length ? (
        <div className="memory-graph-tags">
          {node.tags.slice(0, 12).map((tag) => <span key={tag}>{tag}</span>)}
        </div>
      ) : null}
      {node.source ? <p className="desktop-agent-muted">Source: {node.source}</p> : null}
    </article>
  );
}
