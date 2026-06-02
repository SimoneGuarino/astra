import type {
  MemoryJobQueueSnapshot,
  MemoryRagCloseoutGate,
  MemoryRagCloseoutSnapshot,
  MemoryRagIntegrityReport,
  MemoryRagRecommendedMaintenanceReceipt,
} from "../../../types/memory";

const STATUS_LABELS: Record<string, string> = {
  closeout_ready: "Closeout ready",
  ready_with_warnings: "Ready with warnings",
  needs_maintenance: "Needs maintenance",
  blocked: "Blocked",
  enterprise_ready: "Enterprise ready",
  needs_hardening: "Needs hardening",
  healthy: "Healthy",
  backpressured: "Backpressured",
  saturated: "Saturated",
  degraded: "Degraded",
  queued: "Queued",
  planned: "Planned",
  no_action_needed: "No action needed",
};

type MemoryRagControlCenterProps = {
  closeout: MemoryRagCloseoutSnapshot | null;
  integrity: MemoryRagIntegrityReport | null;
  queue: MemoryJobQueueSnapshot | null;
  maintenanceReceipt: MemoryRagRecommendedMaintenanceReceipt | null;
  status: string | null;
  isBusy: boolean;
  onRefresh: () => void;
  onDryRun: () => void;
  onRunRecommended: () => void;
  onRunKnowledgeRefresh?: () => void;
  onBuildKnowledgePacks?: () => void;
};

export function MemoryRagControlCenter({
  closeout,
  integrity,
  queue,
  maintenanceReceipt,
  status,
  isBusy,
  onRefresh,
  onDryRun,
  onRunRecommended,
  onRunKnowledgeRefresh,
  onBuildKnowledgePacks,
}: MemoryRagControlCenterProps) {
  const closeoutStatus = closeout?.status ?? "unknown";
  const qualityPercent = typeof closeout?.summary?.quality_score_percent === "number"
    ? closeout.summary.quality_score_percent
    : typeof closeout?.summary?.quality_score === "number"
      ? closeout.summary.quality_score * 100
      : null;
  const queueStatus = queue?.status ?? closeout?.summary?.queue_status ?? "unknown";
  const blockingGates = closeout?.gates?.filter((gate) => gate.status === "block") ?? [];
  const warningGates = closeout?.gates?.filter((gate) => gate.status === "warn") ?? [];
  const primaryGates = selectPrimaryGates(closeout?.gates ?? []);
  const pendingEmbeddings = closeout?.summary?.pending_embeddings ?? closeout?.embedding_status?.pending_chunks ?? 0;
  const pendingReconsolidation = closeout?.summary?.pending_reconsolidation ?? closeout?.quality?.reconsolidation?.pending_candidates ?? 0;
  const plannedActions = maintenanceReceipt?.planned_actions ?? [];

  return (
    <article className={`memory-graph-card memory-rag-control-center memory-rag-control-center--${safeStatusClass(closeoutStatus)}`}>
      <div className="memory-rag-control-center__header">
        <div>
          <h4>Memory/RAG Control Center</h4>
          <p>{closeout?.release_recommendation ?? integrity?.summary ?? "Backend governed readiness snapshot not loaded yet."}</p>
        </div>
        <span className={`memory-rag-status-pill memory-rag-status-pill--${safeStatusClass(closeoutStatus)}`}>
          {formatStatus(closeoutStatus)}
        </span>
      </div>

      <div className="memory-rag-control-center__metrics" aria-label="Memory RAG runtime metrics">
        <Metric label="Quality" value={qualityPercent !== null ? `${Math.round(qualityPercent)}%` : "n/a"} />
        <Metric label="Queue" value={formatStatus(queueStatus)} />
        <Metric label="Pending vectors" value={String(pendingEmbeddings)} />
        <Metric label="Reconsolidation" value={String(pendingReconsolidation)} />
      </div>

      <div className="memory-rag-control-center__actions">
        <button type="button" onClick={onRunRecommended} disabled={isBusy || closeoutStatus === "blocked"}>
          Run recommended maintenance
        </button>
        <button type="button" onClick={onDryRun} disabled={isBusy}>
          Plan only
        </button>
        <button type="button" onClick={onRefresh} disabled={isBusy}>
          Refresh snapshot
        </button>
        {onRunKnowledgeRefresh ? (
          <button type="button" onClick={onRunKnowledgeRefresh} disabled={isBusy} title="Detect stale/temporal memory and run bounded deep-search refresh without manual claim review">
            Refresh stale knowledge
          </button>
        ) : null}
        {onBuildKnowledgePacks ? (
          <button type="button" onClick={onBuildKnowledgePacks} disabled={isBusy} title="Build bounded local domain-brain knowledge packs from the current Memory Graph">
            Build domain brain
          </button>
        ) : null}
      </div>

      {status ? <p className="memory-rag-control-center__status">{status}</p> : null}

      {blockingGates.length || warningGates.length ? (
        <div className="memory-rag-control-center__alerts">
          {blockingGates.slice(0, 3).map((gate) => <GateAlert key={gate.id} gate={gate} />)}
          {warningGates.slice(0, Math.max(0, 3 - blockingGates.length)).map((gate) => <GateAlert key={gate.id} gate={gate} />)}
        </div>
      ) : null}

      {primaryGates.length ? (
        <div className="memory-rag-gate-grid">
          {primaryGates.map((gate) => (
            <div key={gate.id} className={`memory-rag-gate memory-rag-gate--${safeStatusClass(gate.status)}`}>
              <strong>{gate.title}</strong>
              <span>{formatStatus(gate.status)}</span>
              <small>{gate.summary}</small>
            </div>
          ))}
        </div>
      ) : null}

      {plannedActions.length ? (
        <div className="memory-rag-planned-actions">
          <strong>Last recommended plan</strong>
          {plannedActions.slice(0, 4).map((action, index) => (
            <p key={`${action.kind}:${index}`}>
              {formatStatus(action.kind)} · {action.priority} · {action.reason ?? action.queued_command ?? "bounded maintenance"}
            </p>
          ))}
        </div>
      ) : null}

      {queue?.active_jobs?.length ? (
        <div className="memory-rag-active-jobs">
          <strong>Active memory jobs</strong>
          {queue.active_jobs.slice(0, 4).map((job) => (
            <p key={job.job_id}>{formatStatus(job.kind)} · {job.status} · {Math.round(job.age_ms / 1000)}s</p>
          ))}
        </div>
      ) : null}
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="memory-rag-metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function GateAlert({ gate }: { gate: MemoryRagCloseoutGate }) {
  return (
    <div className={`memory-rag-alert memory-rag-alert--${safeStatusClass(gate.status)}`}>
      <strong>{gate.title}</strong>
      <span>{gate.next_action || gate.summary}</span>
    </div>
  );
}

function selectPrimaryGates(gates: MemoryRagCloseoutGate[]): MemoryRagCloseoutGate[] {
  const priority = ["graph_persistence", "bounded_job_queue", "quality_score", "semantic_density", "embedding_coverage", "governance_safety"];
  const byId = new Map(gates.map((gate) => [gate.id, gate]));
  const selected = priority.map((id) => byId.get(id)).filter(Boolean) as MemoryRagCloseoutGate[];
  if (selected.length >= 4) return selected;
  return [...selected, ...gates.filter((gate) => !priority.includes(gate.id)).slice(0, 6 - selected.length)];
}

function safeStatusClass(value: string | null | undefined): string {
  return String(value ?? "unknown").replace(/[^a-z0-9_-]/gi, "_").toLowerCase();
}

function formatStatus(value: string | null | undefined): string {
  const key = String(value ?? "unknown");
  return STATUS_LABELS[key] ?? key.replace(/_/g, " ");
}
