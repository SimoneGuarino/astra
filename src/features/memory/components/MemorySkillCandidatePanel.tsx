import type { MemorySkillCandidate } from "../../../types/memory";
import { truncate } from "../layout/memoryGraphLayoutEngine";

type MemorySkillCandidatePanelProps = {
  candidates: MemorySkillCandidate[];
  onExtract: () => void;
  onUpdateStatus: (candidateId: string, status: MemorySkillCandidate["status"]) => void;
  status: string | null;
};

export function MemorySkillCandidatePanel({
  candidates,
  onExtract,
  onUpdateStatus,
  status,
}: MemorySkillCandidatePanelProps) {
  return (
    <article className="memory-graph-card memory-graph-card--skills">
      <div className="memory-graph-node-header">
        <h4>Procedural skills</h4>
        <span>{candidates.length}</span>
      </div>
      <p className="desktop-agent-muted">
        Procedure candidate apprese dalla memoria. Restano advisory finché non sono approvate e non bypassano mai policy, approval o runtime governance.
      </p>
      <button type="button" className="memory-graph-button memory-graph-button--full" onClick={onExtract}>
        Extract skill candidates
      </button>
      {status ? <p className="desktop-agent-muted">{status}</p> : null}
      {candidates.length === 0 ? (
        <p className="desktop-agent-muted">Nessuna skill candidate. Consolida conversazioni, research o procedure per crearle.</p>
      ) : (
        <div className="memory-graph-list memory-skill-list">
          {candidates.slice(0, 8).map((candidate) => (
            <div key={candidate.id} className="memory-skill-item">
              <strong>{truncate(candidate.title, 64)}</strong>
              <span>{candidate.status} · {candidate.risk_level} · {candidate.confidence.toFixed(2)}</span>
              <p>{truncate(candidate.summary, 160)}</p>
              <div className="memory-skill-actions">
                <button type="button" onClick={() => onUpdateStatus(candidate.id, "approved")}>Approve</button>
                <button type="button" onClick={() => onUpdateStatus(candidate.id, "disabled")}>Disable</button>
                <button type="button" onClick={() => onUpdateStatus(candidate.id, "deprecated")}>Deprecate</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}
