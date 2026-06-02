import {
  IoGitNetworkOutline,
  IoOptionsOutline,
  IoRefreshOutline,
  IoSearchOutline,
  IoSparklesOutline,
  IoTextOutline,
  IoEyeOffOutline,
  IoShieldCheckmarkOutline,
} from "react-icons/io5";
import type { MemoryGraphViewMode } from "./MemoryGraphControlsOverlay";

export type MemoryGraphToolbarProps = {
  isBusy: boolean;
  labelsVisible: boolean;
  qualityExpanded: boolean;
  queryExpanded: boolean;
  controlsExpanded: boolean;
  reviewExpanded: boolean;
  reviewCount: number;
  graphMode: MemoryGraphViewMode;
  autopilotStatus?: string | null;
  onRunAutopilot: () => void;
  onToggleControls: () => void;
  onToggleGraphMode: () => void;
  onToggleQuality: () => void;
  onToggleReview: () => void;
  onRefresh: () => void;
  onToggleLabels: () => void;
  onToggleSearch: () => void;
};

export function MemoryGraphToolbar({
  isBusy,
  labelsVisible,
  qualityExpanded,
  queryExpanded,
  controlsExpanded,
  reviewExpanded,
  reviewCount,
  graphMode,
  autopilotStatus,
  onRunAutopilot,
  onToggleControls,
  onToggleGraphMode,
  onToggleQuality,
  onToggleReview,
  onRefresh,
  onToggleLabels,
  onToggleSearch,
}: MemoryGraphToolbarProps) {
  return (
    <section className="memory-graph-hud memory-graph-hud--autopilot" aria-label="Astra Cognitive Memory controls">
      <div className="memory-graph-hud-title">
        <span>ASTRA COGNITIVE MEMORY</span>
        <small>{autopilotStatus || "Governed Brain RAG maintenance"}</small>
      </div>
      <div className="memory-graph-hud-actions">
        <button
          type="button"
          className="memory-graph-icon-button memory-graph-icon-button--primary"
          onClick={onRunAutopilot}
          disabled={isBusy}
          title="Run Cognitive Copilot: canonical cleanup, safe soft-merge auto-apply, embeddings and memory maintenance"
        >
          <IoSparklesOutline />
        </button>
        <button
          type="button"
          className={`memory-graph-icon-button ${graphMode === "local" ? "memory-graph-icon-button--active" : ""}`}
          onClick={onToggleGraphMode}
          title={graphMode === "local" ? "Switch to Global Vault graph" : "Switch to Local Focus graph"}
          aria-pressed={graphMode === "local"}
        >
          <IoGitNetworkOutline />
          <span className="memory-graph-toolbar-mini-label">{graphMode === "local" ? "LF" : "GV"}</span>
        </button>
        <button
          type="button"
          className={`memory-graph-icon-button ${qualityExpanded ? "memory-graph-icon-button--active" : ""}`}
          onClick={onToggleQuality}
          title="Memory health and quality"
          aria-pressed={qualityExpanded}
        >
          <IoShieldCheckmarkOutline />
        </button>
        <button
          type="button"
          className={`memory-graph-icon-button memory-graph-review-button ${reviewExpanded ? "memory-graph-icon-button--active" : ""}`}
          onClick={onToggleReview}
          title="Governed Brain Review queue: duplicates and canonical memory candidates proposed by Autopilot/LLM"
          aria-pressed={reviewExpanded}
        >
          <span className="memory-graph-toolbar-mini-label">RV</span>
          {reviewCount > 0 ? <span className="memory-graph-review-badge">{reviewCount > 99 ? "99+" : reviewCount}</span> : null}
        </button>
        <button
          type="button"
          className={`memory-graph-icon-button ${controlsExpanded ? "memory-graph-icon-button--active" : ""}`}
          onClick={onToggleControls}
          title="Graph controls"
          aria-pressed={controlsExpanded}
        >
          <IoOptionsOutline />
        </button>
        <button type="button" className="memory-graph-icon-button" onClick={onRefresh} disabled={isBusy} title="Refresh graph">
          <IoRefreshOutline />
        </button>
        <button
          type="button"
          className={`memory-graph-icon-button memory-graph-label-toggle ${labelsVisible ? "memory-graph-icon-button--active" : ""}`}
          onPointerDown={(event) => event.stopPropagation()}
          onClick={onToggleLabels}
          title={labelsVisible ? "Nascondi label nodi" : "Mostra label nodi"}
          aria-label={labelsVisible ? "Nascondi label nodi" : "Mostra label nodi"}
          aria-pressed={labelsVisible}
        >
          {labelsVisible ? <IoTextOutline /> : <IoEyeOffOutline />}
        </button>
        <button
          type="button"
          className={`memory-graph-icon-button ${queryExpanded ? "memory-graph-icon-button--active" : ""}`}
          onClick={onToggleSearch}
          title="Search memory"
        >
          <IoSearchOutline />
        </button>
      </div>
    </section>
  );
}
