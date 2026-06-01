import type { MemoryHealthRepairAction, MemoryQualityDashboard } from "../../../types/memory";

function actionBadge(action: MemoryHealthRepairAction) {
  if (action.requires_user_review) return "Review";
  if (action.can_run_automatically) return "Auto";
  return "Blocked";
}

function priorityLabel(priority: string) {
  switch (priority) {
    case "critical": return "Critical";
    case "high": return "High";
    case "medium": return "Medium";
    case "low": return "Low";
    default: return priority || "Normal";
  }
}

export function MemoryQualityOverlay({ dashboard }: { dashboard: MemoryQualityDashboard }) {
  const plan = dashboard.repair_plan ?? null;
  const automaticActions = plan?.actions.filter((action) => action.can_run_automatically && !action.requires_user_review) ?? [];
  const reviewActions = plan?.actions.filter((action) => action.requires_user_review) ?? [];

  return (
    <aside className="memory-quality-dashboard-overlay" aria-label="Memory quality dashboard">
      <div className="memory-quality-dashboard-header">
        <span>Memory quality</span>
        <strong>{Math.round(dashboard.score * 100)}%</strong>
      </div>
      <p>{dashboard.summary}</p>
      <div className="memory-quality-dashboard-grid">
        <span>Nodes</span><strong>{dashboard.totals.nodes}</strong>
        <span>Semantic</span><strong>{dashboard.semantic.semantic_nodes}</strong>
        <span>Episode only</span><strong>{dashboard.semantic.episode_only_nodes}</strong>
        <span>Pending vectors</span><strong>{dashboard.embeddings.pending_chunks}</strong>
        <span>Pending RC</span><strong>{dashboard.reconsolidation.pending_candidates}</strong>
        <span>Confirmed</span><strong>{dashboard.governance.user_confirmed + dashboard.governance.system_verified}</strong>
        <span>Review needed</span><strong>{plan?.review_action_count ?? 0}</strong>
      </div>

      {plan ? (
        <div className="memory-quality-dashboard-section memory-quality-repair-plan">
          <span>Autopilot repair plan</span>
          <p>{plan.summary}</p>
          <div className="memory-quality-repair-summary">
            <strong>{automaticActions.length}</strong><small>automatic</small>
            <strong>{reviewActions.length}</strong><small>review</small>
          </div>
          <div className="memory-quality-repair-actions">
            {plan.actions.slice(0, 5).map((action) => (
              <article key={action.id} className={`memory-quality-repair-action memory-quality-repair-action--${action.priority}`}>
                <div>
                  <strong>{action.title}</strong>
                  <span>{actionBadge(action)} · {priorityLabel(action.priority)} · {action.affected_count} item{action.affected_count === 1 ? "" : "s"}</span>
                </div>
                <p>{action.description}</p>
              </article>
            ))}
          </div>
        </div>
      ) : null}

      {dashboard.warnings.length ? (
        <div className="memory-quality-dashboard-section">
          <span>Needs attention</span>
          {dashboard.warnings.slice(0, 4).map((warning) => <p key={warning}>{warning}</p>)}
        </div>
      ) : null}
      {dashboard.recommendations.length ? (
        <div className="memory-quality-dashboard-section">
          <span>Recommended</span>
          {dashboard.recommendations.slice(0, 3).map((recommendation) => <p key={recommendation}>{recommendation}</p>)}
        </div>
      ) : null}
    </aside>
  );
}
