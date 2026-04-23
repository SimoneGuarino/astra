import { useCallback, useEffect, useMemo, useState } from "react";
import { useDesktopAgent } from "../hooks/useDesktopAgent";
import type {
    CapabilityManifest,
    DesktopAuditEvent,
    DesktopPolicySnapshot,
    GoalLoopRun,
    PendingApproval,
    ScreenAnalysisResult,
    ScreenCaptureResult,
    ScreenObservationStatus,
    ToolDescriptor,
} from "../types/desktopAgent";
import { Button } from "../ui/buttons/Button";
import Switch from "../ui/input/Switch";

import { RxReload } from "react-icons/rx";
import { IoClose } from "react-icons/io5";
import { RiScreenshot2Line } from "react-icons/ri";
import { MdOutlinePolicy } from "react-icons/md";
import { LuMonitor } from "react-icons/lu";
import { VscTools } from "react-icons/vsc";

type DesktopAgentPanelProps = {
    isOpen: boolean;
    onClose: () => void;
};

type ViewKey = "overview" | "approvals" | "audit" | "screen";

export function DesktopAgentPanel({ isOpen, onClose }: DesktopAgentPanelProps) {
    const agent = useDesktopAgent();
    const [view, setView] = useState<ViewKey>("overview");
    const [tools, setTools] = useState<ToolDescriptor[]>([]);
    const [policy, setPolicy] = useState<DesktopPolicySnapshot | null>(null);
    const [capabilities, setCapabilities] = useState<CapabilityManifest | null>(null);
    const [approvals, setApprovals] = useState<PendingApproval[]>([]);
    const [audit, setAudit] = useState<DesktopAuditEvent[]>([]);
    const [screenStatus, setScreenStatus] = useState<ScreenObservationStatus | null>(null);
    const [lastCapture, setLastCapture] = useState<ScreenCaptureResult | null>(null);
    const [recentGoalLoop, setRecentGoalLoop] = useState<GoalLoopRun | null>(null);
    const [screenQuestion, setScreenQuestion] = useState("What am I looking at right now?");
    const [screenAnalysis, setScreenAnalysis] = useState<ScreenAnalysisResult | null>(null);
    const [captureFreshForAnalysis, setCaptureFreshForAnalysis] = useState(true);
    const [isBusy, setIsBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        try {
            setError(null);
            const [nextTools, nextPolicy, nextApprovals, nextAudit, nextScreenStatus, nextCapabilities, nextGoalLoop] = await Promise.all([
                agent.listTools(),
                agent.getPolicySnapshot(),
                agent.getPendingApprovals(),
                agent.getRecentAuditEvents(60),
                agent.getScreenObservationStatus(),
                agent.getCapabilityManifest(),
                agent.getRecentGoalLoop(),
            ]);
            setTools(nextTools);
            setPolicy(nextPolicy);
            setApprovals(nextApprovals);
            setAudit(nextAudit);
            setScreenStatus(nextScreenStatus);
            setCapabilities(nextCapabilities);
            setRecentGoalLoop(nextGoalLoop);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    }, [agent]);

    useEffect(() => {
        if (!isOpen) return;
        void refresh();
        const timer = window.setInterval(() => {
            void refresh();
        }, 4000);
        return () => window.clearInterval(timer);
    }, [isOpen, refresh]);

    const approvalCountLabel = useMemo(() => {
        if (approvals.length === 0) return "No pending approvals";
        if (approvals.length === 1) return "1 pending approval";
        return `${approvals.length} pending approvals`;
    }, [approvals.length]);

    const lastPlannerDiagnostic = recentGoalLoop?.planner_diagnostics?.length
        ? recentGoalLoop.planner_diagnostics[recentGoalLoop.planner_diagnostics.length - 1]
        : null;
    const handoffVerification = recentGoalLoop?.browser_handoff?.verification;
    const lastActionability = lastPlannerDiagnostic?.visible_actionability;
    const lastTargetConfidence = lastPlannerDiagnostic?.target_confidence;
    const actionabilityGaps = lastActionability?.gaps?.length
        ? lastActionability.gaps.join(", ")
        : null;
    const lastFocusedRequest = recentGoalLoop?.focused_perception_requests?.length
        ? recentGoalLoop.focused_perception_requests[recentGoalLoop.focused_perception_requests.length - 1]
        : null;
    const lastExecution = recentGoalLoop?.executed_steps?.length
        ? recentGoalLoop.executed_steps[recentGoalLoop.executed_steps.length - 1]
        : null;
    const lastGeometry = lastExecution?.geometry;

    const handleApproval = useCallback(
        async (actionId: string, decision: "approve" | "reject") => {
            try {
                setIsBusy(true);
                setError(null);
                if (decision === "approve") {
                    await agent.approveAction(actionId, "Approved from desktop panel");
                } else {
                    await agent.rejectAction(actionId, "Rejected from desktop panel");
                }
                await refresh();
            } catch (err) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setIsBusy(false);
            }
        },
        [agent, refresh]
    );

    const handleToggleObservation = useCallback(async () => {
        if (!screenStatus) return;
        try {
            setIsBusy(true);
            setError(null);
            const nextStatus = await agent.setScreenObservationEnabled(!screenStatus.enabled);
            setScreenStatus(nextStatus);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setIsBusy(false);
        }
    }, [agent, screenStatus]);

    const handleCapture = useCallback(async () => {
        try {
            setIsBusy(true);
            setError(null);
            const capture = await agent.captureScreenSnapshot();
            setLastCapture(capture);
            const nextStatus = await agent.getScreenObservationStatus();
            setScreenStatus(nextStatus);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setIsBusy(false);
        }
    }, [agent]);

    const handleAnalyzeScreen = useCallback(async () => {
        try {
            setIsBusy(true);
            setError(null);
            const result = await agent.analyzeScreenContext({
                question: screenQuestion.trim() || undefined,
                capture_fresh: captureFreshForAnalysis,
            });
            setScreenAnalysis(result);
            const nextStatus = await agent.getScreenObservationStatus();
            setScreenStatus(nextStatus);
            setLastCapture({
                capture_id: result.analysis_id,
                captured_at: result.captured_at,
                image_path: result.image_path,
                width: null,
                height: null,
                bytes: 0,
                provider: result.provider,
            });
            await refresh();
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setIsBusy(false);
        }
    }, [agent, captureFreshForAnalysis, refresh, screenQuestion]);

    if (!isOpen) {
        return null;
    }

    return (
        <aside className="flex p-4 bg-white/60 flex-col absolute bottom-0 
        z-10 w-full h-1/2 backdrop-blur-sm border-t border-gray-200 gap-2 min-h-[300px]">
            <div className="desktop-agent-panel__header">
                <div>
                    <p className="desktop-agent-panel__kicker">DESKTOP AGENT</p>
                    <h2>Approval Center</h2>
                    <p className="desktop-agent-panel__subtitle">{approvalCountLabel}</p>
                </div>
                <div>
                    <Button variant="text" title="Aggiorna i dati" radius="full" size="xs" onClick={() => void refresh()}>
                        <RxReload />
                    </Button>
                    <Button variant="text" title="Chiudi sezione" radius="full" size="xs" onClick={onClose}>
                        <IoClose />
                    </Button>
                </div>
            </div>

            <div className="desktop-agent-tabs">
                {(["overview", "screen", "approvals", "audit"] as ViewKey[]).map((key) => (
                    <button
                        key={key}
                        className={`desktop-agent-tab ${view === key ? "active" : ""}`}
                        onClick={() => setView(key)}
                        type="button"
                    >
                        {key}
                    </button>
                ))}
            </div>

            {error ? <div className="desktop-agent-error">{error}</div> : null}

            {view === "overview" ? (
                <div className="desktop-agent-section">
                    <section className="desktop-agent-card-grid">
                        <article className="desktop-agent-card space-y-2">
                            <span className="flex items-center text-center gap-1"><MdOutlinePolicy size={20} /><h3 className="text-lg">Policy</h3></span>
                            <div className="text-sm">
                                <p>Allowed roots: <strong>{policy?.allowed_roots.length ?? 0}</strong></p>
                                <p>Terminal allowlist: <strong>{policy?.terminal_allowed_commands.length ?? 0}</strong></p>
                                <p>Browser enabled: <strong>{policy?.browser_enabled ? "yes" : "no"}</strong></p>
                                <p>Desktop control: <strong>{policy?.desktop_control_enabled ? "yes" : "no"}</strong></p>
                                <p>Pending approvals: <strong>{capabilities?.approvals.pending_count ?? 0}</strong></p>
                            </div>

                        </article>
                        <article className="desktop-agent-card space-y-2">
                            <span className="flex items-center text-center gap-1"><LuMonitor size={20} /><h3 className="text-lg">Screen</h3></span>
                            <div className="text-sm">
                                <h3>Screen</h3>
                                <p>Provider: <strong>{screenStatus?.provider ?? "unknown"}</strong></p>
                                <p>Enabled: <strong>{screenStatus?.enabled ? "yes" : "no"}</strong></p>
                                <p>Captures: <strong>{screenStatus?.capture_count ?? 0}</strong></p>
                                <p>Vision: <strong>{capabilities?.screen.vision_model_name ?? (capabilities?.screen.analysis_available ? "available" : "unavailable")}</strong></p>
                                <p className="desktop-agent-muted">{screenStatus?.note ?? "No status available"}</p>
                                <div className="mt-2 space-x-2">
                                    <Switch checked={screenStatus?.enabled} onChange={() => void handleToggleObservation()} title={screenStatus?.enabled ? "Disabilita osservazione" : "Abilita osservazione"} disabled={isBusy} />
                                    <Button variant="text" title="Cattura lo schermo (screenshot)" radius="full" size="xs" disabled={isBusy} onClick={() => void handleCapture()}>
                                        <RiScreenshot2Line size={18} />
                                    </Button>
                                </div>

                            </div>
                            {lastCapture ? (
                                <p className="desktop-agent-muted">Last capture: {lastCapture.image_path}</p>
                            ) : null}
                        </article>
                        <article className="desktop-agent-card space-y-2">
                            <span className="flex items-center text-center gap-1"><LuMonitor size={20} /><h3 className="text-lg">Goal loop</h3></span>
                            {recentGoalLoop ? (
                                <div className="text-sm">
                                    <p>Status: <strong>{recentGoalLoop.status}</strong></p>
                                    <p>Iteration: <strong>{recentGoalLoop.iteration_count}</strong></p>
                                    <p>Retries: <strong>{recentGoalLoop.retries_used}/{recentGoalLoop.retry_budget}</strong></p>
                                    <p>Strategy: <strong>{recentGoalLoop.current_strategy ?? "none"}</strong></p>
                                    <p>Planner: <strong>{lastPlannerDiagnostic?.source ?? "unknown"}</strong></p>
                                    <p>Handoff: <strong>{recentGoalLoop.browser_handoff?.status ?? "not recorded"}</strong></p>
                                    <p>Page verified: <strong>{recentGoalLoop.browser_handoff?.page_verified ? "yes" : "no"}</strong></p>
                                    <p>Handoff page kind: <strong>{recentGoalLoop.browser_handoff?.page_kind_hint ?? "unknown"}</strong></p>
                                    <p>Normalized page kind: <strong>{handoffVerification?.normalized_page_kind ?? "unknown"}</strong></p>
                                    <p>Handoff decision: <strong>{handoffVerification?.decision ?? "unknown"}</strong></p>
                                    <p>Decision: <strong>{lastPlannerDiagnostic?.decision_status ?? "unknown"}</strong></p>
                                    <p>Visibility: <strong>{lastPlannerDiagnostic?.visibility_assessment ?? "unknown"}</strong></p>
                                    <p>Scroll: <strong>{lastPlannerDiagnostic?.scroll_intent ?? "unknown"}</strong></p>
                                    <p>Actionability: <strong>{lastActionability?.status ?? "unknown"}</strong></p>
                                    <p>Refinement: <strong>{recentGoalLoop.visible_refinement_used ? "yes" : "no"}</strong></p>
                                    <p>Refinement strategy: <strong>{lastActionability?.refinement_strategy ?? "none"}</strong></p>
                                    <p>Fallback available: <strong>{lastActionability?.safe_fallback_available ? "yes" : "no"}</strong></p>
                                    <p>Fallback used: <strong>{lastActionability?.fallback_source_used ?? "none"}</strong></p>
                                    <p>Off-screen stage: <strong>{lastActionability?.offscreen_inference_stage ?? "not_applicable"}</strong></p>
                                    <p>Target confidence: <strong>{lastTargetConfidence ? lastTargetConfidence.derived_confidence.toFixed(2) : "unknown"}</strong></p>
                                    <p>Confidence threshold: <strong>{lastTargetConfidence ? lastTargetConfidence.required_threshold.toFixed(2) : "unknown"}</strong></p>
                                    <p>Confidence derived: <strong>{lastTargetConfidence ? (lastTargetConfidence.confidence_was_derived ? "yes" : "no") : "unknown"}</strong></p>
                                    <p>Item confidence: <strong>{lastTargetConfidence?.raw_item_confidence?.toFixed(2) ?? "missing"}</strong></p>
                                    <p>Region confidence: <strong>{lastTargetConfidence?.raw_region_confidence?.toFixed(2) ?? "missing"}</strong></p>
                                    <p>Page confidence: <strong>{lastTargetConfidence?.raw_page_confidence?.toFixed(2) ?? "missing"}</strong></p>
                                    <p>Signal states: <strong>{lastTargetConfidence ? `${lastTargetConfidence.item_confidence_state ?? "missing"} / ${lastTargetConfidence.region_confidence_state ?? "missing"} / ${lastTargetConfidence.page_confidence_state ?? "missing"}` : "unknown"}</strong></p>
                                    <p>Last perception mode: <strong>{lastFocusedRequest?.mode ?? "none"}</strong></p>
                                    <p>Last routing decision: <strong>{lastFocusedRequest?.routing_decision ?? "none"}</strong></p>
                                    <p>Target anchor present: <strong>{lastFocusedRequest?.target_region_anchor_present ? "yes" : "no"}</strong></p>
                                    <p>Fresh recapture enforced: <strong>{lastExecution?.fresh_capture_required ? "yes" : "no"}</strong></p>
                                    <p>Fresh recapture used: <strong>{lastExecution?.fresh_capture_used ? "yes" : "no"}</strong></p>
                                    <p>Stale reuse prevented: <strong>{recentGoalLoop.stale_capture_reuse_prevented ? "yes" : "no"}</strong></p>
                                    <p>Geometry interpretation: <strong>{lastGeometry?.interpretation ?? "unknown"}</strong></p>
                                    <p>Geometry valid: <strong>{lastGeometry ? (lastGeometry.validation_passed ? "yes" : "no") : "unknown"}</strong></p>
                                    <p>Geometry translated: <strong>{lastGeometry ? (lastGeometry.translation_applied ? "yes" : "no") : "unknown"}</strong></p>
                                    <p>Final click point: <strong>{lastGeometry?.final_x != null && lastGeometry?.final_y != null ? `${lastGeometry.final_x}, ${lastGeometry.final_y}` : "unknown"}</strong></p>
                                    <p>Browser recovery: <strong>{recentGoalLoop.browser_recovery_status ?? "not_needed"}</strong></p>
                                    <p>Browser reacquired: <strong>{recentGoalLoop.browser_recovery_used ? "yes" : "no"}</strong></p>
                                    <p>Repeated-click protection: <strong>{recentGoalLoop.repeated_click_protection_triggered ? "yes" : "no"}</strong></p>
                                    {handoffVerification?.reason ? (
                                        <p>Handoff reason: <strong>{handoffVerification.reason}</strong></p>
                                    ) : null}
                                    {lastPlannerDiagnostic?.rejection_code ? (
                                        <p>Planner reason: <strong>{lastPlannerDiagnostic.rejection_code}</strong></p>
                                    ) : null}
                                    {actionabilityGaps ? (
                                        <p>Grounding gaps: <strong>{actionabilityGaps}</strong></p>
                                    ) : null}
                                    {lastTargetConfidence?.reason ? (
                                        <p>Confidence reason: <strong>{lastTargetConfidence.reason}</strong></p>
                                    ) : null}
                                    {lastGeometry?.reason ? (
                                        <p>Geometry reason: <strong>{lastGeometry.reason}</strong></p>
                                    ) : null}
                                    <p>Focused perception: <strong>{recentGoalLoop.focused_perception_used ? "yes" : "no"}</strong></p>
                                    <p>Verifier: <strong>{recentGoalLoop.verifier_status ?? "unknown"}</strong></p>
                                    {recentGoalLoop.failure_reason ? <p className="desktop-agent-muted">{recentGoalLoop.failure_reason}</p> : null}
                                </div>
                            ) : (
                                <p className="desktop-agent-muted">No goal loop recorded yet.</p>
                            )}
                        </article>
                    </section>

                    <section className="desktop-agent-card space-y-2">
                        <span className="flex items-center text-center gap-1"><VscTools size={20} /><h3 className="text-lg">Tools</h3></span>
                        <div className="flex flex-col text-sm pr-2 gap-2">
                            {tools.map((tool) => (
                                <div key={tool.tool_name} className="desktop-agent-tool-row">
                                    <div>
                                        <strong>{tool.tool_name}</strong>
                                        <p>{tool.description}</p>
                                    </div>
                                    <span>{tool.default_risk}</span>
                                </div>
                            ))}
                        </div>
                    </section>
                </div>
            ) : null}


            {view === "screen" ? (
                <div className="desktop-agent-section">
                    <section className="desktop-agent-card">
                        <h3>Screen context</h3>
                        <p className="desktop-agent-muted">
                            Capture the current screen and ask Astra Vision what is visible, what looks wrong, or what to do next.
                        </p>
                        <textarea
                            className="desktop-agent-textarea"
                            value={screenQuestion}
                            onChange={(event) => setScreenQuestion(event.target.value)}
                            placeholder="What am I looking at right now?"
                            rows={4}
                        />
                        <label className="desktop-agent-checkbox-row">
                            <input
                                type="checkbox"
                                checked={captureFreshForAnalysis}
                                onChange={(event) => setCaptureFreshForAnalysis(event.target.checked)}
                            />
                            <span>Capture a fresh screenshot before analysis</span>
                        </label>
                        <div className="desktop-agent-inline-actions">
                            <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void handleAnalyzeScreen()}>
                                Analyze current screen
                            </Button>
                        </div>
                        {screenAnalysis ? (
                            <div className="desktop-agent-screen-result">
                                <p><strong>Model:</strong> {screenAnalysis.model}</p>
                                <p><strong>Capture:</strong> {screenAnalysis.image_path}</p>
                                <p><strong>Question:</strong> {screenAnalysis.question}</p>
                                <div className="desktop-agent-screen-answer">{screenAnalysis.answer}</div>
                                {screenAnalysis.semantic_frame ? (
                                    <div className="desktop-agent-screen-answer">
                                        <p><strong>Semantic frame:</strong> {screenAnalysis.semantic_frame.scene_summary}</p>
                                        <p><strong>Provider:</strong> {screenAnalysis.semantic_frame.page_evidence.content_provider_hint ?? "unknown"}</p>
                                        <p><strong>Page kind:</strong> {screenAnalysis.semantic_frame.page_evidence.page_kind_hint ?? "unknown"}</p>
                                        <p><strong>Visible results:</strong> {screenAnalysis.semantic_frame.visible_result_items?.length ?? 0}</p>
                                        <pre className="desktop-agent-json">{JSON.stringify(screenAnalysis.semantic_frame, null, 2)}</pre>
                                    </div>
                                ) : null}
                                {recentGoalLoop ? (
                                    <div className="desktop-agent-screen-answer">
                                        <p><strong>Recent goal loop:</strong> {recentGoalLoop.status}</p>
                                        <p><strong>Strategy:</strong> {recentGoalLoop.current_strategy ?? "none"}</p>
                                        <p><strong>Planner source:</strong> {lastPlannerDiagnostic?.source ?? "unknown"}</p>
                                        <p><strong>Last planner step:</strong> {recentGoalLoop.planner_steps?.length ?? 0}</p>
                                        <p><strong>Visible actionability:</strong> {lastActionability?.status ?? "unknown"}</p>
                                        <p><strong>Refinement strategy:</strong> {lastActionability?.refinement_strategy ?? "none"}</p>
                                        <p><strong>Grounding gaps:</strong> {actionabilityGaps ?? "none"}</p>
                                        <p><strong>Verification:</strong> {recentGoalLoop.verifier_status ?? "unknown"}</p>
                                        <pre className="desktop-agent-json">{JSON.stringify(recentGoalLoop, null, 2)}</pre>
                                    </div>
                                ) : null}
                            </div>
                        ) : null}
                    </section>
                </div>
            ) : null}

            {view === "approvals" ? (
                <div className="desktop-agent-section">
                    {approvals.length === 0 ? (
                        <div className="desktop-agent-empty">No pending approvals.</div>
                    ) : (
                        approvals.map((approval) => (
                            <article key={approval.action_id} className="desktop-agent-card desktop-agent-approval-card">
                                <div className="desktop-agent-approval-header">
                                    <div>
                                        <h3>{approval.tool_name}</h3>
                                        <p>{approval.reason}</p>
                                    </div>
                                    <span>{approval.risk_level}</span>
                                </div>
                                <pre className="desktop-agent-json">{JSON.stringify(approval.params, null, 2)}</pre>
                                <div className="desktop-agent-inline-actions">
                                    <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void handleApproval(approval.action_id, "approve")}>
                                        Approve
                                    </Button>
                                    <Button variant="text" radius="full" size="xs" disabled={isBusy} onClick={() => void handleApproval(approval.action_id, "reject")}>
                                        Reject
                                    </Button>
                                </div>
                            </article>
                        ))
                    )}
                </div>
            ) : null}

            {view === "audit" ? (
                <div className="desktop-agent-section">
                    {audit.length === 0 ? (
                        <div className="desktop-agent-empty">No audit events yet.</div>
                    ) : (
                        audit
                            .slice()
                            .reverse()
                            .map((event) => (
                                <article key={event.audit_id} className="desktop-agent-card desktop-agent-audit-card">
                                    <div className="desktop-agent-audit-header">
                                        <strong>{event.tool_name}</strong>
                                        <span className="border rounded-full px-2 text-sm border-gray-400 text-gray-500">{event.status}</span>
                                    </div>
                                    <p>{event.stage} · {new Date(event.timestamp).toLocaleString()}</p>
                                    <pre className="desktop-agent-json">{JSON.stringify(event.details, null, 2)}</pre>
                                </article>
                            ))
                    )}
                </div>
            ) : null}
        </aside>
    );
}
