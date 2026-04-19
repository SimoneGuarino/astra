import { useCallback, useEffect, useMemo, useState } from "react";
import { useDesktopAgent } from "../hooks/useDesktopAgent";
import type {
    CapabilityManifest,
    DesktopAuditEvent,
    DesktopPolicySnapshot,
    PendingApproval,
    ScreenAnalysisResult,
    ScreenCaptureResult,
    ScreenObservationStatus,
    ToolDescriptor,
} from "../types/desktopAgent";
import { Button } from "../ui/buttons/Button";

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
    const [screenQuestion, setScreenQuestion] = useState("What am I looking at right now?");
    const [screenAnalysis, setScreenAnalysis] = useState<ScreenAnalysisResult | null>(null);
    const [captureFreshForAnalysis, setCaptureFreshForAnalysis] = useState(true);
    const [isBusy, setIsBusy] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        try {
            setError(null);
            const [nextTools, nextPolicy, nextApprovals, nextAudit, nextScreenStatus, nextCapabilities] = await Promise.all([
                agent.listTools(),
                agent.getPolicySnapshot(),
                agent.getPendingApprovals(),
                agent.getRecentAuditEvents(60),
                agent.getScreenObservationStatus(),
                agent.getCapabilityManifest(),
            ]);
            setTools(nextTools);
            setPolicy(nextPolicy);
            setApprovals(nextApprovals);
            setAudit(nextAudit);
            setScreenStatus(nextScreenStatus);
            setCapabilities(nextCapabilities);
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
        <aside className="desktop-agent-panel">
            <div className="desktop-agent-panel__header">
                <div>
                    <p className="desktop-agent-panel__kicker">DESKTOP AGENT</p>
                    <h2>Approval Center</h2>
                    <p className="desktop-agent-panel__subtitle">{approvalCountLabel}</p>
                </div>
                <div className="desktop-agent-panel__actions">
                    <Button variant="secondary" radius="full" size="xs" onClick={() => void refresh()}>
                        Refresh
                    </Button>
                    <Button variant="text" radius="full" size="xs" onClick={onClose}>
                        Close
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
                        <article className="desktop-agent-card">
                            <h3>Policy</h3>
                            <p>Allowed roots: {policy?.allowed_roots.length ?? 0}</p>
                            <p>Terminal allowlist: {policy?.terminal_allowed_commands.length ?? 0}</p>
                            <p>Browser enabled: {policy?.browser_enabled ? "yes" : "no"}</p>
                            <p>Desktop control: {policy?.desktop_control_enabled ? "yes" : "no"}</p>
                            <p>Pending approvals: {capabilities?.approvals.pending_count ?? 0}</p>
                        </article>
                        <article className="desktop-agent-card">
                            <h3>Screen</h3>
                            <p>Provider: {screenStatus?.provider ?? "unknown"}</p>
                            <p>Enabled: {screenStatus?.enabled ? "yes" : "no"}</p>
                            <p>Captures: {screenStatus?.capture_count ?? 0}</p>
                            <p>Vision: {capabilities?.screen.vision_model_name ?? (capabilities?.screen.analysis_available ? "available" : "unavailable")}</p>
                            <p className="desktop-agent-muted">{screenStatus?.note ?? "No status available"}</p>
                            <div className="desktop-agent-inline-actions">
                                <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void handleToggleObservation()}>
                                    {screenStatus?.enabled ? "Disable observation" : "Enable observation"}
                                </Button>
                                <Button variant="secondary" radius="full" size="xs" disabled={isBusy} onClick={() => void handleCapture()}>
                                    Capture screen
                                </Button>
                            </div>
                            {lastCapture ? (
                                <p className="desktop-agent-muted">Last capture: {lastCapture.image_path}</p>
                            ) : null}
                        </article>
                    </section>

                    <section className="desktop-agent-card">
                        <h3>Registered tools</h3>
                        <div className="desktop-agent-tools-list">
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
                                        <span>{event.status}</span>
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
