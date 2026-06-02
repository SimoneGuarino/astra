import type { AssistantThinkingTraceState } from "../types/assistant";

type ThinkingGovernancePanelProps = {
    trace?: AssistantThinkingTraceState | null;
};

type UnknownRecord = Record<string, unknown>;

const routeLabels: Record<string, string> = {
    direct_answer: "Direct answer",
    memory_grounded_answer: "Memory grounded",
    tool_arbitration_required: "Tool arbitration",
    deep_search_required: "Deep Search required",
    clarify_required: "Clarification required",
    refuse: "Safe refusal",
};

const reasonLabels: Record<string, string> = {
    not_needed: "Not needed",
    unknown_topic: "Unknown topic",
    current_information: "Current information",
    low_memory_coverage: "Low memory coverage",
    high_stakes: "High-stakes request",
    blocked_by_policy: "Blocked by policy",
};

const qualityGradeLabels: Record<string, string> = {
    excellent: "Excellent",
    good: "Good",
    needs_review: "Needs review",
    risky: "Risky",
};

const qualityStatusLabels: Record<string, string> = {
    pass: "Pass",
    observe: "Observe",
    review: "Review",
};

function asRecord(value: unknown): UnknownRecord | null {
    return value && typeof value === "object" && !Array.isArray(value) ? (value as UnknownRecord) : null;
}

function asBool(value: unknown): boolean | null {
    return typeof value === "boolean" ? value : null;
}

function asNumber(value: unknown): number | null {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string | null {
    return typeof value === "string" && value.trim() ? value : null;
}

function formatPercent(value: unknown): string {
    const number = asNumber(value);
    if (number === null) return "unknown";
    return `${Math.round(number * 100)}%`;
}

function formatBarWidth(value: unknown): string {
    const number = asNumber(value);
    if (number === null) return "0%";
    return `${Math.max(0, Math.min(100, Math.round(number * 100)))}%`;
}

function formatBool(value: unknown): string {
    const bool = asBool(value);
    if (bool === null) return "unknown";
    return bool ? "yes" : "no";
}

function formatDuration(value: unknown): string {
    const number = asNumber(value);
    if (number === null) return "unknown";
    if (number < 1000) return `${Math.round(number)} ms`;
    return `${(number / 1000).toFixed(1)} s`;
}

function labelFor(value: unknown, labels: Record<string, string>): string {
    const key = asString(value);
    if (!key) return "unknown";
    return labels[key] ?? key.replace(/_/g, " ");
}

function getArrayText(value: unknown): string[] {
    if (!Array.isArray(value)) return [];
    return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function getArrayRecord(value: unknown): UnknownRecord[] {
    if (!Array.isArray(value)) return [];
    return value.map(asRecord).filter((item): item is UnknownRecord => Boolean(item));
}

function severityClass(value: unknown): string {
    const severity = asString(value) ?? "info";
    return `thinking-governance-quality-finding--${severity.replace(/[^a-z0-9_-]/gi, "")}`;
}

function getMetadataRecord(trace: AssistantThinkingTraceState | null | undefined, key: string): UnknownRecord | null {
    const metadata = asRecord(trace?.metadata);
    return asRecord(metadata?.[key]);
}

export function ThinkingGovernancePanel({ trace }: ThinkingGovernancePanelProps) {
    const deepSearch = getMetadataRecord(trace, "deep_search");
    const toolDecision = getMetadataRecord(trace, "tool_decision");
    const memoryFeedback = getMetadataRecord(trace, "memory_feedback");
    const quality = getMetadataRecord(trace, "thinking_quality");
    const qualityFindings = getArrayRecord(quality?.findings);
    const qualitySafeguards = getArrayText(quality?.safeguards);
    const memory = getMetadataRecord(trace, "memory_assessment");
    const evidence = getMetadataRecord(trace, "evidence_assessment");
    const uncertainty = getMetadataRecord(trace, "uncertainty");
    const uncertaintyReasons = getArrayText(uncertainty?.reasons);

    if (!trace) {
        return (
            <section className="desktop-agent-section">
                <article className="desktop-agent-card thinking-governance-empty">
                    <h3>Thinking Governance</h3>
                    <p className="desktop-agent-muted">
                        Nessuna traccia Thinking disponibile. Invia una richiesta ad Astra per vedere route, confidence,
                        memoria, Deep Search, evidenze e uncertainty del prossimo cognitive pass.
                    </p>
                </article>
            </section>
        );
    }

    return (
        <section className="desktop-agent-section thinking-governance-panel">
            <article className="desktop-agent-card thinking-governance-hero">
                <div>
                    <p className="desktop-agent-panel__kicker">COGNITIVE GOVERNANCE</p>
                    <h3>{labelFor(trace.route, routeLabels)}</h3>
                    <p className="desktop-agent-muted">
                        {trace.intent_summary || "Intent summary non disponibile per l'ultimo ThinkingPlan."}
                    </p>
                </div>
                <div className="thinking-governance-score">
                    <span>{formatPercent(trace.confidence)}</span>
                    <small>confidence</small>
                </div>
            </article>

            <section className="desktop-agent-card-grid thinking-governance-grid">
                <article className="desktop-agent-card">
                    <h3>Planner</h3>
                    <div className="thinking-governance-kv">
                        <span>Request</span>
                        <strong>{trace.request_id ?? "unknown"}</strong>
                        <span>Source</span>
                        <strong>{trace.planner_source ?? "unknown"}</strong>
                        <span>Duration</span>
                        <strong>{formatDuration(trace.duration_ms)}</strong>
                        <span>Route</span>
                        <strong>{labelFor(trace.route, routeLabels)}</strong>
                    </div>
                </article>

                <article className="desktop-agent-card thinking-governance-quality-card">
                    <h3>Thinking quality</h3>
                    <div className="thinking-governance-kv">
                        <span>Score</span>
                        <strong>{formatPercent(quality?.score)}</strong>
                        <span>Grade</span>
                        <strong>{labelFor(quality?.grade, qualityGradeLabels)}</strong>
                        <span>Status</span>
                        <strong>{labelFor(quality?.status, qualityStatusLabels)}</strong>
                        <span>Raw CoT</span>
                        <strong>{formatBool(quality?.raw_chain_of_thought_included)}</strong>
                    </div>
                    <div className="thinking-governance-quality-bars" aria-label="Thinking quality dimensions">
                        {[
                            ["Route", quality?.route_consistency],
                            ["Evidence", quality?.evidence_alignment],
                            ["Memory", quality?.memory_alignment],
                            ["Tool safety", quality?.tool_safety_alignment],
                            ["Deep Search", quality?.deep_search_alignment],
                            ["Uncertainty", quality?.uncertainty_alignment],
                        ].map(([label, value]) => (
                            <div className="thinking-governance-quality-bar" key={String(label)}>
                                <span>{String(label)}</span>
                                <div>
                                    <i style={{ width: formatBarWidth(value) }} />
                                </div>
                                <strong>{formatPercent(value)}</strong>
                            </div>
                        ))}
                    </div>
                    {qualitySafeguards.length ? (
                        <p className="desktop-agent-muted">Safeguards: {qualitySafeguards.slice(0, 4).join(" · ")}</p>
                    ) : null}
                </article>

                <article className="desktop-agent-card">
                    <h3>Deep Search decision</h3>
                    <div className="thinking-governance-kv">
                        <span>Needed</span>
                        <strong>{formatBool(deepSearch?.needed)}</strong>
                        <span>Reason</span>
                        <strong>{labelFor(deepSearch?.reason, reasonLabels)}</strong>
                        <span>Query hint</span>
                        <strong>{asString(deepSearch?.query_hint) ?? "none"}</strong>
                    </div>
                </article>

                <article className="desktop-agent-card">
                    <h3>Tool arbitration</h3>
                    <div className="thinking-governance-kv">
                        <span>Required</span>
                        <strong>{formatBool(toolDecision?.tool_required)}</strong>
                        <span>Candidate</span>
                        <strong>{asString(toolDecision?.candidate_tool) ?? "none"}</strong>
                        <span>Reason</span>
                        <strong>{asString(toolDecision?.reason) ?? "none"}</strong>
                    </div>
                </article>

                <article className="desktop-agent-card">
                    <h3>Memory coverage</h3>
                    <div className="thinking-governance-kv">
                        <span>Relevant</span>
                        <strong>{formatBool(memory?.relevant)}</strong>
                        <span>Coverage</span>
                        <strong>{formatPercent(memory?.coverage)}</strong>
                        <span>Nodes</span>
                        <strong>{asNumber(memory?.node_count)?.toString() ?? "0"}</strong>
                    </div>
                    {getArrayText(memory?.missing_information).length ? (
                        <ul className="thinking-governance-list">
                            {getArrayText(memory?.missing_information).slice(0, 4).map((item) => (
                                <li key={item}>{item}</li>
                            ))}
                        </ul>
                    ) : null}
                </article>

                <article className="desktop-agent-card thinking-governance-memory-feedback">
                    <h3>Memory feedback loop</h3>
                    <div className="thinking-governance-kv">
                        <span>Enabled</span>
                        <strong>{formatBool(memoryFeedback?.enabled)}</strong>
                        <span>Planned</span>
                        <strong>{formatBool(memoryFeedback?.planned)}</strong>
                        <span>Review required</span>
                        <strong>{formatBool(memoryFeedback?.review_required)}</strong>
                        <span>Auto promote</span>
                        <strong>{formatBool(memoryFeedback?.auto_promote)}</strong>
                        <span>Min score</span>
                        <strong>{formatPercent(memoryFeedback?.min_score)}</strong>
                    </div>
                    <p className="desktop-agent-muted">
                        The feedback loop stores only review-gated learning candidates. Canonical memory promotion remains governed.
                    </p>
                </article>

                <article className="desktop-agent-card">
                    <h3>Evidence</h3>
                    <div className="thinking-governance-kv">
                        <span>Local evidence</span>
                        <strong>{formatBool(evidence?.has_local_evidence)}</strong>
                        <span>Session evidence</span>
                        <strong>{formatBool(evidence?.has_current_session_evidence)}</strong>
                        <span>Current info</span>
                        <strong>{formatBool(evidence?.requires_current_information)}</strong>
                        <span>External sources</span>
                        <strong>{formatBool(evidence?.requires_external_sources)}</strong>
                    </div>
                    {asString(evidence?.evidence_summary) ? (
                        <p className="desktop-agent-muted">{asString(evidence?.evidence_summary)}</p>
                    ) : null}
                </article>

                <article className="desktop-agent-card">
                    <h3>Uncertainty</h3>
                    <div className="thinking-governance-kv">
                        <span>Level</span>
                        <strong>{labelFor(uncertainty?.level, {})}</strong>
                        <span>Reasons</span>
                        <strong>{uncertaintyReasons.length}</strong>
                    </div>
                    {uncertaintyReasons.length ? (
                        <ul className="thinking-governance-list">
                            {uncertaintyReasons.slice(0, 5).map((reason) => (
                                <li key={reason}>{reason}</li>
                            ))}
                        </ul>
                    ) : null}
                </article>

                <article className="desktop-agent-card thinking-governance-quality-findings">
                    <h3>Quality findings</h3>
                    {qualityFindings.length ? (
                        <ul className="thinking-governance-list">
                            {qualityFindings.slice(0, 6).map((finding) => (
                                <li className={severityClass(finding.severity)} key={asString(finding.code) ?? asString(finding.message) ?? JSON.stringify(finding)}>
                                    <strong>{asString(finding.code) ?? "quality_finding"}</strong>
                                    <span>{asString(finding.message) ?? "Finding details unavailable."}</span>
                                    <em>{asString(finding.recommendation) ?? "No recommendation available."}</em>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p className="desktop-agent-muted">No quality findings for the latest ThinkingPlan.</p>
                    )}
                </article>

                <article className="desktop-agent-card">
                    <h3>Warnings</h3>
                    {trace.warnings?.length ? (
                        <ul className="thinking-governance-list">
                            {trace.warnings.slice(0, 6).map((warning) => (
                                <li key={warning}>{warning}</li>
                            ))}
                        </ul>
                    ) : (
                        <p className="desktop-agent-muted">No warnings for the latest ThinkingPlan.</p>
                    )}
                </article>
            </section>

            <article className="desktop-agent-card">
                <h3>User-visible reasoning trace</h3>
                {trace.steps.length ? (
                    <div className="thinking-governance-timeline">
                        {trace.steps.map((step, index) => (
                            <div className="thinking-governance-timeline-step" key={`${step.phase}-${index}`}>
                                <span>{index + 1}</span>
                                <div>
                                    <strong>{step.title}</strong>
                                    {step.detail ? <p>{step.detail}</p> : null}
                                    {typeof step.confidence === "number" ? <small>{formatPercent(step.confidence)}</small> : null}
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <p className="desktop-agent-muted">No user-visible Thinking steps were emitted.</p>
                )}
            </article>
        </section>
    );
}
