import type { AssistantActivityState } from "../types/assistant";

type AssistantThinkingTraceProps = {
    activity?: AssistantActivityState | null;
    isExpanded: boolean;
    onToggleExpanded: () => void;
};

const FORBIDDEN_THINKING_MARKERS = [
    "chain-of-thought",
    "chain of thought",
    "hidden reasoning",
    "private reasoning",
    "scratchpad",
    "internal monologue",
    "raw reasoning",
    "ragionamento nascosto",
    "pensiero nascosto",
];

function safeThinkingText(value: string | null | undefined, fallback: string): string {
    const text = String(value ?? "").trim();
    if (!text) return fallback;
    const lower = text.toLowerCase();
    if (FORBIDDEN_THINKING_MARKERS.some((marker) => lower.includes(marker))) {
        return "Traccia sintetica governata disponibile; ragionamento interno non esposto.";
    }
    return text;
}

export function AssistantThinkingTrace({
    activity,
    isExpanded,
    onToggleExpanded,
}: AssistantThinkingTraceProps) {
    const current = activity?.current;
    const trace = activity?.thinkingTrace;
    const traceSteps = trace?.steps ?? [];
    const legacySteps = activity?.steps ?? [];
    const visibleSteps = traceSteps.length
        ? traceSteps
        : legacySteps.map((step) => ({
              phase: step.stage,
              title: safeThinkingText(step.title, "Thinking step"),
              detail: step.detail ? safeThinkingText(step.detail, "Dettaglio Thinking non disponibile.") : null,
              confidence: null,
          }));
    const lastTraceStep = traceSteps.length ? traceSteps[traceSteps.length - 1] : null;
    const currentTitle = safeThinkingText(lastTraceStep?.title ?? current?.title, "Avvio Thinking");
    const currentDetail = safeThinkingText(
        lastTraceStep?.detail ?? current?.detail,
        "Astra sta valutando intento, memoria, strumenti, incertezza e ricerca governata."
    );
    const showExpand = visibleSteps.length > 0;

    return (
        <div className="assistant-thinking-card assistant-thinking-trace-card">
            <div className="assistant-thinking-rail" aria-hidden="true">
                <span className="assistant-thinking-dot" />
                <span className="assistant-thinking-line" />
            </div>
            <div className="assistant-thinking-main">
                <div className="assistant-thinking-header-row">
                    <div>
                        <div className="assistant-thinking-eyebrow">Thinking</div>
                        <div className="assistant-thinking-title">{currentTitle}</div>
                    </div>
                    {typeof trace?.confidence === "number" ? (
                        <span className="assistant-thinking-confidence">
                            {Math.round(trace.confidence * 100)}%
                        </span>
                    ) : null}
                </div>
                <div className="assistant-thinking-detail">{currentDetail}</div>
                {trace?.intent_summary ? (
                    <div className="assistant-thinking-intent">{safeThinkingText(trace.intent_summary, "Intent summary non disponibile.")}</div>
                ) : null}

                {showExpand ? (
                    <button type="button" className="assistant-thinking-expand" onClick={onToggleExpanded}>
                        {isExpanded ? "Nascondi traccia" : `Mostra traccia (${visibleSteps.length})`}
                    </button>
                ) : null}

                {isExpanded ? (
                    <div className="assistant-thinking-details-panel">
                        {visibleSteps.map((step, index) => (
                            <div className="assistant-thinking-step" key={`${step.phase}-${index}`}>
                                <span className="assistant-thinking-step-index">{index + 1}</span>
                                <div>
                                    <div className="assistant-thinking-step-title">{safeThinkingText(step.title, "Thinking step")}</div>
                                    {step.detail ? (
                                        <div className="assistant-thinking-step-detail">{safeThinkingText(step.detail, "Dettaglio Thinking non disponibile.")}</div>
                                    ) : null}
                                </div>
                            </div>
                        ))}
                        {trace?.warnings?.length ? (
                            <div className="assistant-thinking-warning">
                                {trace.warnings.slice(0, 3).join(" · ")}
                            </div>
                        ) : null}
                    </div>
                ) : null}
            </div>
        </div>
    );
}
