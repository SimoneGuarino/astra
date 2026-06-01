import { useCallback, useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useMeeting } from "../hooks/useMeeting";
import type {
    MeetingFinalizationStatus,
    MeetingIntelligenceResult,
    MeetingLiveCapabilitySnapshot,
    MeetingSession,
    MeetingSessionState,
} from "../types/meeting";
import { Button } from "../ui/buttons/Button";

type WorkSessionStatusStripProps = {
    onOpenDetails: () => void;
};

type WorkSessionAction = "stop" | "attach_screen" | "generate_recap";
type WorkSessionChatCommandEvent = {
    intent?: string;
};

const meetingEvents = [
    "meeting-finalization-updated",
    "meeting-session-updated",
    "meeting-transcript-updated",
    "meeting-artifacts-updated",
    "meeting-diagnostics-updated",
];

function statusKind(status: MeetingSessionState["status"] | MeetingSession["status"] | null | undefined) {
    if (!status) return "unknown";
    if (typeof status === "string") return status;
    if ("failed" in status) return "failed";
    return "error";
}

function statusLabel(session: MeetingSession | null, state: MeetingSessionState | null) {
    const kind = statusKind(state?.status ?? session?.status ?? null);
    if (session && kind === "failed" && session.capture_active === false) return "failed/recoverable";
    if (session && kind === "failed") return "failed";
    if (session) return "active";
    if (state) return "stopped";
    return "none";
}

function statusClassName(label: string) {
    return label.replace("/", "_");
}

function sumWritten(capabilities: MeetingLiveCapabilitySnapshot | null) {
    if (!capabilities) return 0;
    return (
        (capabilities.system_capture_health.metrics.segments_written ?? 0) +
        (capabilities.microphone_capture_health.metrics.segments_written ?? 0)
    );
}

function sumTranscribed(capabilities: MeetingLiveCapabilitySnapshot | null) {
    if (!capabilities) return 0;
    return (
        (capabilities.system_capture_health.metrics.segments_transcribed ?? 0) +
        (capabilities.microphone_capture_health.metrics.segments_transcribed ?? 0)
    );
}

function sumInFlight(capabilities: MeetingLiveCapabilitySnapshot | null) {
    if (!capabilities) return 0;
    return (
        (capabilities.system_capture_health.metrics.segments_in_flight ?? 0) +
        (capabilities.microphone_capture_health.metrics.segments_in_flight ?? 0)
    );
}

function sumQueueDepth(capabilities: MeetingLiveCapabilitySnapshot | null) {
    if (!capabilities) return 0;
    return (
        (capabilities.system_capture_health.metrics.current_queue_depth ?? 0) +
        (capabilities.microphone_capture_health.metrics.current_queue_depth ?? 0)
    );
}

function sumSttTimeouts(capabilities: MeetingLiveCapabilitySnapshot | null) {
    if (!capabilities) return 0;
    return (
        (capabilities.system_capture_health.metrics.segment_transcription_timeouts ?? 0) +
        (capabilities.microphone_capture_health.metrics.segment_transcription_timeouts ?? 0)
    );
}

function maxConsecutiveSttFailures(capabilities: MeetingLiveCapabilitySnapshot | null) {
    if (!capabilities) return 0;
    return Math.max(
        capabilities.system_capture_health.metrics.segment_transcription_failures_consecutive ?? 0,
        capabilities.microphone_capture_health.metrics.segment_transcription_failures_consecutive ?? 0
    );
}

function maxSttFailureThreshold(capabilities: MeetingLiveCapabilitySnapshot | null) {
    if (!capabilities) return 1;
    return Math.max(
        1,
        capabilities.system_capture_health.pipeline.max_consecutive_transcription_failures ?? 1,
        capabilities.microphone_capture_health.pipeline.max_consecutive_transcription_failures ?? 1
    );
}

function compactError(error: unknown) {
    return error instanceof Error ? error.message : String(error);
}

function isFinalizing(status: MeetingFinalizationStatus | null) {
    return Boolean(status && ![
        "idle",
        "completed",
        "completed_partial",
        "failed_recoverable",
        "failed",
    ].includes(status.stage));
}

function finalizationLabel(status: MeetingFinalizationStatus | null) {
    if (!status || status.stage === "idle") return null;
    if (status.stage === "completed") return "Finalized";
    if (status.stage === "completed_partial") return "Finalized with partial transcript";
    if (status.stage === "failed_recoverable") return "Finalization failed: recoverable";
    if (status.stage === "failed") return "Finalization failed";
    if (status.stage === "draining_stt") {
        return `Finalizing: draining STT ${status.transcribed_segments}/${status.written_segments}`;
    }
    return `Finalizing: ${status.progress_label || status.stage.replace(/_/g, " ")}`;
}

export function WorkSessionStatusStrip({ onOpenDetails }: WorkSessionStatusStripProps) {
    const meeting = useMeeting();
    const [activeSession, setActiveSession] = useState<MeetingSession | null>(null);
    const [activeState, setActiveState] = useState<MeetingSessionState | null>(null);
    const [lastCompletedState, setLastCompletedState] = useState<MeetingSessionState | null>(null);
    const [capabilities, setCapabilities] = useState<MeetingLiveCapabilitySnapshot | null>(null);
    const [finalizationStatus, setFinalizationStatus] = useState<MeetingFinalizationStatus | null>(null);
    const [busyAction, setBusyAction] = useState<WorkSessionAction | null>(null);
    const [chatCommandStatus, setChatCommandStatus] = useState<string | null>(null);
    const [lastError, setLastError] = useState<string | null>(null);
    const [isMinimized, setIsMinimized] = useState(false);
    const [isDismissed, setIsDismissed] = useState(false);

    const refresh = useCallback(async () => {
        try {
            setLastError(null);
            const [session, state, completedState, liveCapabilities, finalization] = await Promise.allSettled([
                meeting.getActiveSession(),
                meeting.getActiveState(),
                meeting.getLastCompletedState(),
                meeting.getLiveCapabilities(),
                meeting.readFinalizationStatus(),
            ]);
            if (session.status === "fulfilled") setActiveSession(session.value);
            else setActiveSession(null);
            if (state.status === "fulfilled") setActiveState(state.value);
            else setActiveState(null);
            if (completedState.status === "fulfilled") setLastCompletedState(completedState.value);
            else setLastCompletedState(null);
            if (liveCapabilities.status === "fulfilled") {
                setCapabilities(liveCapabilities.value);
                setFinalizationStatus(liveCapabilities.value.finalization_status);
            } else {
                setCapabilities(null);
            }
            if (finalization.status === "fulfilled") setFinalizationStatus(finalization.value);
        } catch (error) {
            setLastError(compactError(error));
        }
    }, [meeting]);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    useEffect(() => {
        let cancelled = false;
        const unlisteners: Array<() => void> = [];

        const subscribe = async () => {
            for (const eventName of meetingEvents) {
                const unlisten = await listen(eventName, () => {
                    if (!cancelled) void refresh();
                });
                unlisteners.push(unlisten);
            }
        };

        void subscribe().catch((error) => {
            if (!cancelled) setLastError(compactError(error));
        });

        return () => {
            cancelled = true;
            unlisteners.forEach((unlisten) => unlisten());
        };
    }, [refresh]);

    useEffect(() => {
        let cancelled = false;
        const unlisteners: Array<() => void> = [];
        let routeTimer: ReturnType<typeof setTimeout> | null = null;

        const describeIntent = (intent?: string) => {
            switch (intent) {
                case "route_request":
                    return "Routing request";
                case "router_failed":
                    return "Router failed";
                case "stop_session":
                case "stop_and_generate_recap":
                    return "Stopping / draining STT";
                case "attach_screen_context":
                    return "Attaching screen context";
                case "generate_transcript_summary":
                    return "Analyzing transcript";
                case "generate_intelligence":
                case "generate_technical_recap":
                case "generate_follow_up_draft":
                    return "Generating recap";
                case "generate_details":
                    return "Loading session details";
                case "recall_session_memory":
                case "search_session_memory":
                    return "Reading session memory";
                case "show_evidence":
                    return "Showing evidence";
                case "start_session":
                    return "Starting session";
                default:
                    return "Working";
            }
        };

        const subscribe = async () => {
            const started = await listen<WorkSessionChatCommandEvent>(
                "work-session-chat-command-started",
                (event) => {
                    if (routeTimer) {
                        clearTimeout(routeTimer);
                        routeTimer = null;
                    }
                    if (!cancelled) {
                        setChatCommandStatus(describeIntent(event.payload.intent));
                        if (event.payload.intent === "route_request") {
                            routeTimer = setTimeout(() => {
                                if (!cancelled) setChatCommandStatus("Routing with local model...");
                            }, 2000);
                        }
                    }
                }
            );
            unlisteners.push(started);
            const finished = await listen<WorkSessionChatCommandEvent>(
                "work-session-chat-command-finished",
                () => {
                    if (!cancelled) {
                        if (routeTimer) {
                            clearTimeout(routeTimer);
                            routeTimer = null;
                        }
                        setChatCommandStatus(null);
                        void refresh();
                    }
                }
            );
            unlisteners.push(finished);
        };

        void subscribe().catch((error) => {
            if (!cancelled) setLastError(compactError(error));
        });

        return () => {
            cancelled = true;
            if (routeTimer) clearTimeout(routeTimer);
            unlisteners.forEach((unlisten) => unlisten());
        };
    }, [refresh]);

    useEffect(() => {
        const intervalMs = activeSession || busyAction || chatCommandStatus || isFinalizing(finalizationStatus) ? 2000 : 6000;
        const timer = window.setInterval(() => {
            void refresh();
        }, intervalMs);
        return () => window.clearInterval(timer);
    }, [activeSession, busyAction, chatCommandStatus, finalizationStatus, refresh]);

    useEffect(() => {
        if (activeSession || busyAction || chatCommandStatus || isFinalizing(finalizationStatus)) {
            setIsDismissed(false);
        }
    }, [activeSession, busyAction, chatCommandStatus, finalizationStatus]);

    const displayState = activeSession ? activeState : lastCompletedState;
    const transcriptCount = displayState?.transcript.length ?? 0;
    const screenContextCount = displayState?.screen_contexts.length ?? 0;
    const written = sumWritten(capabilities);
    const transcribed = sumTranscribed(capabilities);
    const inFlight = sumInFlight(capabilities);
    const queueDepth = sumQueueDepth(capabilities);
    const sttTimeouts = sumSttTimeouts(capabilities);
    const sttConsecutiveFailures = maxConsecutiveSttFailures(capabilities);
    const sttFailureThreshold = maxSttFailureThreshold(capabilities);
    const micActive = capabilities?.microphone_capture_health.active_handle_present ?? false;
    const systemActive = capabilities?.system_capture_health.active_handle_present ?? false;
    const controlsBusy = busyAction !== null || chatCommandStatus !== null;
    const finalizing = isFinalizing(finalizationStatus);
    const finalizationProgressLabel = finalizationLabel(finalizationStatus);
    const currentStatusLabel = statusLabel(activeSession, displayState);
    const currentStatusClass = statusClassName(currentStatusLabel);
    const captureSummaryStatus = capabilities?.capture_summary_status ?? null;
    const captureFailed =
        captureSummaryStatus === "failed" ||
        (
            captureSummaryStatus === null &&
            (
                Boolean(activeSession && statusKind(displayState?.status ?? activeSession.status) === "failed") ||
                capabilities?.capture_health.state === "failed" ||
                capabilities?.capture_health.status === "failed"
            )
        );
    const captureDegraded = captureSummaryStatus === "degraded";
    const captureStatusLabel = captureDegraded ? "degraded" : captureFailed ? "failed" : null;
    const progressLabel =
        chatCommandStatus ??
        finalizationProgressLabel ??
        (busyAction === "stop"
            ? "Stopping / draining STT"
            : busyAction === "attach_screen"
              ? "Attaching screen context"
              : busyAction === "generate_recap"
                ? "Generating recap"
                : null);
    const hasVisibleSession = Boolean(
        activeSession ||
        displayState ||
        controlsBusy ||
        (finalizationStatus && finalizationStatus.stage !== "idle") ||
        lastError
    );
    const canDismiss = !activeSession && !controlsBusy && !finalizing && Boolean(displayState);

    const sttLabel = useMemo(() => {
        const pending = Math.max(0, written - transcribed);
        if (written === 0 && inFlight === 0 && queueDepth === 0) return "idle";
        if (queueDepth > 0 || inFlight > 0 || pending > 0) {
            const details = [`${transcribed}/${written} transcribed`];
            if (queueDepth > 0) details.push(`${queueDepth} queued`);
            if (inFlight > 0) details.push(`${inFlight} in-flight`);
            if (sttTimeouts > 0) details.push(`${sttTimeouts} timeout${sttTimeouts === 1 ? "" : "s"}`);
            const sttStalled =
                captureFailed ||
                (sttConsecutiveFailures >= sttFailureThreshold && sttTimeouts > 0);
            if (sttStalled) return `stalled: ${details.join(", ")}`;
            if (sttTimeouts > 0) return `delayed: ${details.join(", ")}`;
            return `catching up: ${details.join(", ")}`;
        }
        return `${transcribed}/${written} transcribed`;
    }, [captureFailed, inFlight, queueDepth, sttConsecutiveFailures, sttFailureThreshold, sttTimeouts, transcribed, written]);

    const runAction = useCallback(
        async (action: WorkSessionAction, operation: () => Promise<unknown>) => {
            try {
                setBusyAction(action);
                setLastError(null);
                await operation();
                await refresh();
            } catch (error) {
                setLastError(compactError(error));
            } finally {
                setBusyAction(null);
            }
        },
        [refresh]
    );

    const stopSession = useCallback(
        () =>
            runAction("stop", async (): Promise<MeetingFinalizationStatus> => {
                return meeting.requestStopSession();
            }),
        [meeting, runAction]
    );

    const attachScreen = useCallback(
        () =>
            runAction("attach_screen", async () => {
                return meeting.attachCurrentScreen({ store_screenshot: false, capture_fresh: true });
            }),
        [meeting, runAction]
    );

    const generateRecap = useCallback(
        () =>
            runAction("generate_recap", async (): Promise<MeetingIntelligenceResult> => {
                return meeting.generateIntelligence({ use_local_llm: true, max_transcript_segments: 80 });
            }),
        [meeting, runAction]
    );

    if ((!hasVisibleSession || isDismissed) && !activeSession && !controlsBusy) {
        return null;
    }

    if (isMinimized) {
        return (
            <section className="work-session-strip work-session-strip--minimized" aria-label="Work Session status">
                <div className="work-session-strip__main">
                    <span className={`work-session-strip__state work-session-strip__state--${currentStatusClass}`}>
                        Work Session: {currentStatusLabel}
                    </span>
                    {captureStatusLabel ? <span>Capture: {captureStatusLabel}</span> : null}
                    <span>Transcript: {transcriptCount}</span>
                    <span>STT: {sttLabel}</span>
                    {progressLabel ? <span>{progressLabel}</span> : null}
                </div>
                <div className="work-session-strip__actions">
                    <Button variant="text" radius="full" size="xs" onClick={() => setIsMinimized(false)}>
                        Expand
                    </Button>
                    <Button variant="text" radius="full" size="xs" onClick={onOpenDetails}>
                        Details
                    </Button>
                    {canDismiss ? (
                        <Button variant="text" radius="full" size="xs" onClick={() => setIsDismissed(true)}>
                            Dismiss
                        </Button>
                    ) : null}
                </div>
                {lastError ? <div className="work-session-strip__error">{lastError}</div> : null}
            </section>
        );
    }

    return (
        <section className="work-session-strip" aria-label="Work Session status">
            <div className="work-session-strip__main">
                <span className={`work-session-strip__state work-session-strip__state--${currentStatusClass}`}>
                    Work Session: {currentStatusLabel}
                </span>
                {captureStatusLabel ? <span>Capture: {captureStatusLabel}</span> : null}
                <span>Mic: {micActive ? "active" : "inactive"}</span>
                <span>System: {systemActive ? "active" : "inactive"}</span>
                <span>Transcript: {transcriptCount}</span>
                <span>STT: {sttLabel}</span>
                <span>Screen: {screenContextCount}</span>
                <span>Intel: {displayState?.intelligence ? "generated" : "none"}</span>
                {progressLabel ? <span>{progressLabel}</span> : null}
            </div>
            <div className="work-session-strip__actions">
                {activeSession ? (
                    <Button
                        variant="text"
                        radius="full"
                        size="xs"
                        disabled={controlsBusy}
                        onClick={() => void stopSession()}
                    >
                        {busyAction === "stop" ? "Stopping" : "Stop"}
                    </Button>
                ) : null}
                <Button
                    variant="text"
                    radius="full"
                    size="xs"
                    disabled={!activeSession || controlsBusy}
                    onClick={() => void attachScreen()}
                >
                    {busyAction === "attach_screen" ? "Attaching" : "Attach screen"}
                </Button>
                <Button
                    variant="text"
                    radius="full"
                    size="xs"
                    disabled={!displayState || controlsBusy}
                    onClick={() => void generateRecap()}
                >
                    {busyAction === "generate_recap" ? "Generating" : "Recap"}
                </Button>
                <Button variant="text" radius="full" size="xs" onClick={onOpenDetails}>
                    Details
                </Button>
                <Button variant="text" radius="full" size="xs" onClick={() => setIsMinimized(true)}>
                    Minimize
                </Button>
                {canDismiss ? (
                    <Button variant="text" radius="full" size="xs" onClick={() => setIsDismissed(true)}>
                        Dismiss
                    </Button>
                ) : null}
            </div>
            {lastError ? <div className="work-session-strip__error">{lastError}</div> : null}
        </section>
    );
}
