import { useCallback, useEffect, useMemo, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { useMeeting } from "../hooks/useMeeting";
import type {
    ExportedMeeting,
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
    "meeting-session-updated",
    "meeting-transcript-updated",
    "meeting-artifacts-updated",
    "meeting-diagnostics-updated",
];

function statusLabel(session: MeetingSession | null, state: MeetingSessionState | null) {
    if (session) return "active";
    if (state) return "stopped";
    return "none";
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

function compactError(error: unknown) {
    return error instanceof Error ? error.message : String(error);
}

export function WorkSessionStatusStrip({ onOpenDetails }: WorkSessionStatusStripProps) {
    const meeting = useMeeting();
    const [activeSession, setActiveSession] = useState<MeetingSession | null>(null);
    const [activeState, setActiveState] = useState<MeetingSessionState | null>(null);
    const [lastCompletedState, setLastCompletedState] = useState<MeetingSessionState | null>(null);
    const [capabilities, setCapabilities] = useState<MeetingLiveCapabilitySnapshot | null>(null);
    const [busyAction, setBusyAction] = useState<WorkSessionAction | null>(null);
    const [chatCommandStatus, setChatCommandStatus] = useState<string | null>(null);
    const [lastError, setLastError] = useState<string | null>(null);

    const refresh = useCallback(async () => {
        try {
            setLastError(null);
            const [session, state, completedState, liveCapabilities] = await Promise.allSettled([
                meeting.getActiveSession(),
                meeting.getActiveState(),
                meeting.getLastCompletedState(),
                meeting.getLiveCapabilities(),
            ]);
            if (session.status === "fulfilled") setActiveSession(session.value);
            else setActiveSession(null);
            if (state.status === "fulfilled") setActiveState(state.value);
            else setActiveState(null);
            if (completedState.status === "fulfilled") setLastCompletedState(completedState.value);
            else setLastCompletedState(null);
            if (liveCapabilities.status === "fulfilled") setCapabilities(liveCapabilities.value);
            else setCapabilities(null);
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

        const describeIntent = (intent?: string) => {
            switch (intent) {
                case "stop_session":
                case "stop_and_generate_recap":
                    return "Stopping / draining STT";
                case "attach_screen_context":
                    return "Attaching screen context";
                case "generate_intelligence":
                case "generate_technical_recap":
                case "generate_follow_up_draft":
                    return "Generating recap";
                case "recall_session_memory":
                case "search_session_memory":
                    return "Reading session memory";
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
                    if (!cancelled) setChatCommandStatus(describeIntent(event.payload.intent));
                }
            );
            unlisteners.push(started);
            const finished = await listen<WorkSessionChatCommandEvent>(
                "work-session-chat-command-finished",
                () => {
                    if (!cancelled) {
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
            unlisteners.forEach((unlisten) => unlisten());
        };
    }, [refresh]);

    useEffect(() => {
        const intervalMs = activeSession ? 2000 : 6000;
        const timer = window.setInterval(() => {
            void refresh();
        }, intervalMs);
        return () => window.clearInterval(timer);
    }, [activeSession, refresh]);

    const displayState = activeSession ? activeState : lastCompletedState;
    const transcriptCount = displayState?.transcript.length ?? 0;
    const screenContextCount = displayState?.screen_contexts.length ?? 0;
    const written = sumWritten(capabilities);
    const transcribed = sumTranscribed(capabilities);
    const inFlight = sumInFlight(capabilities);
    const micActive = capabilities?.microphone_capture_health.active_handle_present ?? false;
    const systemActive = capabilities?.system_capture_health.active_handle_present ?? false;
    const controlsBusy = busyAction !== null || chatCommandStatus !== null;

    const sttLabel = useMemo(() => {
        if (written === 0 && inFlight === 0) return "idle";
        if (inFlight > 0) return `${transcribed}/${written} transcribed, ${inFlight} in-flight`;
        return `${transcribed}/${written} transcribed`;
    }, [inFlight, transcribed, written]);

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
            runAction("stop", async (): Promise<ExportedMeeting> => {
                return meeting.stopSession();
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

    return (
        <section className="work-session-strip" aria-label="Work Session status">
            <div className="work-session-strip__main">
                <span className={`work-session-strip__state work-session-strip__state--${statusLabel(activeSession, displayState)}`}>
                    Work Session: {statusLabel(activeSession, displayState)}
                </span>
                <span>Mic: {micActive ? "active" : "inactive"}</span>
                <span>System: {systemActive ? "active" : "inactive"}</span>
                <span>Transcript: {transcriptCount}</span>
                <span>STT: {sttLabel}</span>
                <span>Screen: {screenContextCount}</span>
                <span>Intel: {displayState?.intelligence ? "generated" : "none"}</span>
                {chatCommandStatus ? <span>{chatCommandStatus}</span> : null}
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
            </div>
            {lastError ? <div className="work-session-strip__error">{lastError}</div> : null}
        </section>
    );
}
