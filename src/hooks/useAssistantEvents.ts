import { useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import type {
    AssistantErrorEvent,
    AssistantInterruptedEvent,
    AssistantRequestFinishedEvent,
    AssistantRequestSettledEvent,
    AssistantRequestStartedEvent,
    AssistantStatus,
    RequestMetricsSnapshot,
    SpeechSegmentQueuedEvent,
    StreamChunkEvent,
    VoiceSessionStateEvent,
    VoiceSessionTranscriptEvent,
    VoiceTurnMetricsSnapshot,
} from "../types/assistant";
import type {
    AudioSegmentFailedEvent,
    AudioSegmentReadyEvent,
} from "./useAssistantAudio";
import type { ConversationRouteDiagnostic } from "../types/desktopAgent";

type UseAssistantEventsParams = {
    onRequestStarted: (event: AssistantRequestStartedEvent) => void;
    onStreamChunk: (event: StreamChunkEvent) => void;
    onAudioReady: (event: AudioSegmentReadyEvent) => void;
    onAudioFailed: (event: AudioSegmentFailedEvent) => void;
    onSpeechQueued: (event: SpeechSegmentQueuedEvent) => void;
    onRequestFinished: (event: AssistantRequestFinishedEvent) => void;
    onRequestSettled?: (event: AssistantRequestSettledEvent) => void;
    onAssistantError: (event: AssistantErrorEvent) => void;
    onStatus: (status: AssistantStatus) => void;
    onModel: (model: string) => void;
    onMetrics: (metrics: RequestMetricsSnapshot) => void;
    onAssistantInterrupted: (event: AssistantInterruptedEvent) => void;
    onVoiceSessionState: (event: VoiceSessionStateEvent) => void;
    onVoiceSessionTranscript: (event: VoiceSessionTranscriptEvent) => void;
    onVoiceTurnMetrics: (metrics: VoiceTurnMetricsSnapshot) => void;
    onRouteDiagnostic?: (diagnostic: ConversationRouteDiagnostic) => void;
};

export function useAssistantEvents({
    onRequestStarted,
    onStreamChunk,
    onAudioReady,
    onAudioFailed,
    onSpeechQueued,
    onRequestFinished,
    onRequestSettled,
    onAssistantError,
    onStatus,
    onModel,
    onMetrics,
    onAssistantInterrupted,
    onVoiceSessionState,
    onVoiceSessionTranscript,
    onVoiceTurnMetrics,
    onRouteDiagnostic,
}: UseAssistantEventsParams) {
    useEffect(() => {
        let cleanupFns: Array<() => void> = [];
        let disposed = false;

        (async () => {
            try {
                const unlistenRequestStarted = await listen<AssistantRequestStartedEvent>(
                    "assistant-request-started",
                    (event) => {
                        if (!disposed) onRequestStarted(event.payload);
                    }
                );
                const unlistenChunk = await listen<StreamChunkEvent>(
                    "assistant-stream-chunk",
                    (event) => {
                        if (!disposed) onStreamChunk(event.payload);
                    }
                );
                const unlistenAudioReady = await listen<AudioSegmentReadyEvent>(
                    "assistant-audio-segment-ready",
                    (event) => {
                        if (!disposed) onAudioReady(event.payload);
                    }
                );
                const unlistenAudioFailed = await listen<AudioSegmentFailedEvent>(
                    "assistant-audio-segment-failed",
                    (event) => {
                        if (!disposed) onAudioFailed(event.payload);
                    }
                );
                const unlistenSpeechQueued = await listen<SpeechSegmentQueuedEvent>(
                    "assistant-speech-segment-queued",
                    (event) => {
                        if (!disposed) onSpeechQueued(event.payload);
                    }
                );
                const unlistenRequestFinished = await listen<AssistantRequestFinishedEvent>(
                    "assistant-request-finished",
                    (event) => {
                        if (!disposed) onRequestFinished(event.payload);
                    }
                );
                const unlistenRequestSettled = await listen<AssistantRequestSettledEvent>(
                    "assistant-request-settled",
                    (event) => {
                        if (!disposed) onRequestSettled?.(event.payload);
                    }
                );
                const unlistenError = await listen<AssistantErrorEvent>(
                    "assistant-error",
                    (event) => {
                        if (!disposed) onAssistantError(event.payload);
                    }
                );
                const unlistenStatus = await listen<string>("assistant-status", (event) => {
                    if (!disposed) onStatus(event.payload as AssistantStatus);
                });
                const unlistenModel = await listen<string>("assistant-model", (event) => {
                    if (!disposed) onModel(event.payload);
                });
                const unlistenMetrics = await listen<RequestMetricsSnapshot>(
                    "assistant-metrics-updated",
                    (event) => {
                        if (!disposed) onMetrics(event.payload);
                    }
                );
                const unlistenInterrupted = await listen<AssistantInterruptedEvent>(
                    "assistant-interrupted",
                    (event) => {
                        if (!disposed) onAssistantInterrupted(event.payload);
                    }
                );
                const unlistenVoiceState = await listen<VoiceSessionStateEvent>(
                    "voice-session-state-changed",
                    (event) => {
                        if (!disposed) onVoiceSessionState(event.payload);
                    }
                );
                const unlistenVoiceTranscript = await listen<VoiceSessionTranscriptEvent>(
                    "voice-session-transcript",
                    (event) => {
                        if (!disposed) onVoiceSessionTranscript(event.payload);
                    }
                );
                const unlistenVoiceMetrics = await listen<VoiceTurnMetricsSnapshot>(
                    "voice-turn-metrics-updated",
                    (event) => {
                        if (!disposed) onVoiceTurnMetrics(event.payload);
                    }
                );
                const unlistenRouteDiagnostic = await listen<ConversationRouteDiagnostic>(
                    "assistant-route-diagnostic",
                    (event) => {
                        if (!disposed) onRouteDiagnostic?.(event.payload);
                    }
                );

                cleanupFns = [
                    unlistenRequestStarted,
                    unlistenChunk,
                    unlistenAudioReady,
                    unlistenAudioFailed,
                    unlistenSpeechQueued,
                    unlistenRequestFinished,
                    unlistenRequestSettled,
                    unlistenError,
                    unlistenStatus,
                    unlistenModel,
                    unlistenMetrics,
                    unlistenInterrupted,
                    unlistenVoiceState,
                    unlistenVoiceTranscript,
                    unlistenVoiceMetrics,
                    unlistenRouteDiagnostic,
                ];
            } catch (error) {
                console.error("Listener registration failed:", error);
            }
        })();

        return () => {
            disposed = true;
            cleanupFns.forEach((fn) => fn());
        };
    }, [
        onAssistantError,
        onAssistantInterrupted,
        onAudioFailed,
        onAudioReady,
        onMetrics,
        onModel,
        onRequestFinished,
        onRequestSettled,
        onRequestStarted,
        onSpeechQueued,
        onStatus,
        onStreamChunk,
        onVoiceSessionState,
        onVoiceSessionTranscript,
        onVoiceTurnMetrics,
        onRouteDiagnostic,
    ]);
}
