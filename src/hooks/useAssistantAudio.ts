import { convertFileSrc } from "@tauri-apps/api/core";
import { useCallback, useRef } from "react";

export type AudioSegmentReadyEvent = {
    request_id: string;
    segment_id: string;
    sequence: number;
    output_path: string;
    text: string;
};

export type AudioSegmentFailedEvent = {
    request_id: string;
    segment_id: string;
    sequence: number;
    message: string;
};

type UseAssistantAudioParams = {
    onSpeakingStart: () => void;
    onSpeakingEnd: () => void;
    onQueueIdle?: () => void;
    onSegmentPlaybackStart?: (segment: AudioSegmentReadyEvent) => void;
    onSegmentPlaybackComplete?: (segment: AudioSegmentReadyEvent) => void;
    onSessionPlaybackIdle?: (requestId: string) => void;
};

export function useAssistantAudio({
    onSpeakingStart,
    onSpeakingEnd,
    onQueueIdle,
    onSegmentPlaybackStart,
    onSegmentPlaybackComplete,
    onSessionPlaybackIdle,
}: UseAssistantAudioParams) {
    const currentRequestIdRef = useRef<string | null>(null);
    const pendingSegmentsRef = useRef<Map<number, AudioSegmentReadyEvent>>(new Map());
    const skippedSegmentsRef = useRef<Set<number>>(new Set());
    const nextSequenceRef = useRef(1);
    const currentAudioRef = useRef<HTMLAudioElement | null>(null);
    const isDrainingRef = useRef(false);
    const isSpeakingRef = useRef(false);
    const sessionGenerationRef = useRef(0);

    const setSpeaking = useCallback(
        (isSpeaking: boolean) => {
            if (isSpeakingRef.current === isSpeaking) return;
            isSpeakingRef.current = isSpeaking;

            if (isSpeaking) {
                onSpeakingStart();
            } else {
                onSpeakingEnd();
            }
        },
        [onSpeakingEnd, onSpeakingStart]
    );

    const stopCurrentAudioElement = useCallback(() => {
        const audio = currentAudioRef.current;
        if (!audio) return;

        audio.onended = null;
        audio.onerror = null;
        audio.onplaying = null;
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
        currentAudioRef.current = null;
    }, []);

    const hasPendingWork = useCallback(() => {
        return (
            currentAudioRef.current !== null ||
            pendingSegmentsRef.current.size > 0 ||
            skippedSegmentsRef.current.size > 0 ||
            isDrainingRef.current
        );
    }, []);

    const stopAllAudio = useCallback(() => {
        sessionGenerationRef.current += 1;
        pendingSegmentsRef.current.clear();
        skippedSegmentsRef.current.clear();
        nextSequenceRef.current = 1;
        isDrainingRef.current = false;
        currentRequestIdRef.current = null;

        stopCurrentAudioElement();
        setSpeaking(false);
        onQueueIdle?.();
    }, [onQueueIdle, setSpeaking, stopCurrentAudioElement]);

    const playSegment = useCallback(
        (segment: AudioSegmentReadyEvent, generation: number) => {
            return new Promise<void>((resolve) => {
                if (sessionGenerationRef.current !== generation) {
                    resolve();
                    return;
                }

                const audio = new Audio(convertFileSrc(segment.output_path));
                currentAudioRef.current = audio;

                let settled = false;
                const settle = () => {
                    if (settled) return;
                    settled = true;

                    audio.onended = null;
                    audio.onerror = null;
                    audio.onplaying = null;

                    if (currentAudioRef.current === audio) {
                        currentAudioRef.current = null;
                    }

                    if (sessionGenerationRef.current === generation) {
                        onSegmentPlaybackComplete?.(segment);
                        setSpeaking(false);
                    }

                    resolve();
                };

                audio.onplaying = () => {
                    if (
                        sessionGenerationRef.current === generation &&
                        currentRequestIdRef.current === segment.request_id
                    ) {
                        onSegmentPlaybackStart?.(segment);
                        setSpeaking(true);
                    }
                };

                audio.onended = settle;
                audio.onerror = () => {
                    console.error("Errore riproduzione segmento TTS:", segment.output_path);
                    settle();
                };

                audio.play().catch((error) => {
                    console.error("Audio play failed:", error);
                    settle();
                });
            });
        },
        [onSegmentPlaybackComplete, onSegmentPlaybackStart, setSpeaking]
    );

    const drainQueue = useCallback(async () => {
        if (isDrainingRef.current) return;

        isDrainingRef.current = true;
        const generation = sessionGenerationRef.current;

        try {
            while (sessionGenerationRef.current === generation) {
                while (skippedSegmentsRef.current.has(nextSequenceRef.current)) {
                    skippedSegmentsRef.current.delete(nextSequenceRef.current);
                    nextSequenceRef.current += 1;
                }

                const next = pendingSegmentsRef.current.get(nextSequenceRef.current);
                if (!next) break;

                if (currentRequestIdRef.current !== next.request_id) {
                    pendingSegmentsRef.current.delete(nextSequenceRef.current);
                    nextSequenceRef.current += 1;
                    continue;
                }

                pendingSegmentsRef.current.delete(nextSequenceRef.current);
                nextSequenceRef.current += 1;

                await playSegment(next, generation);
            }
        } finally {
            if (sessionGenerationRef.current === generation) {
                isDrainingRef.current = false;

                while (skippedSegmentsRef.current.has(nextSequenceRef.current)) {
                    skippedSegmentsRef.current.delete(nextSequenceRef.current);
                    nextSequenceRef.current += 1;
                }

                const hasNextSegment = pendingSegmentsRef.current.has(nextSequenceRef.current);
                const hasNextSkip = skippedSegmentsRef.current.has(nextSequenceRef.current);

                if (hasNextSegment || hasNextSkip) {
                    void drainQueue();
                    return;
                }

                if (!hasPendingWork()) {
                    const completedRequestId = currentRequestIdRef.current;
                    setSpeaking(false);
                    onQueueIdle?.();
                    if (completedRequestId) {
                        onSessionPlaybackIdle?.(completedRequestId);
                    }
                }
            }
        }
    }, [hasPendingWork, onQueueIdle, onSessionPlaybackIdle, playSegment, setSpeaking]);

    const startNewRequestAudioSession = useCallback(
        (requestId: string) => {
            stopAllAudio();
            currentRequestIdRef.current = requestId;
            sessionGenerationRef.current += 1;
            nextSequenceRef.current = 1;
        },
        [stopAllAudio]
    );

    const enqueueAudioSegment = useCallback(
        (segment: AudioSegmentReadyEvent) => {
            if (currentRequestIdRef.current !== segment.request_id) return;
            if (segment.sequence < nextSequenceRef.current) return;

            pendingSegmentsRef.current.set(segment.sequence, segment);
            void drainQueue();
        },
        [drainQueue]
    );

    const markAudioSegmentFailed = useCallback(
        (segment: AudioSegmentFailedEvent) => {
            if (currentRequestIdRef.current !== segment.request_id) return;
            if (segment.sequence < nextSequenceRef.current) return;

            console.error("Segmento audio TTS fallito:", segment);
            skippedSegmentsRef.current.add(segment.sequence);
            void drainQueue();
        },
        [drainQueue]
    );

    return {
        enqueueAudioSegment,
        hasPendingWork,
        markAudioSegmentFailed,
        startNewRequestAudioSession,
        stopAllAudio,
    };
}
