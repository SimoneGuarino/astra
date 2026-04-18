import { useCallback, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type {
    VoiceSessionStartResponse,
    VoiceSessionState,
    VoiceSessionStateEvent,
} from "../types/assistant";

const TARGET_SAMPLE_RATE = 16_000;
const CHUNK_MS = 100;
const MAX_CHUNK_MS = 250;

type UseVoiceSessionParams = {
    onError: (message: string) => void;
    onSessionListening: () => void;
    onSessionStopped: () => void;
};

export function useVoiceSession({
    onError,
    onSessionListening,
    onSessionStopped,
}: UseVoiceSessionParams) {
    const [isEnabled, setIsEnabled] = useState(false);
    const [isTransitioning, setIsTransitioning] = useState(false);
    const [inputLevel, setInputLevel] = useState(0);
    const [voiceState, setVoiceState] = useState<VoiceSessionState>("disabled");
    const [sessionId, setSessionId] = useState<string | null>(null);

    const audioContextRef = useRef<AudioContext | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const workletRef = useRef<AudioWorkletNode | null>(null);
    const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const silentGainRef = useRef<GainNode | null>(null);
    const sessionIdRef = useRef<string | null>(null);
    const bufferRef = useRef<number[]>([]);
    const isSendingRef = useRef(false);
    const lifecycleGenerationRef = useRef(0);
    const isStoppingRef = useRef(false);
    const isTransitioningRef = useRef(false);
    const lastInputLevelUpdateRef = useRef(0);

    const cleanupAudioGraph = useCallback(() => {
        workletRef.current?.disconnect();
        scriptProcessorRef.current?.disconnect();
        sourceRef.current?.disconnect();
        silentGainRef.current?.disconnect();

        workletRef.current = null;
        scriptProcessorRef.current = null;
        sourceRef.current = null;
        silentGainRef.current = null;

        streamRef.current?.getTracks().forEach((track) => {
            track.onended = null;
            track.stop();
        });
        streamRef.current = null;

        void audioContextRef.current?.close();
        audioContextRef.current = null;
        bufferRef.current = [];
        isSendingRef.current = false;
        setInputLevel(0);
    }, []);

    const sendBufferedAudio = useCallback(
        async (force = false) => {
            const generation = lifecycleGenerationRef.current;
            const activeSessionId = sessionIdRef.current;
            const audioContext = audioContextRef.current;
            if (!activeSessionId || !audioContext || isSendingRef.current) return;

            const minSamples = Math.floor((TARGET_SAMPLE_RATE * CHUNK_MS) / 1000);
            const maxSamples = Math.floor((TARGET_SAMPLE_RATE * MAX_CHUNK_MS) / 1000);
            if (!force && bufferRef.current.length < minSamples) return;

            const takeCount = force
                ? bufferRef.current.length
                : Math.min(bufferRef.current.length, maxSamples);
            const samples = bufferRef.current.splice(0, takeCount);
            if (samples.length === 0) return;

            isSendingRef.current = true;
            try {
                await invoke("voice_session_audio_chunk", {
                    payload: {
                        session_id: activeSessionId,
                        sample_rate: TARGET_SAMPLE_RATE,
                        samples,
                    },
                });
            } catch (error) {
                if (lifecycleGenerationRef.current !== generation) return;
                const message = error instanceof Error ? error.message : String(error);
                onError(`Voice session audio failed: ${message}`);
            } finally {
                isSendingRef.current = false;
                if (
                    lifecycleGenerationRef.current === generation &&
                    bufferRef.current.length >= minSamples
                ) {
                    void sendBufferedAudio();
                }
            }
        },
        [onError]
    );

    const handleRawSamples = useCallback(
        (samples: Float32Array, sourceSampleRate: number) => {
            if (!sessionIdRef.current || !audioContextRef.current) return;
            const now = performance.now();
            if (now - lastInputLevelUpdateRef.current >= 120) {
                lastInputLevelUpdateRef.current = now;
                setInputLevel(samplesToLevel(samples));
            }
            const downsampled = downsampleToTarget(samples, sourceSampleRate, TARGET_SAMPLE_RATE);
            for (const sample of downsampled) {
                bufferRef.current.push(sample);
            }
            void sendBufferedAudio();
        },
        [sendBufferedAudio]
    );

    const stop = useCallback(async () => {
        lifecycleGenerationRef.current += 1;
        isStoppingRef.current = true;
        isTransitioningRef.current = true;
        setIsTransitioning(true);
        cleanupAudioGraph();
        setIsEnabled(false);
        setVoiceState("disabled");
        setSessionId(null);
        sessionIdRef.current = null;
        onSessionStopped();

        try {
            await invoke("stop_voice_session");
        } catch (error) {
            console.error("stop_voice_session error:", error);
        } finally {
            isStoppingRef.current = false;
            isTransitioningRef.current = false;
            setIsTransitioning(false);
        }
    }, [cleanupAudioGraph, onSessionStopped]);

    const start = useCallback(async () => {
        if (isEnabled || isTransitioningRef.current) return;
        if (!navigator.mediaDevices?.getUserMedia) {
            onError("Microfono non disponibile in questo ambiente.");
            return;
        }

        const generation = lifecycleGenerationRef.current + 1;
        lifecycleGenerationRef.current = generation;
        isStoppingRef.current = false;
        isTransitioningRef.current = true;
        setIsTransitioning(true);
        let stream: MediaStream | null = null;

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            if (lifecycleGenerationRef.current !== generation) {
                stream.getTracks().forEach((track) => track.stop());
                return;
            }

            const response = await invoke<VoiceSessionStartResponse>("start_voice_session");
            if (lifecycleGenerationRef.current !== generation) {
                stream.getTracks().forEach((track) => track.stop());
                return;
            }

            sessionIdRef.current = response.session_id;
            setSessionId(response.session_id);

            const audioContext = new AudioContext();
            if (audioContext.state === "suspended") {
                await audioContext.resume();
            }
            const source = audioContext.createMediaStreamSource(stream);
            const silentGain = audioContext.createGain();
            silentGain.gain.value = 0;

            stream.getAudioTracks().forEach((track) => {
                track.onended = () => {
                    if (
                        lifecycleGenerationRef.current !== generation ||
                        isStoppingRef.current
                    ) {
                        return;
                    }
                    onError("Microfono disconnesso o non piu disponibile.");
                    void stop();
                };
            });

            streamRef.current = stream;
            audioContextRef.current = audioContext;
            sourceRef.current = source;
            silentGainRef.current = silentGain;

            if (audioContext.audioWorklet) {
                await audioContext.audioWorklet.addModule("/voice-capture-processor.js");
                const worklet = new AudioWorkletNode(audioContext, "voice-capture-processor");
                worklet.port.onmessage = (event: MessageEvent<Float32Array>) => {
                    handleRawSamples(event.data, audioContext.sampleRate);
                };
                source.connect(worklet);
                worklet.connect(silentGain);
                workletRef.current = worklet;
            } else {
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                processor.onaudioprocess = (event) => {
                    handleRawSamples(event.inputBuffer.getChannelData(0), audioContext.sampleRate);
                };
                source.connect(processor);
                processor.connect(silentGain);
                scriptProcessorRef.current = processor;
            }

            silentGain.connect(audioContext.destination);
            setIsEnabled(true);
            setVoiceState("passive");
            onSessionListening();
        } catch (error) {
            stream?.getTracks().forEach((track) => track.stop());
            cleanupAudioGraph();
            const message = error instanceof Error ? error.message : String(error);
            setIsEnabled(false);
            setVoiceState("disabled");
            setSessionId(null);
            sessionIdRef.current = null;
            if (lifecycleGenerationRef.current === generation) {
                onError(`Avvio sessione vocale fallito: ${message}`);
            }
            if (lifecycleGenerationRef.current === generation) {
                try {
                    await invoke("report_voice_session_error", { message });
                } catch (reportError) {
                    console.error("report_voice_session_error error:", reportError);
                }
            }
        } finally {
            if (lifecycleGenerationRef.current === generation) {
                isTransitioningRef.current = false;
                setIsTransitioning(false);
            }
        }
    }, [cleanupAudioGraph, handleRawSamples, isEnabled, onError, onSessionListening, stop]);

    const toggle = useCallback(async () => {
        if (isEnabled) {
            await stop();
        } else {
            await start();
        }
    }, [isEnabled, start, stop]);

    const applyStateEvent = useCallback((event: VoiceSessionStateEvent) => {
        setVoiceState(event.state);
        if (event.session_id !== undefined) {
            setSessionId(event.session_id);
            sessionIdRef.current = event.session_id;
        }
        if (event.state === "disabled") {
            setIsEnabled(false);
        }
    }, []);

    return {
        applyStateEvent,
        inputLevel,
        isEnabled,
        isTransitioning,
        sessionId,
        start,
        stop,
        toggle,
        voiceState,
    };
}

function downsampleToTarget(
    input: Float32Array,
    sourceSampleRate: number,
    targetSampleRate: number
) {
    if (sourceSampleRate === targetSampleRate) {
        return Array.from(input, clampSample);
    }

    const ratio = sourceSampleRate / targetSampleRate;
    const outputLength = Math.floor(input.length / ratio);
    const output = new Array<number>(outputLength);

    for (let i = 0; i < outputLength; i += 1) {
        const start = Math.floor(i * ratio);
        const end = Math.min(Math.floor((i + 1) * ratio), input.length);
        let sum = 0;
        let count = 0;
        for (let j = start; j < end; j += 1) {
            sum += input[j] ?? 0;
            count += 1;
        }
        output[i] = clampSample(count > 0 ? sum / count : 0);
    }

    return output;
}

function clampSample(value: number) {
    return Math.max(-1, Math.min(1, value));
}

function samplesToLevel(samples: Float32Array) {
    if (samples.length === 0) return 0;

    let sum = 0;
    for (const sample of samples) {
        sum += sample * sample;
    }

    const rms = Math.sqrt(sum / samples.length);
    return Math.max(0, Math.min(1, rms / 0.05));
}
