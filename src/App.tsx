import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import "./App.css";
import { AssistantChat } from "./components/AssistantChat";
import { AssistantHeader } from "./components/AssistantHeader";
import { AssistantInputBar } from "./components/AssistantInputBar";
import { DesktopAgentPanel } from "./components/DesktopAgentPanel";
import AstraOrb from "./components/AstraOrb";
import { useAssistantSession } from "./hooks/useAssistantSession";
import { useAssistantVisualState } from "./hooks/useAssistantVisualState";
import { useWindowControls } from "./hooks/useWindowControls";
import SplitText from "./components/SplitText";

import AstraLogo from "./assets/astra_01.png";

type IntroPhase = "logo" | "splitText" | "main";

function App() {
    const session = useAssistantSession();
    const [isDesktopPanelOpen, setIsDesktopPanelOpen] = useState(false);

    const [introPhase, setIntroPhase] = useState<IntroPhase>("logo");
    const [splitTextDone, setSplitTextDone] = useState(false);

    const statusLabel = useAssistantVisualState(session.status);
    const windowControls = useWindowControls({
        onBeforeClose: session.stopAllAudio,
    });

    const logoWrapRef = useRef<HTMLDivElement | null>(null);
    const splitWrapRef = useRef<HTMLDivElement | null>(null);
    const headerWrapRef = useRef<HTMLDivElement | null>(null);
    const orbWrapRef = useRef<HTMLDivElement | null>(null);
    const chatWrapRef = useRef<HTMLDivElement | null>(null);
    const inputWrapRef = useRef<HTMLDivElement | null>(null);

    const revealTimeoutRef = useRef<number | null>(null);

    useEffect(() => {
        if (headerWrapRef.current) {
            gsap.set(headerWrapRef.current, { autoAlpha: 0, y: -36 });
        }
        if (orbWrapRef.current) {
            gsap.set(orbWrapRef.current, { autoAlpha: 0, scale: 0.94 });
        }
        if (chatWrapRef.current) {
            gsap.set(chatWrapRef.current, { autoAlpha: 0, y: 18 });
        }
        if (inputWrapRef.current) {
            gsap.set(inputWrapRef.current, { autoAlpha: 0, y: 36 });
        }
    }, []);

    useEffect(() => {
        if (introPhase !== "logo" || !logoWrapRef.current) return;

        const tl = gsap.timeline({
            onComplete: () => {
                setIntroPhase("splitText");
            },
        });

        tl.set(logoWrapRef.current, { autoAlpha: 0, scale: 0.82, y: 18 })
            .to(logoWrapRef.current, {
                autoAlpha: 1,
                scale: 1,
                y: 0,
                duration: 0.95,
                ease: "power3.out",
            })
            .to(
                logoWrapRef.current,
                {
                    scale: 1.04,
                    duration: 0.5,
                    ease: "power2.inOut",
                },
                ">-0.1"
            )
            .to(logoWrapRef.current, {
                autoAlpha: 0,
                scale: 0.9,
                y: -12,
                duration: 0.7,
                ease: "power3.inOut",
                delay: 0.35,
            });

        return () => {
            tl.kill();
        };
    }, [introPhase]);

    useEffect(() => {
        if (!splitTextDone || introPhase !== "splitText") return;

        revealTimeoutRef.current = window.setTimeout(() => {
            setIntroPhase("main");

            const tl = gsap.timeline({ defaults: { ease: "power3.out" } });

            if (headerWrapRef.current) {
                tl.to(
                    headerWrapRef.current,
                    {
                        autoAlpha: 1,
                        y: 0,
                        duration: 0.75,
                    },
                    0
                );
            }

            if (orbWrapRef.current) {
                tl.to(
                    orbWrapRef.current,
                    {
                        autoAlpha: 1,
                        scale: 1,
                        duration: 0.85,
                    },
                    0.15
                );
            }

            if (chatWrapRef.current) {
                tl.to(
                    chatWrapRef.current,
                    {
                        autoAlpha: 1,
                        y: 0,
                        duration: 0.7,
                    },
                    0.28
                );
            }

            if (inputWrapRef.current) {
                tl.to(
                    inputWrapRef.current,
                    {
                        autoAlpha: 1,
                        y: 0,
                        duration: 0.8,
                    },
                    0.22
                );
            }
        }, 4000);

        return () => {
            if (revealTimeoutRef.current !== null) {
                window.clearTimeout(revealTimeoutRef.current);
            }
        };
    }, [introPhase, splitTextDone]);

    return (
        <main className="overlay-shell">
            <section
                className="assistant-overlay bg-white"
                style={{
                    position: "relative",
                    overflow: "hidden",
                    display: "flex",
                    flexDirection: "column",
                    minHeight: "100vh",
                }}
            >
                <div ref={headerWrapRef}>
                    <AssistantHeader
                        activeModel={session.activeModel}
                        isPinned={windowControls.isPinned}
                        isDesktopPanelOpen={isDesktopPanelOpen}
                        onClose={windowControls.close}
                        onMinimize={windowControls.minimize}
                        onToggleDesktopPanel={() => setIsDesktopPanelOpen((current) => !current)}
                        onTogglePin={windowControls.togglePin}
                        startDrag={windowControls.startDrag}
                        statusLabel={statusLabel}
                    />
                </div>

                {introPhase === "logo" && (
                    <div
                        ref={logoWrapRef}
                        className="absolute flex items-center justify-center z-30 pointer-events-none inset-0"
                    >
                        <img
                            src={AstraLogo}
                            alt="Astra"
                            style={{
                                width: "100%",
                                height: "100%",
                                objectFit: "contain",
                            }}
                        />
                    </div>
                )}

                {introPhase === "splitText" && (
                    <div
                        ref={splitWrapRef}
                        style={{
                            position: "absolute",
                            inset: 0,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            zIndex: 25,
                            pointerEvents: "none",
                        }}
                    >
                        <SplitText
                            text="Hello, you!"
                            className="text-3xl md:text-4xl font-semibold text-center"
                            delay={150}
                            duration={1.25}
                            ease="power3.out"
                            splitType="chars"
                            from={{ opacity: 0, y: 40 }}
                            to={{ opacity: 1, y: 0 }}
                            threshold={0.1}
                            rootMargin="-100px"
                            textAlign="center"
                            onLetterAnimationComplete={() => {
                                setSplitTextDone(true);
                            }}
                        />
                    </div>
                )}

                <div
                    ref={orbWrapRef}
                    style={{
                        width: "100%",
                        height: "300px",
                        position: "relative",
                    }}
                >
                    <section
                        className="orb-stage relative"
                        style={{ width: "100%", height: "300px", position: "relative" }}
                    >
                        <AstraOrb
                            status={session.status}
                            hoverIntensity={0.1}
                            rotateOnHover
                            forceHoverState={false}
                            backgroundColor="#ffffff"
                        />
                    </section>
                </div>

                <div ref={chatWrapRef} style={{ flex: 1, minHeight: 0 }}>
                    <AssistantChat messages={session.messages} />
                </div>

                <DesktopAgentPanel
                    isOpen={isDesktopPanelOpen}
                    onClose={() => setIsDesktopPanelOpen(false)}
                />

                <div ref={inputWrapRef}>
                    <AssistantInputBar
                        autoSubmitVoice={session.autoSubmitVoice}
                        inputValue={session.inputValue}
                        isLoading={session.isLoading}
                        lastVoiceTranscript={session.lastVoiceTranscript}
                        onSubmit={() => session.submitMessage()}
                        setAutoSubmitVoice={session.setAutoSubmitVoice}
                        setInputValue={session.setInputValue}
                        voiceInput={session.voiceInput}
                        voiceSession={session.voiceSession}
                    />
                </div>
            </section>
        </main>
    );
}

export default App;