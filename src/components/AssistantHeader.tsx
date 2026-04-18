import type { AssistantStatus } from "../types/assistant";
import Astra from "../assets/astra.png";

type AssistantHeaderProps = {
    activeModel: string;
    isPinned: boolean;
    onClose: () => void;
    onMinimize: () => void;
    onTogglePin: () => void;
    startDrag: (e: React.MouseEvent) => void;
    status: AssistantStatus;
    statusLabel: string;
};

export function AssistantHeader({
    activeModel,
    isPinned,
    onClose,
    onMinimize,
    onTogglePin,
    startDrag,
    status,
    statusLabel,
}: AssistantHeaderProps) {
    return (
        <>
            <header className="overlay-topbar window-drag-strip" onMouseDown={startDrag}>
                <div className="overlay-brand">
                    <div>
                        <p className="overlay-kicker">PERSONAL AI</p>
                        <h1>Astra</h1>
                    </div>
                </div>

                <div className="overlay-controls">
                    <button className="overlay-control" onClick={onTogglePin}>
                        {isPinned ? "Unpin" : "Pin"}
                    </button>
                    <button className="overlay-control" onClick={onMinimize}>
                        Min
                    </button>
                    <button className="overlay-control danger" onClick={onClose}>
                        X
                    </button>
                </div>
            </header>

            <section className="flex items-center justify-between px-4 mb-2">
                <p className="assistant-model-label">Model: {activeModel}</p>

                <div className={`assistant-status ${status}`}>
                    <span className={`status-dot ${status}`} />
                    <span>{statusLabel}</span>
                </div>
            </section>
        </>
    );
}
