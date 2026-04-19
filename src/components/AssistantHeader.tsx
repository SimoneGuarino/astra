import { IoClose } from "react-icons/io5";
import { MdOutlineMinimize } from "react-icons/md";
import { LuMonitor, LuMonitorOff } from "react-icons/lu";

import { Button } from "../ui/buttons/Button";

type AssistantHeaderProps = {
    activeModel: string;
    isPinned: boolean;
    isDesktopPanelOpen: boolean;
    onClose: () => void;
    onMinimize: () => void;
    onTogglePin: () => void;
    onToggleDesktopPanel: () => void;
    startDrag: (e: React.MouseEvent) => void;
    statusLabel: string;
};

export function AssistantHeader({
    activeModel,
    isPinned,
    isDesktopPanelOpen,
    onClose,
    onMinimize,
    onToggleDesktopPanel,
    onTogglePin,
    startDrag,
    statusLabel,
}: AssistantHeaderProps) {
    return (
        <>
            <header className="overlay-topbar" >
                <div className="overlay-brand w-full cursor-move"  onMouseDown={startDrag}>
                    <div>
                        <p className="overlay-kicker">PERSONAL AI</p>
                        <h1>Astra</h1>
                        <p className="assistant-model-label">Model: {activeModel} | {statusLabel}</p>
                    </div>
                </div>

                <div className="flex">
                    <Button variant="text" radius="full" size="xs" title="Desktop agent panel" onClick={onToggleDesktopPanel}>
                        {isDesktopPanelOpen ? "Agent −" : "Agent +"}
                    </Button>
                    <Button variant="text" radius="full" size="xs" title={isPinned ? "Rimuovi fissaggio" : "Fissa la finestra sempre on top"} onClick={onTogglePin}>
                        {isPinned ? <LuMonitorOff /> : <LuMonitor />}
                    </Button>
                    <Button
                        variant="text" radius="full" size="xs" title="Minimizza" onClick={onMinimize}>
                        <MdOutlineMinimize />
                    </Button>
                    <Button
                        variant="text" radius="full" size="xs" title="Chiudi" onClick={onClose}>
                        <IoClose />
                    </Button>
                </div>
            </header>
        </>
    );
}
