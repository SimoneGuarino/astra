import { IoClose } from "react-icons/io5";
import { MdOutlineMinimize } from "react-icons/md";
import { LuBrain, LuMonitor, LuMonitorOff } from "react-icons/lu";

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
            <header className="overlay-topbar pl-2 pr-1" >
                <div className="w-fullcursor-move flex w-full space-x-1 cursor-move" onMouseDown={startDrag}>
                    <span title="Modello attivo" className="p-2 rounded-full text-xs bg-gray-200/80 text-gray-500">{activeModel}</span>
                    <span title="Stato attuale" className="p-2 rounded-full text-xs bg-gray-200/80 text-gray-500">{statusLabel}</span>
                </div>

                <div className="flex">
                    <Button variant={isDesktopPanelOpen ? "primary" : "text"} radius="full" size="xs" title={`Desktop agent panel: ${isDesktopPanelOpen ? "Open" : "Closed"}`} onClick={onToggleDesktopPanel}>
                        <LuBrain />
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
