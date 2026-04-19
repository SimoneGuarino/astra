import { useMemo } from "react";
import type { AssistantStatus } from "../types/assistant";

export function useAssistantVisualState(status: AssistantStatus) {
    return useMemo(() => {
        if (status === "passive") return 'Live: di pure "Astra"';
        if (status === "armed") return "Astra attiva";
        if (status === "thinking") return "Elaboro...";
        if (status === "listening") return "Ti ascolto...";
        if (status === "speaking") return "Sto parlando...";
        if (status === "settling") return "Sto finendo...";
        return "Online";
    }, [status]);
}
