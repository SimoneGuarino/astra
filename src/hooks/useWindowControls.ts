import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

type UseWindowControlsParams = {
    onBeforeClose: () => void;
};

export function useWindowControls({ onBeforeClose }: UseWindowControlsParams) {
    const [isPinned, setIsPinned] = useState(true);

    const minimize = useCallback(async () => {
        await invoke("minimize_window");
    }, []);

    const close = useCallback(async () => {
        onBeforeClose();
        await invoke("close_window");
    }, [onBeforeClose]);

    const togglePin = useCallback(async () => {
        const nextValue = await invoke<boolean>("toggle_always_on_top");
        setIsPinned(nextValue);
    }, []);

    const startDrag = useCallback(async () => {
        try {
            await invoke("start_window_drag");
        } catch (error) {
            console.error("Drag error:", error);
        }
    }, []);

    return {
        close,
        isPinned,
        minimize,
        startDrag,
        togglePin,
    };
}
