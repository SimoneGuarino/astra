import { useEffect, useRef, useState } from "react";
import type { GraphSurfaceSize } from "../layout/memoryGraphLayoutTypes";

export function useMemoryGraphSurfaceSize() {
  const shellRef = useRef<HTMLDivElement | null>(null);
  const [surfaceSize, setSurfaceSize] = useState<GraphSurfaceSize>({ width: 0, height: 0 });

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;

    const updateSize = () => {
      const rect = shell.getBoundingClientRect();
      setSurfaceSize((previous) => {
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));
        if (previous.width === width && previous.height === height) return previous;
        return { width, height };
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(shell);
    return () => observer.disconnect();
  }, []);

  return { shellRef, surfaceSize };
}
