import { useRef, useEffect, useState } from "react";
import gsap from "gsap";

interface BackdropProps {
    onClick: () => void;
    passThrough?: boolean;
    isVisible?: boolean;
}

const DURATION = 0.25;

export const Backdrop: React.FC<BackdropProps> = ({
    onClick,
    passThrough,
    isVisible = true,
}) => {
    const ref = useRef<HTMLDivElement>(null);
    const [mounted, setMounted] = useState(true);

    // Enter: al mount
    useEffect(() => {
        gsap.fromTo(ref.current, { opacity: 0 }, { opacity: 1, duration: DURATION });
    }, []);

    // Exit: quando isVisible diventa false
    useEffect(() => {
        if (isVisible) return;

        gsap.to(ref.current, {
            opacity: 0,
            duration: DURATION,
            onComplete: () => setMounted(false), // smonta dopo l'animazione
        });
    }, [isVisible]);

    if (!mounted) return null;

    return (
        <div
            ref={ref}
            style={{ opacity: 0 }} // parte invisibile, GSAP anima da qui
            className={`fixed inset-0 z-10 bg-black/40 backdrop-blur-xs ${
                passThrough ? "pointer-events-none" : ""
            }`}
            onClick={!passThrough ? onClick : undefined}
            aria-hidden={passThrough ? "true" : "false"}
        />
    );
};