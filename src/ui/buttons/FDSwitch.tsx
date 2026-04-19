import React, { forwardRef, memo, useState } from "react";

export type FDSwitchSize = "sm" | "md" | "lg";
export type FDSwitchColor = "primary" | "success" | "warning" | "error" | "neutral";

export interface FDSwitchProps extends Omit<
    React.ButtonHTMLAttributes<HTMLButtonElement>,
    "children" | "onChange" | "type" | "role" | "aria-checked"
> {
    checked?: boolean;
    defaultChecked?: boolean;
    onChange?: (nextChecked: boolean) => void;
    size?: FDSwitchSize;
    color?: FDSwitchColor;
    disabled?: boolean;
    ariaLabel?: string;
    thumbClassName?: string;
}

function cn(...v: Array<string | false | null | undefined>) {
    return v.filter(Boolean).join(" ");
}

const sizeMap: Record<FDSwitchSize, { track: string; thumb: string; on: string; off: string }> = {
    sm: { track: "h-5 w-9", thumb: "h-4 w-4", on: "translate-x-4", off: "translate-x-0.5" },
    md: { track: "h-6 w-11", thumb: "h-5 w-5", on: "translate-x-5", off: "translate-x-0.5" },
    lg: { track: "h-7 w-14", thumb: "h-6 w-6", on: "translate-x-7", off: "translate-x-0.5" },
};

const checkedTrackByColor: Record<FDSwitchColor, string> = {
    primary: "bg-sky-600",
    success: "bg-emerald-600",
    warning: "bg-amber-500",
    error: "bg-rose-600",
    neutral: "bg-neutral-700 dark:bg-neutral-300",
};

const focusRingByColor: Record<FDSwitchColor, string> = {
    primary: "focus-visible:ring-sky-500/60",
    success: "focus-visible:ring-emerald-500/60",
    warning: "focus-visible:ring-amber-500/60",
    error: "focus-visible:ring-rose-500/60",
    neutral: "focus-visible:ring-neutral-500/60",
};

const FDSwitchInner = forwardRef<HTMLButtonElement, FDSwitchProps>(function FDSwitchInner(
    {
        checked,
        defaultChecked = false,
        onChange,
        onClick,
        disabled,
        size = "md",
        color = "primary",
        className,
        thumbClassName,
        ariaLabel,
        ...rest
    },
    ref
) {
    const isControlled = checked !== undefined;
    const [uncontrolledChecked, setUncontrolledChecked] = useState<boolean>(defaultChecked);

    const isChecked = isControlled ? Boolean(checked) : uncontrolledChecked;
    const sizeCfg = sizeMap[size];

    const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
        if (disabled) return;

        const next = !isChecked;
        if (!isControlled) setUncontrolledChecked(next);
        onChange?.(next);
        onClick?.(event);
    };

    return (
        <button
            ref={ref}
            type="button"
            role="switch"
            aria-checked={isChecked}
            aria-label={ariaLabel}
            disabled={disabled}
            onClick={handleClick}
            className={cn(
                "relative inline-flex items-center rounded-full transition-colors duration-200",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-white dark:focus-visible:ring-offset-neutral-900",
                focusRingByColor[color],
                sizeCfg.track,
                isChecked ? checkedTrackByColor[color] : "bg-neutral-300 dark:bg-neutral-700",
                disabled ? "opacity-60 cursor-not-allowed" : "cursor-pointer",
                className,
            )}
            {...rest}
        >
            <span
                className={cn(
                    "inline-block transform rounded-full bg-white shadow transition-transform duration-200",
                    sizeCfg.thumb,
                    isChecked ? sizeCfg.on : sizeCfg.off,
                    thumbClassName,
                )}
            />
        </button>
    );
});

export const FDSwitch = memo(FDSwitchInner);
export default FDSwitch;
