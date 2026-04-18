
type Options = "none" | "xs" | "sm" | "md" | "lg" | "full";

type ButtonProps = {
    variant?: "primary" | "secondary" | "danger" | "dark" | "text";
    radius?: Options;
    size?: Options;
    className?: string;
    onClick?: () => void;
    disabled?: boolean;
    loading?: boolean;
    title?: string;
    children?: React.ReactNode;
};

const sizeStyles: Record<Options, string> = {
    none: "",
    xs: "p-2 text-sm",
    sm: "p-3 text-sm",
    md: "p-4 text-base",
    lg: "p-6 text-lg",
    full: "w-full",
};

const variantStyles: Record<string, string> = {
    primary: "bg-blue-500 hover:bg-blue-600 text-white",
    secondary: "bg-gray-500 hover:bg-gray-600 text-white",
    danger: "bg-red-500 hover:bg-red-600 text-white",
    dark: "bg-gray-900/90 hover:bg-gray-900 text-white",
    text: "bg-transparent hover:bg-gray-200 text-gray-700",
};

const radiusStyles: Record<Options, string> = {
    none: "rounded-none",
    xs: "rounded-xs",
    sm: "rounded-sm",
    md: "rounded-md",
    lg: "rounded-lg",
    full: "rounded-full",
};

export function Button({
    variant = "text",
    radius = "md",
    size = "md",
    className = "",
    onClick = () => {},
    disabled = false,
    loading = false,
    title,
    children,
}: ButtonProps) {
    return (
        <button
            className={`${variantStyles[variant]} ${className} ${disabled ? "cursor-not-allowed opacity-20" : "cursor-pointer"} ${radiusStyles[radius]} ${sizeStyles[size]} h-fit
                ${loading ? "opacity-70 cursor-not-allowed" : ""}`}
            onClick={onClick}
            disabled={disabled}
            title={title}
        >
            {loading ? "Caricamento..." : children}
        </button>
    );
}