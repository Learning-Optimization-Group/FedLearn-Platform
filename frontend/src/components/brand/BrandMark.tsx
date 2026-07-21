interface BrandMarkProps {
    /** Rendered px size (square). */
    size?: number;
    /** Deprecated — the mark is flat now; accepted so call sites keep compiling. */
    glow?: boolean;
    className?: string;
    title?: string;
}

/**
 * FedLearn network mark — six device nodes feeding updates into a shared
 * model. Flat single-ink rendering via currentColor (defaults to the accent
 * ink through the `text-accent` class at call sites); crisp at any size.
 */
export function BrandMark({ size = 32, glow: _glow, className, title = 'FedLearn' }: BrandMarkProps) {
    return (
        <svg
            width={size}
            height={size}
            viewBox="0 0 1024 1024"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            role="img"
            aria-label={title}
            className={className}
        >
            <g stroke="currentColor" strokeOpacity="0.35" strokeWidth="14" strokeLinecap="round">
                <line x1="512" y1="512" x2="512" y2="212" />
                <line x1="512" y1="512" x2="772" y2="362" />
                <line x1="512" y1="512" x2="772" y2="662" />
                <line x1="512" y1="512" x2="512" y2="812" />
                <line x1="512" y1="512" x2="252" y2="662" />
                <line x1="512" y1="512" x2="252" y2="362" />
            </g>
            <g fill="currentColor">
                <circle cx="512" cy="212" r="48" />
                <circle cx="772" cy="362" r="34" />
                <circle cx="772" cy="662" r="44" />
                <circle cx="512" cy="812" r="34" />
                <circle cx="252" cy="662" r="44" />
                <circle cx="252" cy="362" r="34" />
                <circle cx="512" cy="512" r="118" />
            </g>
        </svg>
    );
}

interface WordmarkProps {
    /** Mark size in px; the wordmark text scales with it. */
    size?: number;
    glow?: boolean;
    className?: string;
}

/** Mark + "FedLearn" lockup — navy mark, single-ink wordmark. */
export function Wordmark({ size = 30, glow, className }: WordmarkProps) {
    return (
        <span className={`inline-flex items-center gap-2.5 ${className ?? ''}`}>
            <BrandMark size={size} glow={glow} className="text-accent" />
            <span
                className="font-semibold tracking-tight text-fg"
                style={{ fontSize: `${Math.round(size * 0.62)}px`, lineHeight: 1 }}
            >
                FedLearn
            </span>
        </span>
    );
}
