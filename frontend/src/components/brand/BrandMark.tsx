import { useId } from 'react';

interface BrandMarkProps {
    /** Rendered px size (square). */
    size?: number;
    /** Show the soft ember bloom behind the core. Off for tiny sizes. */
    glow?: boolean;
    className?: string;
    title?: string;
}

/**
 * FedLearn "ember-node" mark — six device nodes feeding updates into a glowing
 * shared model. Pure inline SVG so it stays crisp at any size and inherits the
 * page's AMOLED background. Gradient IDs are namespaced per instance.
 */
export function BrandMark({ size = 32, glow = true, className, title = 'FedLearn' }: BrandMarkProps) {
    const raw = useId().replace(/[:]/g, '');
    const e = `e${raw}`;
    const ef = `ef${raw}`;
    const b = `b${raw}`;
    const sp = `sp${raw}`;
    const sf = `sf${raw}`;
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
            <defs>
                <radialGradient id={e} cx="42%" cy="34%" r="72%">
                    <stop offset="0%" stopColor="#FFE2C2" />
                    <stop offset="42%" stopColor="#F7A35C" />
                    <stop offset="100%" stopColor="#DB7430" />
                </radialGradient>
                <radialGradient id={ef} cx="42%" cy="34%" r="72%">
                    <stop offset="0%" stopColor="#FFC893" />
                    <stop offset="60%" stopColor="#F1924A" />
                    <stop offset="100%" stopColor="#C9651F" />
                </radialGradient>
                <radialGradient id={b} cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#FFAE6B" stopOpacity="0.85" />
                    <stop offset="45%" stopColor="#F7913F" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#F7913F" stopOpacity="0" />
                </radialGradient>
                <linearGradient id={sp} x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#F7A35C" stopOpacity="0.85" />
                    <stop offset="100%" stopColor="#F7A35C" stopOpacity="0.22" />
                </linearGradient>
                <filter id={sf} x="-60%" y="-60%" width="220%" height="220%">
                    <feGaussianBlur stdDeviation="26" />
                </filter>
            </defs>
            <g stroke={`url(#${sp})`} strokeWidth="9" strokeLinecap="round">
                <line x1="512" y1="512" x2="512" y2="212" />
                <line x1="512" y1="512" x2="772" y2="362" />
                <line x1="512" y1="512" x2="772" y2="662" />
                <line x1="512" y1="512" x2="512" y2="812" />
                <line x1="512" y1="512" x2="252" y2="662" />
                <line x1="512" y1="512" x2="252" y2="362" />
            </g>
            <g>
                <circle cx="512" cy="212" r="48" fill={`url(#${e})`} />
                <circle cx="772" cy="362" r="34" fill={`url(#${ef})`} />
                <circle cx="772" cy="662" r="44" fill={`url(#${e})`} />
                <circle cx="512" cy="812" r="34" fill={`url(#${ef})`} />
                <circle cx="252" cy="662" r="44" fill={`url(#${e})`} />
                <circle cx="252" cy="362" r="34" fill={`url(#${ef})`} />
            </g>
            {glow && <circle cx="512" cy="512" r="200" fill={`url(#${b})`} filter={`url(#${sf})`} />}
            <circle cx="512" cy="512" r="118" fill={`url(#${e})`} />
            <circle cx="476" cy="476" r="40" fill="#FFF1E2" fillOpacity="0.7" filter={`url(#${sf})`} />
        </svg>
    );
}

interface WordmarkProps {
    /** Mark size in px; the wordmark text scales with it. */
    size?: number;
    glow?: boolean;
    className?: string;
}

/** Mark + "FedLearn" lockup in the display face. "Learn" picks up the ember. */
export function Wordmark({ size = 30, glow = true, className }: WordmarkProps) {
    return (
        <span className={`inline-flex items-center gap-2.5 ${className ?? ''}`}>
            <BrandMark size={size} glow={glow} />
            <span
                className="font-display font-semibold tracking-tight text-fg"
                style={{ fontSize: `${Math.round(size * 0.62)}px`, lineHeight: 1 }}
            >
                Fed<span className="text-accent">Learn</span>
            </span>
        </span>
    );
}
