import { forwardRef, type HTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

type CardPadding = 'none' | 'sm' | 'md' | 'lg';

const PADDING: Record<CardPadding, string> = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
};

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
    /** Inner padding. Defaults to `md` (16px). Use `none` when composing custom regions. */
    padding?: CardPadding;
    /** Hover affordance: warms the border to ember and lifts slightly. For clickable cards. */
    interactive?: boolean;
    /** Adds a soft ember halo — for focal / featured cards. */
    glow?: boolean;
}

/**
 * Surface container — the base panel for cards, panels and modals.
 * Near-black bg-surface-1 + 1px hairline + rounded-card, lifted from the void by
 * the surface ladder + a faint top highlight. `interactive` adds an ember hover.
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(
    ({ className, padding = 'md', interactive = false, glow = false, ...props }, ref) => (
        <div
            ref={ref}
            className={cn(
                'relative bg-surface-1 border border-hairline rounded-card',
                'before:pointer-events-none before:absolute before:inset-x-0 before:top-0 before:h-px',
                'before:bg-gradient-to-r before:from-transparent before:via-white/[0.06] before:to-transparent before:rounded-t-card',
                interactive &&
                    'transition-[transform,border-color,background-color] duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] hover:-translate-y-0.5 hover:border-accent/35 hover:bg-surface-2 cursor-pointer',
                glow && 'glow-ember',
                PADDING[padding],
                className,
            )}
            {...props}
        />
    ),
);

Card.displayName = 'Card';

export default Card;
