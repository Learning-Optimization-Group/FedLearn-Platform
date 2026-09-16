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
    /** Hover affordance: strengthens the border and lifts the shadow. For clickable cards. */
    interactive?: boolean;
    /** Deprecated — halos are retired; accepted so call sites keep compiling. */
    glow?: boolean;
}

/**
 * Surface container — the base panel for cards, panels and modals.
 * White surface + 1px hairline + rounded-card + the quiet card shadow.
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(
    ({ className, padding = 'md', interactive = false, glow: _glow = false, ...props }, ref) => (
        <div
            ref={ref}
            className={cn(
                'relative bg-surface-1 border border-hairline rounded-card shadow-card',
                interactive &&
                    'transition-[border-color,box-shadow] duration-[180ms] ease-[cubic-bezier(0.4,0,0.2,1)] hover:border-line hover:shadow-card-hover cursor-pointer',
                PADDING[padding],
                className,
            )}
            {...props}
        />
    ),
);

Card.displayName = 'Card';

export default Card;
