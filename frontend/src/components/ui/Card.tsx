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
}

/**
 * Surface container — the base panel for cards, panels and modals.
 * bg-surface-1 + 1px hairline border + rounded-card. Depth on dark comes from the
 * surface ladder + hairline, never box-shadow.
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(
    ({ className, padding = 'md', ...props }, ref) => (
        <div
            ref={ref}
            className={cn(
                'bg-surface-1 border border-hairline rounded-card',
                PADDING[padding],
                className,
            )}
            {...props}
        />
    ),
);

Card.displayName = 'Card';

export default Card;
