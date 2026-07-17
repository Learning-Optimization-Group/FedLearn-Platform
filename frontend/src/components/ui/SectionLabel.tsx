import { type HTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type SectionLabelProps = HTMLAttributes<HTMLDivElement>;

/**
 * The one uppercase micro-label. Replaces the hand-typed
 * `text-caption uppercase tracking-*` variants scattered across views.
 */
export function SectionLabel({ className, ...props }: SectionLabelProps) {
    return (
        <div
            className={cn(
                'text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-muted',
                className,
            )}
            {...props}
        />
    );
}

export default SectionLabel;
