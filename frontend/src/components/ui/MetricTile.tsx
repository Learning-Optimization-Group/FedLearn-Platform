import { type ReactNode } from 'react';
import { cn } from '../../lib/utils';

export interface MetricTileProps {
    /** Uppercase caption above the value. */
    label: ReactNode;
    /** The metric value — rendered mono + tabular-nums for stable alignment. */
    value: ReactNode;
    /** Optional small slot (sparkline, delta chip) shown under the value. */
    sparkline?: ReactNode;
    className?: string;
}

/**
 * Single stat readout. Label is an uppercase caption in fg-muted; value is h3
 * mono tabular-nums in fg. Numbers stay grid-aligned across tiles.
 */
export function MetricTile({ label, value, sparkline, className }: MetricTileProps) {
    return (
        <div className={cn('flex flex-col gap-1', className)}>
            <span className="text-caption uppercase tracking-wide text-fg-muted">
                {label}
            </span>
            <span className="text-h3 font-mono tabular-nums text-fg">{value}</span>
            {sparkline && <div className="mt-1">{sparkline}</div>}
        </div>
    );
}

export default MetricTile;
