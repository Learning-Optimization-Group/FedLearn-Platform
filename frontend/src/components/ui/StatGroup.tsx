import { type ReactNode } from 'react';
import { cn } from '../../lib/utils';
import { Card } from './Card';
import { MetricTile } from './MetricTile';

export interface StatGroupItem {
    label: ReactNode;
    value: ReactNode;
    /** Optional small slot (sparkline, delta chip) under the value. */
    sparkline?: ReactNode;
}

export interface StatGroupProps {
    stats: StatGroupItem[];
    className?: string;
}

/**
 * A row of stats in ONE card with hairline dividers — replaces the
 * card-per-number grids. Wraps on narrow viewports.
 */
export function StatGroup({ stats, className }: StatGroupProps) {
    return (
        <Card padding="none" className={cn('flex flex-wrap', className)}>
            {stats.map((s, i) => (
                <div
                    key={i}
                    className={cn(
                        'flex-1 min-w-[140px] px-5 py-4',
                        i > 0 && 'border-l border-hairline',
                    )}
                >
                    <MetricTile label={s.label} value={s.value} sparkline={s.sparkline} />
                </div>
            ))}
        </Card>
    );
}

export default StatGroup;
