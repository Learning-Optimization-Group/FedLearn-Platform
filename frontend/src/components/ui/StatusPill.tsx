import { type ReactNode } from 'react';
import { cn } from '../../lib/utils';

export type StatusKind =
    | 'running'
    | 'completed'
    | 'error'
    | 'pending'
    | 'idle';

/**
 * The ONE status-semantics map. Every status surface in the app routes through
 * these tokens — running is the accent (never a separate blue/green).
 *  running -> accent · completed -> success · error -> danger ·
 *  pending -> warning · idle/stopped -> fg-muted
 */
const STATUS: Record<StatusKind, { text: string; dot: string; live?: boolean }> = {
    running: { text: 'text-accent', dot: 'bg-accent', live: true },
    completed: { text: 'text-success', dot: 'bg-success' },
    error: { text: 'text-danger', dot: 'bg-danger' },
    pending: { text: 'text-warning', dot: 'bg-warning' },
    idle: { text: 'text-fg-muted', dot: 'bg-fg-muted' },
};

/**
 * The ONE domain-status → StatusKind mapper. Takes the backend's project
 * status string (any case): INITIALIZING → pending, RUNNING → running,
 * COMPLETED → completed, FAILED → error, everything else (CREATED / STOPPED /
 * unknown) → idle.
 */
// eslint-disable-next-line react-refresh/only-export-components -- the mapper belongs beside the pill's STATUS map
export function toStatusKind(status: string): StatusKind {
    switch (status?.toUpperCase()) {
        case 'INITIALIZING':
            return 'pending';
        case 'RUNNING':
            return 'running';
        case 'COMPLETED':
            return 'completed';
        case 'FAILED':
            return 'error';
        default:
            return 'idle';
    }
}

export interface StatusPillProps {
    status: StatusKind;
    /** Label text. Defaults to the capitalised status. */
    children?: ReactNode;
    className?: string;
}

export function StatusPill({ status, children, className }: StatusPillProps) {
    const tone = STATUS[status];
    const label = children ?? status.charAt(0).toUpperCase() + status.slice(1);
    return (
        <span
            className={cn(
                'inline-flex items-center gap-1.5 rounded-pill px-2.5 py-0.5',
                'bg-surface-2 border border-hairline',
                'text-caption font-medium',
                tone.text,
                className,
            )}
        >
            <span
                className={cn('h-1.5 w-1.5 rounded-pill', tone.dot, tone.live && 'dot-pulse')}
                aria-hidden
            />
            {label}
        </span>
    );
}

export default StatusPill;
