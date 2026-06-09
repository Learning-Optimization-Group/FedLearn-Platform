import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';
import { cn } from '../../lib/utils';

export interface LogConsoleProps extends HTMLAttributes<HTMLDivElement> {
    children: ReactNode;
}

/**
 * Streaming log surface. bg-code-well + hairline + rounded-card, mono text-label,
 * scrollable, no animation. Consumers render their own line rows inside.
 * Forwarded ref lets callers manage auto-scroll-to-bottom.
 */
export const LogConsole = forwardRef<HTMLDivElement, LogConsoleProps>(
    ({ className, children, ...props }, ref) => (
        <div
            ref={ref}
            className={cn(
                'bg-code-well border border-hairline rounded-card',
                'font-mono text-label text-fg',
                'overflow-auto p-3 leading-relaxed',
                className,
            )}
            {...props}
        >
            {children}
        </div>
    ),
);

LogConsole.displayName = 'LogConsole';

export default LogConsole;
