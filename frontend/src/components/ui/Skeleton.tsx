import { type HTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type SkeletonProps = HTMLAttributes<HTMLDivElement>;

/**
 * Loading placeholder. bg-surface-2 + animate-pulse + rounded-md. Size it with
 * className (e.g. `h-4 w-32`).
 */
export function Skeleton({ className, ...props }: SkeletonProps) {
    return (
        <div
            className={cn('rounded-md bg-surface-2 animate-pulse', className)}
            {...props}
        />
    );
}

export default Skeleton;
