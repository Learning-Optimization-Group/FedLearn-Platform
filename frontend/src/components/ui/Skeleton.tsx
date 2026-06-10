import { type HTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type SkeletonProps = HTMLAttributes<HTMLDivElement>;

/**
 * Loading placeholder. bg-surface-2 + animate-pulse + rounded-sm. Size it with
 * className (e.g. `h-4 w-32`).
 */
export function Skeleton({ className, ...props }: SkeletonProps) {
    return (
        <div
            className={cn(
                'relative overflow-hidden rounded-md bg-surface-2',
                'after:absolute after:inset-0 after:-translate-x-full',
                'after:bg-gradient-to-r after:from-transparent after:via-white/[0.06] after:to-transparent',
                'after:animate-[ember-shimmer_1.6s_ease-in-out_infinite]',
                className,
            )}
            {...props}
        />
    );
}

export default Skeleton;
