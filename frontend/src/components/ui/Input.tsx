import { forwardRef, type InputHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type InputProps = InputHTMLAttributes<HTMLInputElement>;

/**
 * Text input. bg-surface-2 + hairline border + rounded-sm, 36px tall (h-9),
 * focus ring collapses to an accent border.
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
    ({ className, type = 'text', ...props }, ref) => (
        <input
            ref={ref}
            type={type}
            className={cn(
                'w-full bg-surface-2 border border-hairline rounded-sm h-9 px-3',
                'text-body text-fg placeholder:text-fg-subtle',
                'transition-colors duration-[120ms]',
                'focus:outline-none focus:border-accent',
                'disabled:opacity-50 disabled:pointer-events-none',
                className,
            )}
            {...props}
        />
    ),
);

Input.displayName = 'Input';

export default Input;
