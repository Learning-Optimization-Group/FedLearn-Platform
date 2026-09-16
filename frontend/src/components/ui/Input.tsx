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
                'w-full bg-surface-2 border border-hairline rounded-md h-9 px-3',
                'text-body text-fg placeholder:text-fg-subtle',
                'transition-[border-color,box-shadow,background-color] duration-[140ms]',
                'hover:border-line',
                'focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20',
                'disabled:opacity-50 disabled:pointer-events-none',
                className,
            )}
            {...props}
        />
    ),
);

Input.displayName = 'Input';

export default Input;
