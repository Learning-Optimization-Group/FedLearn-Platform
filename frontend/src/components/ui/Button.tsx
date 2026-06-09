import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md';

const VARIANTS: Record<ButtonVariant, string> = {
    primary: 'bg-accent text-accent-fg hover:bg-accent-hover',
    secondary: 'bg-surface-2 border border-hairline text-fg hover:bg-surface-3',
    ghost: 'text-fg hover:bg-surface-2',
    danger: 'text-danger hover:bg-surface-2',
};

const SIZES: Record<ButtonSize, string> = {
    sm: 'h-8 px-3 text-label',
    md: 'h-9 px-4 text-body',
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant;
    size?: ButtonSize;
}

/**
 * Action button. `primary` is the single accent fill; `secondary` is a raised
 * surface; `ghost` is borderless; `danger` is text-only danger.
 * 36px default height (`md`), rounded-md, 120ms color transition.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'primary', size = 'md', type = 'button', ...props }, ref) => (
        <button
            ref={ref}
            type={type}
            className={cn(
                'inline-flex items-center justify-center gap-2 rounded-md font-medium',
                'transition-colors duration-[120ms]',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                'disabled:opacity-50 disabled:pointer-events-none',
                VARIANTS[variant],
                SIZES[size],
                className,
            )}
            {...props}
        />
    ),
);

Button.displayName = 'Button';

export default Button;
