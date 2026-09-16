import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

const VARIANTS: Record<ButtonVariant, string> = {
    // Navy structural ink — the one primary action per view.
    primary: 'text-accent-fg bg-accent hover:bg-accent-hover',
    secondary: 'bg-surface-1 border border-hairline text-fg hover:bg-surface-2 hover:border-line',
    ghost: 'text-fg-muted hover:text-fg hover:bg-surface-2',
    // Solid destructive fill — a destructive confirm must never read weaker
    // than Cancel. Low-emphasis destructive entry points should use ghost +
    // a danger text class instead of this variant.
    danger: 'text-accent-fg bg-danger hover:brightness-95',
};

const SIZES: Record<ButtonSize, string> = {
    sm: 'h-8 px-3 text-label rounded-md',
    md: 'h-9 px-4 text-body rounded-md',
    lg: 'h-11 px-6 text-body-lg font-semibold rounded-lg',
};

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: ButtonVariant;
    size?: ButtonSize;
}

/**
 * Action button. `primary` is the navy CTA; `secondary` is a bordered surface;
 * `ghost` is borderless; `danger` is a solid destructive fill.
 * Sizes sm/md/lg. Flat fills, no glow, no press-scale.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'primary', size = 'md', type = 'button', ...props }, ref) => (
        <button
            ref={ref}
            type={type}
            className={cn(
                'inline-flex items-center justify-center gap-2 font-medium select-none',
                'transition-[background-color,color,border-color,filter] duration-[120ms] ease-[cubic-bezier(0.4,0,0.2,1)]',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 focus-visible:ring-offset-canvas',
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
