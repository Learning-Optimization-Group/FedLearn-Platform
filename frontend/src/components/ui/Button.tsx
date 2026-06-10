import { forwardRef, type ButtonHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

const VARIANTS: Record<ButtonVariant, string> = {
    // The ember CTA: the flat ember accent (the same warm tone as "Learn" in the
    // wordmark) with a faint sheen + soft glow. Brightens on hover.
    primary: cn(
        'text-accent-fg bg-accent',
        'shadow-[0_1px_0_0_rgba(255,255,255,0.16)_inset,0_8px_24px_-10px_rgba(247,163,92,0.55)]',
        'hover:bg-accent hover:brightness-[1.07] hover:shadow-[0_1px_0_0_rgba(255,255,255,0.22)_inset,0_12px_32px_-10px_rgba(247,163,92,0.75)]',
    ),
    secondary: 'bg-surface-2 border border-hairline text-fg hover:bg-surface-3 hover:border-line',
    ghost: 'text-fg-muted hover:text-fg hover:bg-surface-2',
    danger: 'text-danger hover:bg-danger/10',
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
 * Action button. `primary` is the ember gradient CTA (warm glow); `secondary` is
 * a raised surface; `ghost` is borderless; `danger` is text-danger.
 * Sizes sm/md/lg. Presses with a subtle scale; 160ms transition.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
    ({ className, variant = 'primary', size = 'md', type = 'button', ...props }, ref) => (
        <button
            ref={ref}
            type={type}
            className={cn(
                'inline-flex items-center justify-center gap-2 font-medium select-none',
                'transition-[transform,box-shadow,background-color,filter,color,border-color] duration-[160ms] ease-[cubic-bezier(0.16,1,0.3,1)]',
                'active:scale-[0.98]',
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
