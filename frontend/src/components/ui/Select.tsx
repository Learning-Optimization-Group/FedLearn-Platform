import { forwardRef, type SelectHTMLAttributes } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '../../lib/utils';

export type SelectProps = SelectHTMLAttributes<HTMLSelectElement>;

/**
 * Styled native <select>. Same surface/border/height as Input, with a lucide
 * chevron overlaid on the right. Native dropdown behaviour is preserved.
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
    ({ className, children, ...props }, ref) => (
        <div className="relative w-full">
            <select
                ref={ref}
                className={cn(
                    'w-full appearance-none bg-surface-2 border border-hairline rounded-md h-9 pl-3 pr-9',
                    'text-body text-fg cursor-pointer',
                    'transition-[border-color,box-shadow,background-color] duration-[140ms]',
                    'hover:border-line',
                    'focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20',
                    'disabled:opacity-50 disabled:pointer-events-none',
                    className,
                )}
                {...props}
            >
                {children}
            </select>
            <ChevronDown
                strokeWidth={1.5}
                className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-fg-muted"
            />
        </div>
    ),
);

Select.displayName = 'Select';

export default Select;
