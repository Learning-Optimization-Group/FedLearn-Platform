import { useEffect, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { cn } from '../../lib/utils';

export interface ModalProps {
    open: boolean;
    onClose: () => void;
    /** Optional header title. Omit for fully custom content. */
    title?: ReactNode;
    children: ReactNode;
    /** Footer region (typically actions). */
    footer?: ReactNode;
    /** Constrains panel width. Defaults to `md`. */
    size?: 'sm' | 'md' | 'lg';
    /** Show the top-right close affordance. Defaults to true. */
    showClose?: boolean;
    className?: string;
}

const SIZES = {
    sm: 'max-w-sm',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
};

/**
 * The single dialog pattern. Backdrop is bg-canvas/70 + backdrop-blur; panel is
 * bg-surface-1 + hairline + rounded-card and enters with a fade + 2px translate
 * over 160ms. Closes on backdrop click and Escape. Renders to document.body.
 */
export function Modal({
    open,
    onClose,
    title,
    children,
    footer,
    size = 'md',
    showClose = true,
    className,
}: ModalProps) {
    const [mounted, setMounted] = useState(false);

    // Drive the enter transition on the frame after mount.
    useEffect(() => {
        if (!open) {
            setMounted(false);
            return;
        }
        const raf = requestAnimationFrame(() => setMounted(true));
        return () => cancelAnimationFrame(raf);
    }, [open]);

    useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open, onClose]);

    if (!open) return null;

    return createPortal(
        <div
            role="dialog"
            aria-modal="true"
            className={cn(
                'fixed inset-0 z-50 flex items-center justify-center p-4',
                'bg-canvas/70 backdrop-blur-sm',
                'transition-opacity duration-[160ms] ease-[cubic-bezier(0.16,1,0.3,1)]',
                mounted ? 'opacity-100' : 'opacity-0',
            )}
            onClick={(e) => {
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <div
                className={cn(
                    'w-full bg-surface-1 border border-hairline rounded-card',
                    'transition-[opacity,transform] duration-[160ms] ease-[cubic-bezier(0.16,1,0.3,1)]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-[2px]',
                    SIZES[size],
                    className,
                )}
                onClick={(e) => e.stopPropagation()}
            >
                {(title || showClose) && (
                    <div className="flex items-center justify-between gap-4 px-6 pt-5 pb-4">
                        {title ? (
                            <h2 className="text-h4 text-fg">{title}</h2>
                        ) : (
                            <span />
                        )}
                        {showClose && (
                            <button
                                type="button"
                                aria-label="Close"
                                onClick={onClose}
                                className="text-fg-muted hover:text-fg transition-colors duration-[120ms] -mr-1"
                            >
                                <X strokeWidth={1.5} className="h-5 w-5" />
                            </button>
                        )}
                    </div>
                )}
                <div className="px-6 pb-6 pt-1">{children}</div>
                {footer && (
                    <div className="flex items-center justify-end gap-2 px-6 pb-5 pt-1">
                        {footer}
                    </div>
                )}
            </div>
        </div>,
        document.body,
    );
}

export default Modal;
