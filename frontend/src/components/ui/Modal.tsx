import { useEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { cn } from '../../lib/utils';
import { useFocusTrap } from '../../hooks/useFocusTrap';

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
 * The single dialog pattern. Backdrop is the scrim token; panel is
 * bg-surface-1 + hairline + rounded-card + the overlay shadow and enters with a
 * fade + 2px translate. Closes on backdrop click and Escape. Renders to
 * document.body. Footer actions are right-aligned, natural width, Cancel
 * before Primary.
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
    const panelRef = useRef<HTMLDivElement>(null);

    // Trap focus inside the panel while open; restore it to the trigger on close.
    useFocusTrap(open, panelRef);

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
                'bg-scrim',
                'transition-opacity duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)]',
                mounted ? 'opacity-100' : 'opacity-0',
            )}
            onClick={(e) => {
                if (e.target === e.currentTarget) onClose();
            }}
        >
            <div
                ref={panelRef}
                tabIndex={-1}
                className={cn(
                    'relative w-full bg-surface-1 border border-hairline rounded-card',
                    'shadow-overlay',
                    'transition-[opacity,transform] duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)]',
                    mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-2',
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
                                className={cn(
                                    'flex h-8 w-8 items-center justify-center rounded-md -mr-2 -mt-1',
                                    'text-fg-muted hover:text-fg hover:bg-surface-2 transition-colors duration-[120ms]',
                                    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent',
                                )}
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
