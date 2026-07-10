import { useEffect, type RefObject } from 'react';

/**
 * Standard focusable-element selector. Excludes disabled controls and anything
 * explicitly removed from the tab order (`tabindex="-1"`). The trap container
 * itself carries `tabindex="-1"` so it is a valid *programmatic* focus target
 * (the zero-focusable fallback) without appearing in this query.
 */
const FOCUSABLE_SELECTOR = [
    'a[href]',
    'button:not([disabled])',
    'textarea:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
].join(', ');

/**
 * Traps keyboard focus inside `containerRef` while `active` is true.
 *
 * On activation it records the currently-focused element, moves focus into the
 * container (first focusable, or the container itself when none exist), and
 * installs a Tab / Shift+Tab handler that keeps focus cycling within the
 * container — wrapping at both ends. On deactivation (or unmount) it removes the
 * handler and restores focus to the element that had it before activation.
 *
 * Intended for the modal-dialog pattern: pass `active={open}` and a ref to the
 * dialog panel (which must have `tabIndex={-1}` for the empty fallback).
 */
export function useFocusTrap<T extends HTMLElement>(
    active: boolean,
    containerRef: RefObject<T | null>,
): void {
    useEffect(() => {
        if (!active) return;
        const container = containerRef.current;
        if (!container) return;

        // Remember the trigger so focus can be handed back when the trap lifts.
        const previouslyFocused = document.activeElement as HTMLElement | null;

        const getFocusable = (): HTMLElement[] =>
            Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));

        // Move focus into the dialog on open.
        const initial = getFocusable();
        if (initial.length > 0) {
            initial[0].focus();
        } else {
            container.focus();
        }

        const onKeyDown = (e: KeyboardEvent): void => {
            if (e.key !== 'Tab') return;
            const items = getFocusable();
            if (items.length === 0) {
                // Nothing tabbable — keep focus pinned to the container.
                e.preventDefault();
                container.focus();
                return;
            }
            const first = items[0];
            const last = items[items.length - 1];
            const activeEl = document.activeElement;
            const inside = container.contains(activeEl);

            if (e.shiftKey) {
                if (activeEl === first || !inside) {
                    e.preventDefault();
                    last.focus();
                }
            } else if (activeEl === last || !inside) {
                e.preventDefault();
                first.focus();
            }
        };

        document.addEventListener('keydown', onKeyDown);
        return () => {
            document.removeEventListener('keydown', onKeyDown);
            // Restore focus to the trigger that opened the dialog.
            if (previouslyFocused && typeof previouslyFocused.focus === 'function') {
                previouslyFocused.focus();
            }
        };
    }, [active, containerRef]);
}

export default useFocusTrap;
