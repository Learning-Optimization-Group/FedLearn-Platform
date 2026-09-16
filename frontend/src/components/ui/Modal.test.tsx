import { describe, it, expect } from 'vitest';
import { useState } from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Modal } from './Modal';

/**
 * FE-13 — focus management for the shared dialog primitive.
 *
 * A trigger button + a Modal with two extra focusable controls. With `showClose`
 * on (the default), the panel's focusable order is:
 *   [Close, First action, Last action]
 * so `first` is the Close button and `last` is "Last action".
 */
function Harness() {
    const [open, setOpen] = useState(false);
    return (
        <div>
            <button onClick={() => setOpen(true)}>Open dialog</button>
            <Modal open={open} onClose={() => setOpen(false)} title="Test Dialog">
                <button>First action</button>
                <button>Last action</button>
            </Modal>
        </div>
    );
}

/** Focus the trigger, then click it — so the trap records a real trigger to restore to. */
function openViaTrigger() {
    const trigger = screen.getByRole('button', { name: /open dialog/i });
    trigger.focus();
    expect(trigger).toHaveFocus();
    fireEvent.click(trigger);
    return trigger;
}

describe('Modal — focus management (FE-13)', () => {
    it('moves focus into the dialog when it opens', () => {
        render(<Harness />);
        openViaTrigger();

        const dialog = screen.getByRole('dialog');
        // Focus landed inside the dialog (on the first focusable — the Close button),
        // not left on the background trigger.
        expect(dialog.contains(document.activeElement)).toBe(true);
        expect(screen.getByRole('button', { name: /close/i })).toHaveFocus();
    });

    it('traps Tab / Shift+Tab within the dialog, wrapping at both ends', () => {
        render(<Harness />);
        openViaTrigger();

        const closeBtn = screen.getByRole('button', { name: /close/i });     // first
        const lastBtn = screen.getByRole('button', { name: /last action/i }); // last

        // Focus starts on the first focusable.
        expect(closeBtn).toHaveFocus();

        // Shift+Tab from the first wraps to the last.
        fireEvent.keyDown(document.activeElement as HTMLElement, { key: 'Tab', shiftKey: true });
        expect(lastBtn).toHaveFocus();

        // Tab from the last wraps back to the first.
        fireEvent.keyDown(document.activeElement as HTMLElement, { key: 'Tab' });
        expect(closeBtn).toHaveFocus();
    });

    it('restores focus to the trigger when closed via the close button', async () => {
        render(<Harness />);
        const trigger = openViaTrigger();

        fireEvent.click(screen.getByRole('button', { name: /close/i }));

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(trigger).toHaveFocus();
    });

    it('closes on Escape and restores focus to the trigger', async () => {
        render(<Harness />);
        const trigger = openViaTrigger();

        // Modal listens for Escape on window; a keydown on the focused element bubbles up.
        fireEvent.keyDown(document.activeElement as HTMLElement, { key: 'Escape' });

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(trigger).toHaveFocus();
    });

    it('closes on backdrop click and restores focus to the trigger', async () => {
        render(<Harness />);
        const trigger = openViaTrigger();

        // The backdrop is the dialog element itself; clicking it (target === currentTarget) closes.
        fireEvent.click(screen.getByRole('dialog'));

        await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
        expect(trigger).toHaveFocus();
    });
});
