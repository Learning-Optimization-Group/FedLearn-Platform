import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ResultsModalV2 } from './ResultsModal';

// FE-13: ResultsModal is a standalone dialog (not the shared Modal primitive), so it needs the
// same a11y contract on its own: a labelled dialog role, focus moved in + trapped, Escape and
// backdrop-click close. results=[] renders the empty state (no recharts), keeping this jsdom-safe.
describe('ResultsModalV2 — dialog a11y (FE-13)', () => {
  const base = { projectName: 'Fraud model', results: [] };

  it('is a labelled modal dialog and moves focus into itself on open', () => {
    render(<ResultsModalV2 isOpen onClose={() => {}} {...base} />);
    const dialog = screen.getByRole('dialog', { name: 'Fraud model — Results' });
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog.contains(document.activeElement)).toBe(true); // focus trapped inside
  });

  it('closes on Escape', () => {
    const onClose = vi.fn();
    render(<ResultsModalV2 isOpen onClose={onClose} {...base} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes on backdrop click but not on a click inside the panel', () => {
    const onClose = vi.fn();
    render(<ResultsModalV2 isOpen onClose={onClose} {...base} />);
    const dialog = screen.getByRole('dialog');
    fireEvent.click(dialog); // inside the panel -> no close
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(dialog.parentElement as HTMLElement); // the backdrop
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('renders nothing when closed', () => {
    const { container } = render(<ResultsModalV2 isOpen={false} onClose={() => {}} {...base} />);
    expect(container).toBeEmptyDOMElement();
  });
});
