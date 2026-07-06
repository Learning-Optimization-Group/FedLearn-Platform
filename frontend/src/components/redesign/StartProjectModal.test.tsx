import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { StartProjectModal } from './StartProjectModal';
import type { Project } from '../../services/apiServices';

const PROJECT: Project = {
  id: 'p1',
  name: 'Fraud model',
  modelType: 'CNN',
  modelName: 'net',
  optimizer: 'Adam',
  status: 'CREATED',
};

/** The shape axios errors take when the backend returns a message body. */
function backendError(message: string): Error {
  return Object.assign(new Error(message), {
    isAxiosError: true,
    response: { data: { message } },
  });
}

// FE-4: a failed start must keep the modal open and surface the backend detail
// inline — not close silently or swallow the error behind the route beneath.
describe('StartProjectModal — start failures stay visible (FE-4)', () => {
  it('keeps the modal open and shows the backend detail inline when starting fails', async () => {
    const onSubmit = vi.fn().mockRejectedValue(
      backendError('No free port available in the 50000-50010 range.'),
    );
    const onClose = vi.fn();
    render(<StartProjectModal isOpen project={PROJECT} onClose={onClose} onSubmit={onSubmit} />);

    fireEvent.click(screen.getByRole('button', { name: /start training/i }));

    const dialog = await screen.findByRole('dialog');
    expect(
      await within(dialog).findByText('No free port available in the 50000-50010 range.'),
    ).toBeInTheDocument();
    // Still open, not closed/swallowed: the form and its submit are still there.
    expect(onClose).not.toHaveBeenCalled();
    expect(within(dialog).getByRole('button', { name: /start training/i })).toBeEnabled();
  });

  it('falls back to a readable message when the failure carries no backend detail', async () => {
    const onSubmit = vi.fn().mockRejectedValue(new Error('network down'));
    const onClose = vi.fn();
    render(<StartProjectModal isOpen project={PROJECT} onClose={onClose} onSubmit={onSubmit} />);

    fireEvent.click(screen.getByRole('button', { name: /start training/i }));

    expect(await screen.findByText('Could not start training. Please try again.')).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });
});
