import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, within, waitFor } from '@testing-library/react';
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

// FR-11 / UI exposure: the strategy picker offers the strategies that run end-to-end when selected.
describe('StartProjectModal — strategy options', () => {
  it('offers every end-to-end strategy including FedProx (FR-32)', () => {
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={vi.fn()} />);
    const dialog = screen.getByRole('dialog');
    const values = within(dialog)
      .getAllByRole('option')
      .map((o) => (o as HTMLOptionElement).value);
    expect(values).toEqual(
      expect.arrayContaining(['FedAvg', 'FedProx', 'DeComFL', 'FedOpt', 'Robust', 'FoT']),
    );
    // FR-32: FedProx is now exposed — the production client honors its proximal term.
    expect(values).toContain('FedProx');
  });
});

// Robust aggregation rule. The picker appears only for the Robust method and shows exactly the one
// parameter each rule reads. Robust fields must never be sent for another method: the backend rejects
// them there rather than silently ignoring them, so a user who tried Robust and switched back would
// otherwise get an error for a field they can no longer see.
describe('StartProjectModal — robust aggregation rule', () => {
  const FRACTION = 'Share of devices that may be malicious';
  const TRIM = 'Share trimmed from each end';
  const TAU = 'Clipping radius';
  const PARAM_LABELS = [FRACTION, TRIM, TAU];

  const chooseStrategy = (value: string) =>
    fireEvent.change(screen.getByLabelText('Training method'), { target: { value } });
  const chooseRule = (value: string) =>
    fireEvent.change(screen.getByLabelText('Aggregation rule'), { target: { value } });

  it('offers the rule picker only for the Robust method, with every server-side rule', () => {
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.queryByLabelText('Aggregation rule')).not.toBeInTheDocument();

    chooseStrategy('Robust');
    const values = within(screen.getByLabelText('Aggregation rule'))
      .getAllByRole('option')
      .map((o) => (o as HTMLOptionElement).value);
    expect(values).toEqual(['MEDIAN', 'TRIMMED_MEAN', 'KRUM', 'MULTI_KRUM', 'BULYAN', 'CENTERED_CLIP']);
  });

  it.each([
    ['MEDIAN', [] as string[]],
    ['TRIMMED_MEAN', [TRIM]],
    ['KRUM', [FRACTION]],
    ['MULTI_KRUM', [FRACTION]],
    ['BULYAN', [FRACTION]],
    ['CENTERED_CLIP', [TAU]],
  ])('%s shows exactly the parameter it uses', (rule, expected) => {
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={vi.fn()} />);
    chooseStrategy('Robust');
    chooseRule(rule);
    for (const label of PARAM_LABELS) {
      if (expected.includes(label)) {
        expect(screen.getByLabelText(label)).toBeInTheDocument();
      } else {
        expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
      }
    }
  });

  it('submits the chosen rule with its parameter', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={onSubmit} />);
    chooseStrategy('Robust');
    chooseRule('BULYAN');
    fireEvent.change(screen.getByLabelText(FRACTION), { target: { value: '0.2' } });
    fireEvent.change(screen.getByLabelText('Devices needed to start'), { target: { value: '20' } });
    fireEvent.click(screen.getByRole('button', { name: /start training/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('p1', {
      strategy: 'Robust', numRounds: 5, minClients: 20, robustMethod: 'BULYAN', byzantineFraction: 0.2,
    });
  });

  it('sends only the rule for median, which reads no parameter', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={onSubmit} />);
    chooseStrategy('Robust');
    chooseRule('MEDIAN');
    fireEvent.click(screen.getByRole('button', { name: /start training/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('p1', {
      strategy: 'Robust', numRounds: 5, minClients: 2, robustMethod: 'MEDIAN',
    });
  });

  it('sends no robust fields after switching back to another method', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={onSubmit} />);
    chooseStrategy('Robust');
    chooseRule('KRUM');
    fireEvent.change(screen.getByLabelText(FRACTION), { target: { value: '0.1' } });
    chooseStrategy('FedAvg');
    fireEvent.click(screen.getByRole('button', { name: /start training/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    const [, config] = onSubmit.mock.calls[0];
    expect(Object.keys(config).sort()).toEqual(['minClients', 'numRounds', 'strategy']);
  });
});

// Secure aggregation. Offered only for DeComFL, because masking exists only on DeComFL's gradient-scalar
// channel; on any other method the server flag would do nothing. The threshold appears only when it is on, and
// the dialog says plainly that phones cannot join a secure run yet.
describe('StartProjectModal — secure aggregation', () => {
  const SECURE = 'Secure aggregation';
  const THRESHOLD = 'Devices needed to rebuild the sum';

  const chooseStrategy = (value: string) =>
    fireEvent.change(screen.getByLabelText('Training method'), { target: { value } });
  const setSecure = (value: 'on' | 'off') =>
    fireEvent.change(screen.getByLabelText(SECURE), { target: { value } });
  const submit = () => fireEvent.click(screen.getByRole('button', { name: /start training/i }));

  it('is offered only for the DeComFL method', () => {
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={vi.fn()} />);
    expect(screen.queryByLabelText(SECURE)).not.toBeInTheDocument();
    chooseStrategy('Robust');
    expect(screen.queryByLabelText(SECURE)).not.toBeInTheDocument();
    chooseStrategy('DeComFL');
    expect(screen.getByLabelText(SECURE)).toBeInTheDocument();
  });

  it('shows the threshold only once it is on, and warns that phones cannot join', () => {
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={vi.fn()} />);
    chooseStrategy('DeComFL');
    expect(screen.queryByLabelText(THRESHOLD)).not.toBeInTheDocument();
    setSecure('on');
    expect(screen.getByLabelText(THRESHOLD)).toBeInTheDocument();
    expect(screen.getByText(/phones can.?t join/i)).toBeInTheDocument();
  });

  it('submits secure aggregation with its threshold', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={onSubmit} />);
    chooseStrategy('DeComFL');
    setSecure('on');
    fireEvent.change(screen.getByLabelText(THRESHOLD), { target: { value: '3' } });
    fireEvent.change(screen.getByLabelText('Devices needed to start'), { target: { value: '5' } });
    submit();

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit).toHaveBeenCalledWith('p1', {
      strategy: 'DeComFL', numRounds: 5, minClients: 5, secureAggregation: true, secureAggThreshold: 3,
    });
  });

  it('sends no secure-aggregation fields while it is off', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={onSubmit} />);
    chooseStrategy('DeComFL');
    submit();

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(Object.keys(onSubmit.mock.calls[0][1]).sort()).toEqual(['minClients', 'numRounds', 'strategy']);
  });

  it('sends no secure-aggregation fields after switching to another method', async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(<StartProjectModal isOpen project={PROJECT} onClose={vi.fn()} onSubmit={onSubmit} />);
    chooseStrategy('DeComFL');
    setSecure('on');
    chooseStrategy('FedAvg');
    submit();

    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(Object.keys(onSubmit.mock.calls[0][1]).sort()).toEqual(['minClients', 'numRounds', 'strategy']);
  });
});
