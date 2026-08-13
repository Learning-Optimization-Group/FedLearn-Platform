/**
 * P1-4: the picker offers the training arm and shows what the choice costs — measured, not asserted.
 *
 * The frozen-vs-full choice is the one decision in project creation that the research record has a
 * direct, quantified answer for. Offering it as a bare dropdown would make the platform's own
 * measurements invisible at exactly the moment they are decision-relevant.
 *
 * These tests pin the parts that make the display honest rather than merely present: the numbers
 * come from the catalog (so they cannot drift from the record), the caveats travel with them, and
 * a recipe that offers no choice shows no trade-off.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { CreateProjectModalV2 } from './CreateProjectModal';
import * as api from '../../services/apiServices';

vi.mock('../../services/apiServices');

const TRADEOFF: NonNullable<api.ModelRecipe['armTradeoff']> = {
  headline: 'Full fine-tuning buys +0.0224 AUC for 3,321x the communication. '
    + 'Defensible in a datacenter, not on a phone.',
  commRatio: 3321,
  ondeviceRatio: 5988,
  measuredOn: {
    task: 'chest X-ray pneumonia, binary AUC',
    backbone: 'timm resnet50_gn.a1h_in1k (identical for both arms)',
    protocol: '400 rounds, 3 seeds, alpha=1.0, 20 clients, 10/round, 3 local epochs',
  },
  arms: {
    FULL: { accuracyAuc: 0.9786, commTotalMb400r: 752502.5, ondeviceFeasible: false,
            summary: 'Higher accuracy. Needs a datacenter — 44.8 s per on-device step.' },
    FROZEN_HEAD: { accuracyAuc: 0.9562, commTotalMb400r: 226.6, ondeviceFeasible: true,
                   summary: '3,321x less communication and the only arm that runs on-device.' },
  },
  caveats: [
    'One task (chest X-ray), one alpha (1.0), three seeds per cell.',
    'The communication ratio is ROUND-BUDGET DEPENDENT. Quoted at 400 rounds.',
    'Accuracy and on-device latency were measured on DIFFERENT hardware.',
  ],
};

const DUAL_ARM: api.ModelRecipe = {
  key: 'PNEUMONIA_CNN',
  displayName: 'Pneumonia CNN',
  inputKind: 'image',
  classes: ['NORMAL', 'PNEUMONIA'],
  baseModels: ['pneumonia_cnn'],
  optimizers: ['Adam'],
  supportedArms: ['FULL', 'FROZEN_HEAD'],
  armTradeoff: TRADEOFF,
};

const SINGLE_ARM: api.ModelRecipe = {
  key: 'MLP',
  displayName: 'Tabular model (MLP)',
  inputKind: 'tabular',
  classes: [],
  baseModels: ['mlp'],
  optimizers: ['Adam'],
  supportedArms: ['FULL'],
};

function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

function project(): api.Project {
  return { id: 'p1', name: 'p', modelType: 'PNEUMONIA_CNN', modelName: 'pneumonia_cnn',
           optimizer: 'Adam', status: 'CREATED' };
}

async function open(recipes: api.ModelRecipe[]) {
  const onSubmit = vi.fn().mockResolvedValue(project());
  vi.mocked(api.fetchModelRecipes).mockResolvedValue(resp(recipes));
  vi.mocked(api.fetchProject).mockResolvedValue(resp(project()));
  render(<CreateProjectModalV2 isOpen onSubmit={onSubmit} onCreated={vi.fn()} onClose={vi.fn()} />);
  await waitFor(() => expect(screen.getByLabelText(/what kind of model/i)).toBeInTheDocument());
  return onSubmit;
}

beforeEach(() => vi.clearAllMocks());

describe('CreateProjectModal — training arm', () => {
  it('offers the arm for a recipe that supports both', async () => {
    await open([DUAL_ARM]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    const select = await screen.findByLabelText(/training arm/i);
    expect(select).toBeInTheDocument();
    expect(within(select as HTMLSelectElement).map((o) => o.value).sort())
      .toEqual(['FROZEN_HEAD', 'FULL']);
  });

  it('does not offer a choice when the recipe supports one arm', async () => {
    await open([SINGLE_ARM]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'MLP' } });
    await waitFor(() => expect(screen.getByLabelText(/what kind of model/i)).toHaveValue('MLP'));
    expect(screen.queryByLabelText(/training arm/i)).not.toBeInTheDocument();
  });

  it('defaults to FULL, preserving pre-P1 behaviour', async () => {
    await open([DUAL_ARM]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    expect(await screen.findByLabelText(/training arm/i)).toHaveValue('FULL');
  });

  it('shows the measured trade-off headline next to the choice', async () => {
    await open([DUAL_ARM]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    expect(await screen.findByText(/\+0\.0224 AUC/)).toBeInTheDocument();
    expect(screen.getByText(/3,321x the communication/)).toBeInTheDocument();
  });

  it('carries the caveats, so no number is shown as an unqualified claim', async () => {
    await open([DUAL_ARM]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    expect(await screen.findByText(/ROUND-BUDGET DEPENDENT/i)).toBeInTheDocument();
    expect(screen.getByText(/DIFFERENT hardware/i)).toBeInTheDocument();
  });

  it('warns that the full arm is not feasible on-device', async () => {
    // 44.8 s per on-device step is the finding that decides the on-device question. It must reach
    // the user as a warning at the moment of choosing, not as a number to interpret later.
    await open([DUAL_ARM]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    expect(await screen.findByLabelText(/training arm/i)).toHaveValue('FULL');
    expect(screen.getByText(/not feasible on-device/i)).toBeInTheDocument();
  });

  it('drops that warning once the frozen arm is selected', async () => {
    await open([DUAL_ARM]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    fireEvent.change(await screen.findByLabelText(/training arm/i),
                     { target: { value: 'FROZEN_HEAD' } });
    await waitFor(() =>
      expect(screen.queryByText(/not feasible on-device/i)).not.toBeInTheDocument());
    // The arm's own measured summary takes over.
    expect(screen.getByText(/runs on-device/i)).toBeInTheDocument();
  });

  it('sends the chosen arm on submit', async () => {
    const onSubmit = await open([DUAL_ARM]);
    fireEvent.change(screen.getByLabelText(/project name/i), { target: { value: 'Frozen run' } });
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    fireEvent.change(await screen.findByLabelText(/training arm/i),
                     { target: { value: 'FROZEN_HEAD' } });
    fireEvent.click(screen.getByRole('button', { name: /create project/i }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0]).toMatchObject({ trainingArm: 'FROZEN_HEAD' });
  });

  it('omits the arm entirely for a single-arm recipe', async () => {
    const onSubmit = await open([SINGLE_ARM]);
    fireEvent.change(screen.getByLabelText(/project name/i), { target: { value: 'Plain' } });
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'MLP' } });
    fireEvent.click(screen.getByRole('button', { name: /create project/i }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0].trainingArm).toBeUndefined();
  });

  it('still renders the choice when the catalog carries no trade-off', async () => {
    // The measurement is untracked and generated; its absence must cost the user an explanation,
    // not the ability to pick an arm.
    await open([{ ...DUAL_ARM, armTradeoff: undefined }]);
    fireEvent.change(screen.getByLabelText(/what kind of model/i), { target: { value: 'PNEUMONIA_CNN' } });
    expect(await screen.findByLabelText(/training arm/i)).toBeInTheDocument();
    expect(screen.queryByText(/\+0\.0224 AUC/)).not.toBeInTheDocument();
  });
});

/** Options of a <select>, as a plain array. */
function within(select: HTMLSelectElement) {
  return Array.from(select.options);
}
