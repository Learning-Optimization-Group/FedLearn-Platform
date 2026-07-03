import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { CreateProjectModalV2 } from './CreateProjectModal';
import * as api from '../../services/apiServices';

vi.mock('../../services/apiServices');

const RECIPE: api.ModelRecipe = {
  key: 'CNN',
  displayName: 'Image model (CNN)',
  inputKind: 'image',
  classes: [],
  baseModels: ['net'],
  optimizers: ['Adam'],
};

/** Minimal AxiosResponse wrapper — the modal only ever reads `.data`. */
function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

function project(status: api.Project['status']): api.Project {
  return { id: 'p1', name: 'My proj', modelType: 'CNN', modelName: 'net', optimizer: 'Adam', status };
}

async function fillAndSubmit(onSubmit: ReturnType<typeof vi.fn>) {
  const onCreated = vi.fn();
  const onClose = vi.fn();
  render(<CreateProjectModalV2 isOpen onSubmit={onSubmit} onCreated={onCreated} onClose={onClose} />);
  // Wait for the recipe catalog to populate the picker, then fill the required name.
  await screen.findByRole('option', { name: /Image model/i });
  fireEvent.change(screen.getByPlaceholderText(/My first model/i), { target: { value: 'My proj' } });
  fireEvent.click(screen.getByRole('button', { name: /create project/i }));
  return { onCreated, onClose };
}

describe('CreateProjectModal — async init polling (BA-1)', () => {
  beforeEach(() => {
    vi.mocked(api.fetchModelRecipes).mockResolvedValue(resp([RECIPE]));
  });

  it('shows "Preparing" then closes + refreshes once the project is ready', async () => {
    const onSubmit = vi.fn().mockResolvedValue(project('INITIALIZING'));
    vi.mocked(api.fetchProject).mockResolvedValue(resp(project('CREATED')));

    const { onCreated, onClose } = await fillAndSubmit(onSubmit);

    expect(await screen.findByText(/Preparing your model/i)).toBeInTheDocument();
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1), { timeout: 4000 });
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(api.fetchProject).toHaveBeenCalledWith('p1');
  });

  it('surfaces an error and stays open when init fails', async () => {
    const onSubmit = vi.fn().mockResolvedValue(project('INITIALIZING'));
    vi.mocked(api.fetchProject).mockResolvedValue(resp(project('FAILED')));

    const { onCreated, onClose } = await fillAndSubmit(onSubmit);

    expect(await screen.findByText(/Model preparation failed/i, {}, { timeout: 4000 })).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    expect(onCreated).toHaveBeenCalledTimes(1);   // list still refreshed so the failed project appears
  });

  it('closes immediately without polling when the project is already ready', async () => {
    const onSubmit = vi.fn().mockResolvedValue(project('CREATED'));

    const { onCreated, onClose } = await fillAndSubmit(onSubmit);

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(onCreated).toHaveBeenCalledTimes(1);
    expect(api.fetchProject).not.toHaveBeenCalled();
  });
});
