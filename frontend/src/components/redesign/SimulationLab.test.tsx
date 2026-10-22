import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { AxiosError, type InternalAxiosRequestConfig } from 'axios';
import api from '../../api/axiosConfig';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';
import { recipe, limits, workload, simulationRun, round, runA, runB } from '../../test/simulationFixtures';
import { SimulationLab } from './SimulationLab';
vi.mock('../../context/AuthContext', async (original) => ({ ...await original<typeof import('../../context/AuthContext')>(), useAuth: vi.fn() }));
const oldAdapter = api.defaults.adapter;
let requests: { method: string; url: string; data: unknown }[];
let available = true;
let current = simulationRun();
let override: ((config: InternalAxiosRequestConfig) => Promise<unknown> | undefined) | undefined;
const owner = (username = 'owner') => makeAuth({ currentUser: { username, email: `${username}@example.com`, role: 'PROJECT_OWNER' }, isOwner: true });
beforeEach(() => {
  vi.mocked(useAuth).mockReturnValue(owner()); requests = []; available = true;
  current = simulationRun(); override = undefined;
  api.defaults.adapter = async (config) => {
    requests.push({ method: config.method!, url: config.url!, data: config.data ? JSON.parse(config.data as string) : undefined });
    const custom = override?.(config);
    const data = custom ? await custom : config.url?.endsWith('/workloads') ? { workloads: [{ ...workload, ready: available, reason: available ? null : 'Qualified workload unavailable.' }], scope: limits.scope }
      : config.url === '/simulations/resolve' ? { workloadId: 'nih-qualified', configHash: 'b'.repeat(64), recipe, limits, scope: limits.scope }
      : config.url === '/simulations/runs' && config.method === 'post' ? { id: runA, status: 'STARTING', configHash: current.configHash, recipe, createdAt: current.createdAt, scope: current.scope }
      : config.url === '/simulations/runs' ? { content: [current, simulationRun({ id: runB })], page: 0, size: 20, totalElements: 2, scope: limits.scope }
      : config.url?.endsWith('/stop') ? { ...current, status: 'STOPPING', qualification: null }
      : config.url?.endsWith('/record') ? { runId: current.id, configHash: current.configHash, researchRecordDpSafe: false, execution: { wallSeconds: 1.2, peakRssBytes: 1048576, peakRssScope: 'run', scope: 'Measured engine scope' }, source: { runtime: { git_commit: 'abcdef123456', python: '3.12', torch: '2.12' } }, scope: current.scope }
      : { ...current, id: config.url?.endsWith(runB) ? runB : runA };
    return { data, status: 200, statusText: 'OK', headers: {}, config };
  };
});
afterEach(() => { api.defaults.adapter = oldAdapter; vi.useRealTimers(); });
function show(path = '/simulations') {
  return render(<MemoryRouter initialEntries={[path]}><SimulationLab /></MemoryRouter>);
}
async function review() {
  await screen.findByText(workload.label);
  fireEvent.click(screen.getByRole('button', { name: 'Review recipe' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start simulation' })).toBeEnabled());
}

describe('Simulation Lab workload choice', () => {
  const effusion = { ...workload, id: 'nih-effusion-r0', label: 'Effusion (1) versus not (0); report-derived labels',
    representation: 'Frozen ImageNet ResNet-18 features, train-standardised; development selection half only (holdout and official test excluded)',
    splitSummary: { train: { images: 68650, patients: 22399 }, selection: { images: 9075, patients: 2800 } },
    supportedChoices: { recipe: ['effusion-clean', 'effusion-dp'] } };
  beforeEach(() => {
    override = (config) => config.url?.endsWith('/workloads') ? Promise.resolve({ workloads: [workload, effusion], scope: limits.scope }) : undefined;
  });
  it('switches to the Effusion workload with measured recipes and no free clip choice', async () => {
    show(); await screen.findByText(workload.label);
    fireEvent.change(screen.getByLabelText('Workload'), { target: { value: 'nih-effusion-r0' } });
    expect(await screen.findByText(effusion.label)).toBeInTheDocument();
    expect(screen.getByText(/9,075 development-selection images/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Client clipping norm')).toBeNull();
    fireEvent.change(screen.getByLabelText('Measured recipe'), { target: { value: 'effusion-dp' } });
    fireEvent.click(screen.getByRole('button', { name: 'Review recipe' }));
    await waitFor(() => expect(requests.some(r => r.url === '/simulations/resolve')).toBe(true));
    const sent = (requests.find(r => r.url === '/simulations/resolve')!.data as { choices: Record<string, unknown> }).choices;
    expect(sent).toMatchObject({ workloadId: 'nih-effusion-r0', recipe: 'effusion-dp' });
    expect(sent).not.toHaveProperty('clipNorm');
  });
  it('returns to the historical workload with its standard recipe and clip', async () => {
    show(); await screen.findByText(workload.label);
    fireEvent.change(screen.getByLabelText('Workload'), { target: { value: 'nih-effusion-r0' } });
    fireEvent.change(screen.getByLabelText('Workload'), { target: { value: 'nih-qualified' } });
    expect(screen.getByLabelText('Client clipping norm')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Review recipe' }));
    await waitFor(() => expect(requests.some(r => r.url === '/simulations/resolve')).toBe(true));
    expect((requests.find(r => r.url === '/simulations/resolve')!.data as { choices: unknown }).choices)
      .toMatchObject({ workloadId: 'nih-qualified', recipe: 'standard', clipNorm: .05 });
  });
});

describe('Simulation Lab workbench', () => {
  it('prevents launches when no qualified workload is available', async () => {
    available = false; show();
    expect(await screen.findByText('Qualified workload unavailable.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Review recipe' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start simulation' })).toBeDisabled();
  });
  it('requires recipe review again after any setting changes', async () => {
    show(); await review();
    fireEvent.change(screen.getByLabelText('Rounds'), { target: { value: '5' } });
    expect(screen.getByRole('button', { name: 'Start simulation' })).toBeDisabled();
    expect(requests.filter(r => r.method === 'post' && r.url === '/simulations/runs')).toHaveLength(0);
  });
  it('refuses out-of-bound rounds before sending a recipe', async () => {
    show(); await screen.findByText(workload.label);
    fireEvent.change(screen.getByLabelText('Rounds'), { target: { value: '21' } });
    expect(screen.getByRole('button', { name: 'Review recipe' })).toBeDisabled();
    expect(requests.some(r => r.url === '/simulations/resolve')).toBe(false);
  });
  it('restores an existing partial run without starting training or inventing completed rounds', async () => {
    current = simulationRun({ status: 'TIMED_OUT', completedRounds: 1, rounds: [round(1,.58)], finalSummary: round(1,.58), qualification: null });
    show(`/simulations?run=${runA}`);
    const selected = await screen.findByRole('region', { name: 'Selected simulation' });
    expect(within(selected).getByRole('status')).toHaveTextContent('Timed out');
    expect(screen.getByText('1 / 3 rounds completed')).toBeInTheDocument();
    expect(screen.getAllByText('0.580').length).toBeGreaterThan(0);
    expect(screen.queryByText('Execution qualified')).not.toBeInTheDocument();
    expect(requests.some(r => r.method === 'post')).toBe(false);
  });
  it('keeps stopping visible until the server verifies termination', async () => {
    current = simulationRun({ status: 'RUNNING', completedRounds: 1, rounds: [round(1)], finalSummary: null, qualification: null, endedAt: null, recordAvailable: false });
    const view = show(`/simulations?run=${runA}`);
    fireEvent.click(await screen.findByRole('button', { name: 'Stop simulation' }));
    expect(await screen.findByText('Stopping; waiting for the worker to exit')).toBeInTheDocument();
    expect(screen.queryByText('Stopped')).not.toBeInTheDocument();
    view.unmount();
    expect(requests.filter(r => r.url.endsWith('/stop'))).toHaveLength(1);
  });
  it('does not stop server-owned training when the page unmounts', async () => {
    const view = show(`/simulations?run=${runA}`);
    await screen.findByText('Execution qualified'); view.unmount();
    expect(requests.filter(r => r.url.endsWith('/stop'))).toHaveLength(0);
  });
  it('ignores a late result from a previously selected run', async () => {
    let release!: (value: unknown) => void;
    override = config => config.url === `/simulations/runs/${runA}` ? new Promise(resolve => { release = resolve; }) : undefined;
    show(`/simulations?run=${runA}`);
    fireEvent.click(await screen.findByRole('button', { name: `Open ${runB}` }));
    expect(await screen.findByText(`Run ${runB}`)).toBeInTheDocument();
    await act(async () => { release(simulationRun({ recipe: { ...recipe, seed: 123456789 } })); });
    expect(screen.queryByText('123456789')).not.toBeInTheDocument();
  });
  it('clears results when the authenticated identity changes', async () => {
    const view = show(`/simulations?run=${runA}`);
    await screen.findByText('Execution qualified');
    override = config => config.url === `/simulations/runs/${runA}` ? Promise.reject(new AxiosError('Not found', 'ERR_BAD_REQUEST', config, {}, { data: { message: 'Simulation not found.' }, status: 404, statusText: 'Not found', headers: {}, config })) : undefined;
    vi.mocked(useAuth).mockReturnValue(owner('different-owner'));
    view.rerender(<MemoryRouter initialEntries={[]}><SimulationLab /></MemoryRouter>);
    expect(await screen.findByText('Simulation not found.')).toBeInTheDocument();
    expect(screen.queryByText('Execution qualified')).not.toBeInTheDocument();
  });
  it('does not present the best intermediate AUROC as the final endpoint', async () => {
    show(`/simulations?run=${runA}`);
    await screen.findByText('Execution qualified');
    expect(screen.getByLabelText('Final AUROC')).toHaveTextContent('0.610');
  });
});

describe('Simulation Lab mutation and polling controls', () => {
  it('admits one start request while acceptance is pending', async () => {
    let accept!: (value: unknown) => void;
    override = config => config.method === 'post' && config.url === '/simulations/runs' ? new Promise(resolve => { accept = resolve; }) : undefined;
    show(); await review();
    fireEvent.click(screen.getByRole('button', { name: 'Start simulation' }));
    fireEvent.click(screen.getByRole('button', { name: 'Starting…' }));
    await waitFor(() => expect(requests.filter(r => r.method === 'post' && r.url === '/simulations/runs')).toHaveLength(1));
    await act(async () => accept({ id: runA, status: 'STARTING', configHash: current.configHash, recipe, createdAt: current.createdAt, scope: current.scope }));
    expect(await screen.findByRole('region', { name: 'Selected simulation' })).toBeInTheDocument();
  });
  it('retains a busy refusal and does not retry training automatically', async () => {
    override = config => config.method === 'post' && config.url === '/simulations/runs' ? Promise.reject(new AxiosError('Busy', 'ERR_BAD_REQUEST', config, {}, { data: { message: 'A worker is already active.' }, status: 409, statusText: 'Conflict', headers: {}, config })) : undefined;
    show(); await review(); fireEvent.click(screen.getByRole('button', { name: 'Start simulation' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('A worker is already active.');
    expect(screen.getByRole('button', { name: 'Start simulation' })).toBeDisabled();
    expect(requests.filter(r => r.method === 'post' && r.url === '/simulations/runs')).toHaveLength(1);
  });
  it('does not overlap polls or advance progress while a poll is pending', async () => {
    current = simulationRun({ status: 'RUNNING', completedRounds: 1, rounds: [round(1)], finalSummary: null, qualification: null, endedAt: null, recordAvailable: false });
    let polls = 0;
    let finish!: (value: unknown) => void;
    override = config => config.url === `/simulations/runs/${runA}` && ++polls > 1 ? new Promise(resolve => { finish = resolve; }) : undefined;
    vi.useFakeTimers(); show(`/simulations?run=${runA}`);
    await act(async () => {});
    expect(screen.getByText('1 / 3 rounds completed')).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1500); });
    await act(async () => { await vi.advanceTimersByTimeAsync(30000); });
    expect(polls).toBe(2);
    expect(screen.getByText('1 / 3 rounds completed')).toBeInTheDocument();
    await act(async () => finish(simulationRun({ status: 'TIMED_OUT', completedRounds: 2, rounds: [round(1), round(2)], qualification: null })));
    expect(screen.getByText('2 / 3 rounds completed')).toBeInTheDocument();
  });
  it('preserves an active state when stop fails', async () => {
    current = simulationRun({ status: 'RUNNING', qualification: null, endedAt: null, recordAvailable: false });
    override = config => config.url?.endsWith('/stop') ? Promise.reject(new Error('network failed')) : undefined;
    show(`/simulations?run=${runA}`);
    fireEvent.click(await screen.findByRole('button', { name: 'Stop simulation' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Could not request a stop');
    expect(within(screen.getByRole('region', { name: 'Selected simulation' })).getByRole('status')).toHaveTextContent('Running');
  });
  it('downloads only after the owned blob response, using the run-specific filename', async () => {
    const create = Object.getOwnPropertyDescriptor(URL, 'createObjectURL');
    const revoke = Object.getOwnPropertyDescriptor(URL, 'revokeObjectURL');
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: () => 'blob:owned-record' });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: () => {} });
    const files: string[] = [];
    const revoked: string[] = [];
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: (url: string) => revoked.push(url) });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { files.push(this.download); });
    override = config => config.url?.endsWith('/record') ? Promise.resolve(new Blob(['{"researchRecordDpSafe":false}'])) : undefined;
    try {
      show(`/simulations?run=${runA}`);
      fireEvent.click(await screen.findByRole('button', { name: 'Download research record' }));
      await waitFor(() => expect(files).toEqual([`simulation-${runA}.json`]));
      await waitFor(() => expect(revoked).toEqual(['blob:owned-record']), { timeout: 2000 });
    } finally {
      if (create) Object.defineProperty(URL, 'createObjectURL', create); else Reflect.deleteProperty(URL, 'createObjectURL');
      if (revoke) Object.defineProperty(URL, 'revokeObjectURL', revoke); else Reflect.deleteProperty(URL, 'revokeObjectURL');
    }
  });
  it('does not turn a refused record request into a browser download', async () => {
    const clicked = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    override = config => config.url?.endsWith('/record') ? Promise.reject(new AxiosError('Not found', 'ERR_BAD_REQUEST', config, {}, { data: { message: 'Record not found.' }, status: 404, statusText: 'Not found', headers: {}, config })) : undefined;
    show(`/simulations?run=${runA}`);
    fireEvent.click(await screen.findByRole('button', { name: 'Download research record' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Record not found.');
    expect(clicked).not.toHaveBeenCalled();
  });
  it('loads matching retained execution measurements and provenance', async () => {
    show(`/simulations?run=${runA}`); await screen.findByText('Execution qualified');
    fireEvent.click(screen.getByText('Recipe and provenance'));
    fireEvent.click(screen.getByRole('button', { name: 'Load execution details' }));
    expect(await screen.findByText('abcdef123456')).toBeInTheDocument();
    expect(screen.getByText('1.200 s')).toBeInTheDocument();
    expect(screen.getByText('1.0 MiB (run)')).toBeInTheDocument();
  });
  it('opens comparisons from owned history without starting another run', async () => {
    show();
    fireEvent.click(await screen.findByRole('checkbox', { name: `Compare ${runA}` }));
    fireEvent.click(screen.getByRole('checkbox', { name: `Compare ${runB}` }));
    fireEvent.click(screen.getByRole('button', { name: 'Compare selected (2)' }));
    expect(await screen.findByRole('heading', { name: 'Exploratory comparison' })).toBeInTheDocument();
    expect(requests.some(r => r.method === 'post')).toBe(false);
  });
});

it('keeps the acknowledged stopping state visible while its verification poll is pending', async () => {
  current = simulationRun({ status: 'RUNNING', completedRounds: 1, rounds: [round(1)], qualification: null, endedAt: null, recordAvailable: false });
  const view = show(`/simulations?run=${runA}`);
  await screen.findByRole('button', { name: 'Stop simulation' });
  override = config => config.url === `/simulations/runs/${runA}` ? new Promise(() => {}) : undefined;
  await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Stop simulation' })); });
  expect(within(screen.getByRole('region', { name: 'Selected simulation' })).getByRole('status')).toHaveTextContent('Stopping; waiting for the worker to exit');
  expect(screen.getByText('1 / 3 rounds completed')).toBeInTheDocument();
  view.unmount();
});

it('links directly to the packaged HTML report rather than the SPA directory fallback', async () => {
  show();
  await screen.findByText(workload.label);
  expect(screen.getByRole('link', { name: 'View completed reference study' })).toHaveAttribute('href', '/capstone/study-v1/index.html');
});
