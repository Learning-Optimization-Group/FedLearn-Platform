import type { ReactNode } from 'react';
import { beforeEach, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

vi.mock('../../services/apiServices', () => ({
  fetchBenchmarkOverview: vi.fn(), fetchProjectBenchmark: vi.fn(), errorMessage: () => 'Request failed',
}));
vi.mock('recharts', () => {
  const container = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
  const chart = ({ children }: { children?: ReactNode }) => <div data-testid="benchmark-chart">{children}</div>;
  const empty = () => null;
  return { ResponsiveContainer: container, LineChart: chart, BarChart: chart, Line: empty, Bar: empty,
    XAxis: empty, YAxis: empty, Tooltip: empty, CartesianGrid: empty, Legend: empty };
});
import * as api from '../../services/apiServices';
import { BenchmarkDashboard } from './BenchmarkDashboard';

const base = { projectId: 'project-1', projectName: 'Recorded project', modelType: 'CNN',
  taskType: 'SEQ_CLASSIFICATION', roundsCompleted: 1, finalAccuracy: .8, bestAccuracy: .8, finalLoss: .2 };

beforeEach(() => vi.clearAllMocks());

function prepare(runId: string | null, verified: boolean) {
  const summary = { ...base, runId, runIdentityVerified: verified, roundsToTarget: 17 };
  vi.mocked(api.fetchBenchmarkOverview).mockResolvedValue({ data: { runs: [summary], benchmarkedProjects: 1,
    totalRoundsRecorded: 1, classificationRuns: 1, generativeRuns: 0 } } as never);
  vi.mocked(api.fetchProjectBenchmark).mockResolvedValue({ data: { runId, summary,
    rounds: [{ serverRound: 1, loss: .2, accuracy: .8 }], taskType: 'SEQ_CLASSIFICATION',
    classLabels: null, latestPerClass: null, latestConfusionMatrix: null } } as never);
}

it('opens the run represented by the overview row', async () => {
  prepare('recorded-run-1', true);
  render(<BenchmarkDashboard />);
  await userEvent.click(await screen.findByRole('button', { name: 'Recorded project' }));
  await waitFor(() => expect(api.fetchProjectBenchmark).toHaveBeenCalledWith('project-1', 'recorded-run-1'));
  expect(await screen.findByText('Run recorded')).toBeInTheDocument();
  expect(screen.getAllByTestId('benchmark-chart').length).toBeGreaterThan(0);
});

it('keeps unknown legacy records separate and does not draw them as one run', async () => {
  prepare(null, false);
  render(<BenchmarkDashboard />);
  await userEvent.click(await screen.findByRole('button', { name: 'Recorded project' }));
  await waitFor(() => expect(api.fetchProjectBenchmark).toHaveBeenCalledWith('project-1', null));
  expect(await screen.findByRole('heading', { name: 'Legacy benchmark records' })).toBeInTheDocument();
  expect(screen.queryAllByTestId('benchmark-chart')).toHaveLength(0);
  expect(screen.getByRole('table', { name: 'Legacy benchmark measurements' })).toBeInTheDocument();
});

it('does not present a reported legacy run identity as verified chart data', async () => {
  prepare('reported-old-run', false);
  render(<BenchmarkDashboard />);
  await userEvent.click(await screen.findByRole('button', { name: 'Recorded project' }));
  expect(await screen.findByRole('heading', { name: 'Legacy benchmark records' })).toBeInTheDocument();
  expect(screen.queryAllByTestId('benchmark-chart')).toHaveLength(0);
});

it('labels legacy overview snapshots and suppresses their unverified target claim', async () => {
  prepare('reported-old-run', false);
  render(<BenchmarkDashboard />);
  expect(await screen.findByText('Legacy · unverified')).toBeInTheDocument();
  expect(screen.queryByText('17r')).not.toBeInTheDocument();
});
