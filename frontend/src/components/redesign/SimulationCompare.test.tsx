import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { simulationRun, recipe, round, runB } from '../../test/simulationFixtures';
import { SimulationCompare } from './SimulationCompare';

describe('exploratory run comparisons', () => {
  it('refuses runs with different workload pins rather than making a utility ranking', () => {
    render(<SimulationCompare runs={[simulationRun(), simulationRun({ id: runB, recipe: { ...recipe, featureReportSha256: 'c'.repeat(64) } })]} />);
    expect(screen.getByRole('alert')).toHaveTextContent('same workload, feature pin and evaluation split');
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });
  it('refuses an active run and an empty workload identity', () => {
    render(<SimulationCompare runs={[simulationRun(), simulationRun({ id: runB, status: 'RUNNING', recipe: {} })]} />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });
  it('shows final observed endpoints, explicit recipe differences and partial execution', () => {
    render(<SimulationCompare runs={[simulationRun(), simulationRun({ id: runB, status: 'STOPPED', recipe: { ...recipe, privacy: 'central', epsilon: 4 }, rounds: [round(1,.55)], completedRounds: 1, qualification: null })]} />);
    const row = screen.getByRole('row', { name: /Final AUROC/ });
    expect(within(row).getByText('0.610')).toBeInTheDocument();
    expect(within(row).getByText('0.550')).toBeInTheDocument();
    expect(screen.getByText('Stopped (partial)')).toBeInTheDocument();
    expect(screen.getByRole('row', { name: /Privacy/ })).toHaveTextContent('central');
    expect(screen.getByText(/Modeled upload/)).toBeInTheDocument();
    expect(screen.queryByText(/winner|confidence interval|best round/i)).not.toBeInTheDocument();
  });
});

it('keeps unavailable endpoints and communication unknown rather than zero', () => {
  const missing = round(1); missing.auc = null; missing.comm.up_bytes = null;
  render(<SimulationCompare runs={[simulationRun(), simulationRun({ id: runB, rounds: [missing], completedRounds: 1, status: 'FAILED', qualification: null })]} />);
  expect(within(screen.getByRole('row', { name: /Final AUROC/ })).getByText('—')).toBeInTheDocument();
  expect(within(screen.getByRole('row', { name: /Modeled upload/ })).getByText('—')).toBeInTheDocument();
});
it('refuses a different evaluation split', () => {
  render(<SimulationCompare runs={[simulationRun(), simulationRun({ id: runB, recipe: { ...recipe, evaluationSplit: 'test' } })]} />);
  expect(screen.getByRole('alert')).toHaveTextContent('same workload, feature pin and evaluation split');
});

it('keeps partial curve tails missing instead of extrapolating them', async () => {
  const { fireEvent } = await import('@testing-library/react');
  render(<SimulationCompare runs={[simulationRun(), simulationRun({ id: runB, status: 'STOPPED', qualification: null, completedRounds: 1, rounds: [round(1,.55)] })]} />);
  fireEvent.click(screen.getByText('Round-by-round values'));
  const values = screen.getByRole('table', { name: 'Observed AUROC by completed round' });
  const last = within(values).getByRole('row', { name: '3 0.610 —' });
  expect(within(last).getByText('—')).toBeInTheDocument();
  expect(within(values).getByRole('row', { name: '1 0.920 0.550' })).toBeInTheDocument();
});
