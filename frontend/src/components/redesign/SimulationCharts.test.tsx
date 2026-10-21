import { Component, type ReactNode } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { SimulationRunView } from './SimulationRunView';
import { SimulationCompare } from './SimulationCompare';
import { round, runB, simulationRun } from '../../test/simulationFixtures';

class ChartBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? <p role="alert">Chart crashed</p> : this.props.children; }
}

beforeEach(() => {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const height = this.classList.contains('recharts-legend-wrapper') ? 24 : 240;
    return { x: 0, y: 0, width: 640, height, top: 0, left: 0, right: 640, bottom: height, toJSON() {} };
  });
  vi.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockReturnValue(640);
  vi.spyOn(HTMLElement.prototype, 'offsetHeight', 'get').mockImplementation(function (this: HTMLElement) { return this.classList.contains('recharts-legend-wrapper') ? 24 : 240; });
});
afterEach(() => vi.restoreAllMocks());

async function hoverLastRound(container: HTMLElement) {
  const chart = container.querySelector('.recharts-wrapper');
  expect(chart).not.toBeNull();
  fireEvent.mouseMove(chart!, { clientX: 620, clientY: 100 });
  await waitFor(() => {
    expect(screen.queryByText('Chart crashed')).not.toBeInTheDocument();
    expect(container.querySelector('.recharts-tooltip-label')).toHaveTextContent('3');
  });
}

it('keeps a failed run usable when its last loss is a tagged nonfinite value and the chart is hovered', async () => {
  const failed = round(3);
  failed.loss = { __nonfinite_float__: 'inf' };
  const run = simulationRun({ status: 'FAILED', qualification: null, recordEncoding: 'nonfinite_tagged_v1', rounds: [round(1), round(2), failed] });
  const view = render(<ChartBoundary><SimulationRunView run={run} record={null} onStop={() => {}} onExport={() => {}} onInspect={() => {}} stopping={false} exporting={false} inspecting={false} /></ChartBoundary>);
  fireEvent.change(screen.getByLabelText('Metric curve'), { target: { value: 'loss' } });
  await hoverLastRound(view.container);
  expect(screen.getByRole('status')).toHaveTextContent('Failed');
  expect(screen.getByLabelText('Final loss')).toHaveTextContent('—');
  expect(screen.getByRole('button', { name: 'Download research record' })).toBeEnabled();
});

it('keeps comparisons usable when a failed run has tagged AUROC and another run has a finite observation', async () => {
  const failed = round(3);
  failed.auc = { __nonfinite_float__: 'nan' };
  const view = render(<ChartBoundary><SimulationCompare runs={[simulationRun(), simulationRun({ id: runB, status: 'FAILED', qualification: null, rounds: [round(1), round(2), failed] })]} /></ChartBoundary>);
  await hoverLastRound(view.container);
  const endpoint = screen.getByRole('row', { name: /Final AUROC/ });
  expect(within(endpoint).getByText('0.610')).toBeInTheDocument();
  expect(within(endpoint).getByText('—')).toBeInTheDocument();
  expect(screen.getByText('Failed (partial)')).toBeInTheDocument();
});
