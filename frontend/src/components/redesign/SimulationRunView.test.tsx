import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { recipe, simulationRun, runB } from '../../test/simulationFixtures';
import { SimulationRunView } from './SimulationRunView';
import { SimulationCompare } from './SimulationCompare';

const selection = { ...recipe, workloadId: 'nih-effusion-r0', recipe: 'effusion-dp', evaluationSplit: 'development selection half' };
const view = (run: ReturnType<typeof simulationRun>) => render(<SimulationRunView run={run} record={null} onStop={() => {}} onExport={() => {}}
  onInspect={() => {}} stopping={false} exporting={false} inspecting={false} />);

describe('evaluation wording follows the run recipe', () => {
  it('names the development selection half for an Effusion run', () => {
    view(simulationRun({ recipe: selection }));
    expect(screen.getByText(/Selection-half results from simulated patient groups/)).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Selection-half AUROC' })).toBeInTheDocument();
    expect(screen.getByText(/AUROC from the development selection half/)).toBeInTheDocument();
    expect(screen.queryByText(/Validation results/)).toBeNull();
  });
  it('keeps the historical validation wording', () => {
    view(simulationRun());
    expect(screen.getByText(/Validation results from simulated patient groups/)).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'Validation AUROC' })).toBeInTheDocument();
  });
  it('titles a comparison of two Effusion runs by their own evaluation set', () => {
    render(<SimulationCompare runs={[simulationRun({ recipe: selection }), simulationRun({ id: runB, recipe: { ...selection, privacy: 'central' } })]} />);
    expect(screen.getByText('Selection-half AUROC across completed rounds')).toBeInTheDocument();
  });
});
