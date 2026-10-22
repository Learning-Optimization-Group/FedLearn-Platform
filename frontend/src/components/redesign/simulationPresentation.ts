import type { SimulationRound, SimulationRun, SimulationScalar } from '../../services/simulationService';

export function finiteMetric(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
export function metric(value: unknown, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}
export function sumRounds(run: SimulationRun, read: (round: SimulationRound) => SimulationScalar): number | null {
  const values = run.rounds.map(round => finiteMetric(read(round)));
  return values.length && values.every(v => typeof v === 'number' && Number.isFinite(v))
    ? values.reduce<number>((sum, value) => sum + value!, 0) : null;
}
export function statusLabel(run: SimulationRun): string {
  return ({ ACCEPTED: 'Accepted', STARTING: 'Starting', RUNNING: 'Running',
    STOPPING: 'Stopping; waiting for the worker to exit', COMPLETED: 'Completed',
    STOPPED: 'Stopped', FAILED: 'Failed', TIMED_OUT: 'Timed out', INTERRUPTED: 'Interrupted' })[run.status];
}
export function finalRound(run: SimulationRun): SimulationRound | undefined { return run.rounds[run.rounds.length - 1]; }

/** The evaluation set a run's recipe names: the Effusion bundle's selection half, or the historical validation split. */
export function evaluationScope(recipe: { evaluationSplit?: string }): { noun: string; adjective: string } {
  return recipe.evaluationSplit === 'development selection half'
    ? { noun: 'development selection half', adjective: 'Selection-half' }
    : { noun: 'validation', adjective: 'Validation' };
}
