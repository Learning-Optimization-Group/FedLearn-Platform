// Stage 3 B2: on a run that trains the device's own data, Home lists the datasets on the device, says which ones fit
// the run and why the others do not, and imports a new one against the run's classes.
import type * as ReactTypes from 'react';

jest.mock('react', () =>
  jest.requireActual<typeof import('../testUtils/componentHarness')>('../testUtils/componentHarness')
    .createReactMock(),
);
jest.mock('lucide-react-native', () => new Proxy({}, { get: () => () => null }));
jest.mock('../theme/useThemeTokens', () => ({
  useThemeTokens: () => ({ colors: new Proxy({}, { get: () => '#000000' }) }),
}));
jest.mock('../lib/restClient', () => ({ api: { get: jest.fn() } }));

import { TrainingDataCard, type TrainingDataCardProps } from '../components/TrainingDataCard';
import type { DatasetSnapshot } from '../lib/datasetService';
import type { RunDataPlan } from '../lib/runDataPlan';
import { allElements, press, pressableByLabel, renderComponent, screenText } from '../testUtils/componentHarness';

const LABELS = 'labels-sha256:4817507d9942bf24635535b19b1588bb44e00d2b51314536926f3580c7f5d896';
const PLAN = {
  source: 'LOCAL_SNAPSHOT', classNames: ['c0', 'c1', 'c2'], shape: { kind: 'vector', width: 4 }, batchSize: 8,
  maxExamples: 8,
  requirement: {
    labelSchemaId: LABELS, inputShape: [BigInt(4)], inputDtype: 1, classCount: 3,
    transforms: [{ operation: { case: 'identityVector', value: { width: 4 } } }],
  },
} as unknown as RunDataPlan;

function snapshot(id: string, over: Partial<DatasetSnapshot> = {}): DatasetSnapshot {
  return {
    snapshotId: id.repeat(64), recordCount: 6, inputShape: [4], inputDtype: 'f32', classNames: ['c0', 'c1', 'c2'],
    labelSchemaId: LABELS, inputsPath: '/i', targetsPath: '/t', ...over,
  };
}

function render(over: Partial<TrainingDataCardProps> = {}) {
  const props: TrainingDataCardProps = {
    plan: PLAN, datasets: [], selectedId: null, onSelect: jest.fn(), onImport: jest.fn(), importing: false,
    importError: null, ...over,
  };
  renderComponent(() => (TrainingDataCard as unknown as (p: TrainingDataCardProps) => ReactTypes.ReactNode)(props));
  return props;
}

test('shows nothing for a run on the built-in data', () => {
  render({ plan: { source: 'FIXTURE' } });
  expect(screenText()).toBe('');
});

test('says the run trains on this device\'s own data, and what it takes', () => {
  render();
  expect(screenText()).toContain('This run trains on your own data');
  expect(screenText()).toContain('4 numbers with one label');
  expect(screenText()).toContain('c0, c1, c2');
});

test('states no example cap for a run that trains in minibatches', () => {
  render({ plan: { ...PLAN, maxExamples: null } as RunDataPlan });
  expect(screenText()).not.toContain('At most');
  expect(screenText()).toContain('in batches of 8');
});

test('a dataset that fits the run can be chosen', async () => {
  const props = render({ datasets: [snapshot('a')] });
  await press(pressableByLabel('Use dataset aaaaaaaa'));
  expect(props.onSelect).toHaveBeenCalledWith('a'.repeat(64));
});

test('a dataset that does not fit says why and cannot be chosen', async () => {
  const props = render({ datasets: [snapshot('b', { recordCount: 9 })] });
  expect(screenText()).toContain('more than 8 examples');
  const row = pressableByLabel('Use dataset bbbbbbbb');
  expect(row.props.disabled).toBe(true);
  expect(props.onSelect).not.toHaveBeenCalled();
});

test('marks the chosen dataset', () => {
  render({ datasets: [snapshot('a')], selectedId: 'a'.repeat(64) });
  expect(pressableByLabel('Use dataset aaaaaaaa').props.accessibilityState).toMatchObject({ selected: true });
});

test('imports a file against the run', async () => {
  const props = render();
  await press(pressableByLabel('Import a dataset file'));
  expect(props.onImport).toHaveBeenCalled();
});

test('shows why an import failed', () => {
  render({ importError: 'row 3 has 5 values, expected 4' });
  expect(allElements().some((e) => e.props.message === 'row 3 has 5 values, expected 4')).toBe(true);
});

test('says when no dataset on the device fits yet', () => {
  render({ datasets: [snapshot('b', { labelSchemaId: 'labels-sha256:' + '0'.repeat(64) })] });
  expect(screenText()).toContain('different classes');
  expect(screenText()).toContain('No dataset on this phone fits this run yet');
});

test('an image run asks for an image package, not a .csv', () => {
  const imagePlan = { ...PLAN, shape: { kind: 'image', height: 32, width: 32, channels: 3 } } as unknown as RunDataPlan;
  render({ plan: imagePlan, datasets: [] });
  expect(screenText()).toContain('Import a zipped image package.');
  expect(screenText()).not.toContain('.csv');
});
