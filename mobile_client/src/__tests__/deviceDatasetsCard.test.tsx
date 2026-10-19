// Stage 3 B2: Settings lists the datasets imported on this phone and deletes them, except one a run is training.
import type * as ReactTypes from 'react';

jest.mock('react', () =>
  jest.requireActual<typeof import('../testUtils/componentHarness')>('../testUtils/componentHarness')
    .createReactMock(),
);
const mockFocusCallbacks: Array<() => void> = [];
jest.mock('@react-navigation/native', () => ({
  useFocusEffect: (cb: () => void) => {
    mockFocusCallbacks.push(cb);
  },
}));
jest.mock('lucide-react-native', () => new Proxy({}, { get: () => () => null }));
jest.mock('../theme/useThemeTokens', () => ({
  useThemeTokens: () => ({ colors: new Proxy({}, { get: () => '#000000' }) }),
}));
jest.mock('../lib/datasetService', () => ({ listDatasets: jest.fn(), deleteDataset: jest.fn() }));
let mockInUse: string | null = null;
jest.mock('../state/TrainingContext', () => ({ useTraining: () => ({ state: { datasetInUse: mockInUse } }) }));

import { DeviceDatasetsCard } from '../components/DeviceDatasetsCard';
import { deleteDataset, listDatasets, type DatasetSnapshot } from '../lib/datasetService';
import { allElements, flush, press, pressableByLabel, renderComponent, screenText } from '../testUtils/componentHarness';

const mList = listDatasets as jest.Mock;
const mDelete = deleteDataset as jest.Mock;

function snapshot(id: string): DatasetSnapshot {
  return {
    snapshotId: id.repeat(64), recordCount: 6, inputShape: [4], inputDtype: 'f32', classNames: ['c0', 'c1', 'c2'],
    labelSchemaId: 'labels-sha256:x', inputsPath: '/i', targetsPath: '/t',
  };
}

async function open() {
  mockFocusCallbacks.length = 0;
  renderComponent(() => (DeviceDatasetsCard as unknown as () => ReactTypes.ReactNode)());
  mockFocusCallbacks.forEach((cb) => cb());
  await flush();
}

beforeEach(() => {
  mockInUse = null;
  mList.mockReset().mockResolvedValue([snapshot('a'), snapshot('b')]);
  mDelete.mockReset().mockResolvedValue(undefined);
});

test('lists the datasets on this phone', async () => {
  await open();
  expect(screenText()).toContain('6 examples · aaaaaaaa');
  expect(screenText()).toContain('c0, c1, c2');
  expect(screenText()).toContain('bbbbbbbb');
});

test('deletes a dataset and refreshes the list', async () => {
  await open();
  mList.mockResolvedValue([snapshot('b')]);
  await press(pressableByLabel('Delete dataset aaaaaaaa'));
  expect(mDelete).toHaveBeenCalledWith('a'.repeat(64), []);
  expect(screenText()).not.toContain('aaaaaaaa');
});

test('a dataset a run is training cannot be deleted', async () => {
  mockInUse = 'a'.repeat(64);
  await open();
  expect(pressableByLabel('Delete dataset aaaaaaaa').props.disabled).toBe(true);
  expect(screenText()).toContain('In use by training');
  await press(pressableByLabel('Delete dataset bbbbbbbb'));
  expect(mDelete).toHaveBeenCalledWith('b'.repeat(64), ['a'.repeat(64)]);
});

test('a refused delete says why', async () => {
  mDelete.mockRejectedValue(Object.assign(new Error('the snapshot is in use'), { code: 'DATASET_PINNED' }));
  await open();
  await press(pressableByLabel('Delete dataset aaaaaaaa'));
  expect(allElements().some((e) => e.props.message === 'the snapshot is in use')).toBe(true);
});

test('says when there are no datasets', async () => {
  mList.mockResolvedValue([]);
  await open();
  expect(screenText()).toContain('No datasets on this phone');
});
