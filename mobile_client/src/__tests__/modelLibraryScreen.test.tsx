// Behavior tests for the Models hub (stage 2) via the shared function-call harness: the model
// list renders from the on-device registry, and each row pushes the two model surfaces —
// Test → ModelTesting (with THAT row's model path), Playground → the server-side playground.
import type * as ReactTypes from 'react';

const mockNavigate = jest.fn();
const mockFocusCallbacks: Array<() => void> = [];
const mockListModels = jest.fn();

jest.mock('react', () =>
  jest
    .requireActual<typeof import('../testUtils/componentHarness')>('../testUtils/componentHarness')
    .createReactMock(),
);
jest.mock('@react-navigation/native', () => ({
  useNavigation: () => ({ navigate: mockNavigate }),
  useFocusEffect: (cb: () => void) => {
    mockFocusCallbacks.push(cb);
  },
}));
jest.mock('lucide-react-native', () => new Proxy({}, { get: () => () => null }));
// RN 0.80's jest setup registers a RefreshControl mock whose mock file no longer resolves —
// accessing the RefreshControl getter throws in ANY test. Shim just that export.
jest.mock('react-native', () => {
  const rn = jest.requireActual<Record<string, unknown>>('react-native');
  return new Proxy(rn, {
    get: (target, prop: string | symbol) =>
      prop === 'RefreshControl' ? () => null : target[prop as keyof typeof target],
  });
});
jest.mock('../theme/useThemeTokens', () => ({
  useThemeTokens: () => ({ colors: new Proxy({}, { get: () => '#000000' }) }),
}));
jest.mock('../lib/modelStore', () => ({ listModels: () => mockListModels() }));

import { ModelLibraryScreen } from '../screens/ModelLibraryScreen';
import {
  flatListRows,
  flush,
  press,
  pressableByLabel,
  renderComponent,
  screenText,
  textOf,
} from '../testUtils/componentHarness';

const MODEL_A = {
  name: 'Alpha',
  path: '/models/alpha.pt',
  sha256: 'aaaa1111',
  tier: '1M',
  round: 3,
  savedAt: '2026-07-10T00:00:00.000Z',
};
const MODEL_B = {
  name: 'Beta',
  path: '/models/beta.pt',
  sha256: 'bbbb2222',
  tier: '10M',
  round: 9,
  savedAt: '2026-07-12T00:00:00.000Z',
};

function renderScreen(): void {
  mockFocusCallbacks.length = 0;
  renderComponent(() => (ModelLibraryScreen as unknown as () => ReactTypes.ReactNode)());
}

async function focusScreen(): Promise<void> {
  mockFocusCallbacks[mockFocusCallbacks.length - 1]?.();
  await flush();
}

beforeEach(() => {
  jest.clearAllMocks();
  mockListModels.mockResolvedValue([MODEL_A, MODEL_B]);
});

describe('Models hub', () => {
  test('lists saved snapshots with an honest count (no fabricated storage size)', async () => {
    renderScreen();
    await focusScreen();
    expect(screenText()).toContain('2 on-device snapshots');
    const text = flatListRows().map(textOf).join('\n');
    expect(text).toContain('Alpha');
    expect(text).toContain('Beta');
    expect(text).toContain('sha256 aaaa1111');
    expect(text).toContain('tier 10M');
  });

  test('Test pushes ModelTesting with THAT row’s model path', async () => {
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Test Beta', flatListRows()));
    expect(mockNavigate).toHaveBeenCalledWith('ModelTesting', { modelPath: '/models/beta.pt' });
  });

  test('Playground pushes the server-side playground', async () => {
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Open the playground from Alpha', flatListRows()));
    expect(mockNavigate).toHaveBeenCalledWith('Playground');
  });

  test('empty registry: no rows and the plain caption', async () => {
    mockListModels.mockResolvedValue([]);
    renderScreen();
    await focusScreen();
    expect(flatListRows()).toHaveLength(0);
    expect(screenText()).toContain('On-device snapshots, encrypted at rest');
  });
});
