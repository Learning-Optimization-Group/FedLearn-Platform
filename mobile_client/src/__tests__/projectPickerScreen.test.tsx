// Behavior tests for the segmented Projects tab (stage 2), driven through the shared
// function-call harness (see testUtils/componentHarness.ts — no renderer exists in this repo).
//
// Pins the stage-2 flow contract:
//   · an accessible two-segment control (Joined default, Discover);
//   · Joined rows carry the device-local ledger's last-contribution info and open ProjectDetail;
//   · Discover rows carry eligibility markers;
//   · the PUBLIC one-tap Join affordance ROUTES THROUGH ProjectDetail (the privacy label is the
//     single interstitial) — this screen must never call the REST join itself.
import type * as ReactTypes from 'react';

const mockNavigate = jest.fn();
const mockFocusCallbacks: Array<() => void> = [];
const mockListProjects = jest.fn();
const mockJoinProject = jest.fn();
const mockLedgerList = jest.fn();
const mockCaps = jest.fn();
const mockResultFor: Record<
  string,
  { eligible: boolean; hardFailures: string[]; softWarnings: string[] }
> = {};

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
jest.mock('../theme/useThemeTokens', () => ({
  useThemeTokens: () => ({ colors: new Proxy({}, { get: () => '#000000' }) }),
}));
jest.mock('../lib/projectsApi', () => ({
  listProjects: () => mockListProjects(),
  joinProject: (id: string) => mockJoinProject(id),
  annotateEligibility: (projects: Array<{ projectId: string }>) =>
    projects.map((p) => ({
      project: p,
      result: mockResultFor[p.projectId] ?? { eligible: true, hardFailures: [], softWarnings: [] },
    })),
}));
jest.mock('../lib/deviceClass', () => ({ collectDeviceCapabilities: () => mockCaps() }));
jest.mock('../lib/contributionLedger', () => ({
  contributionLedger: { list: () => mockLedgerList() },
}));

import { ProjectPickerScreen } from '../screens/ProjectPickerScreen';
import {
  allElements,
  flatListRows,
  flush,
  press,
  pressableByLabel,
  pressables,
  renderComponent,
  textOf,
} from '../testUtils/componentHarness';

const P_JOINED = {
  projectId: 'p-joined',
  name: 'Thermal Study',
  modelType: 'CNN',
  status: 'READY',
  visibility: 'PUBLIC',
  joined: true,
};
const P_PUBLIC = {
  projectId: 'p-public',
  name: 'Open MNIST',
  modelType: 'MLP',
  status: 'READY',
  visibility: 'PUBLIC',
  joined: false,
};
const P_RESTRICTED = {
  projectId: 'p-restricted',
  name: 'Hospital ECG',
  modelType: 'TRANSFORMER',
  status: 'READY',
  visibility: 'RESTRICTED',
  joined: false,
};

function renderScreen(): void {
  mockFocusCallbacks.length = 0;
  renderComponent(() => (ProjectPickerScreen as unknown as () => ReactTypes.ReactNode)());
}

async function focusScreen(): Promise<void> {
  mockFocusCallbacks[mockFocusCallbacks.length - 1]?.();
  await flush();
}

beforeEach(() => {
  jest.clearAllMocks();
  for (const k of Object.keys(mockResultFor)) delete mockResultFor[k];
  mockListProjects.mockResolvedValue([P_JOINED, P_PUBLIC, P_RESTRICTED]);
  mockCaps.mockResolvedValue({});
  mockLedgerList.mockResolvedValue([
    {
      projectId: 'p-joined',
      projectName: 'Thermal Study',
      round: 7,
      wallClockMs: 4000,
      bytesUp: 128,
      bytesDown: 256,
      at: '2026-07-10T00:00:00.000Z',
    },
  ]);
});

describe('segmented control', () => {
  test('renders an accessible tablist with Joined (selected by default) and Discover', async () => {
    renderScreen();
    await focusScreen();
    expect(allElements().some((e) => e.props.accessibilityRole === 'tablist')).toBe(true);
    const tabs = allElements().filter((e) => e.props.accessibilityRole === 'tab');
    expect(tabs.map((t) => t.props.accessibilityLabel)).toEqual(['Joined', 'Discover']);
    expect(tabs[0]?.props.accessibilityState).toEqual({ selected: true });
    expect(tabs[1]?.props.accessibilityState).toEqual({ selected: false });
  });

  test('pressing Discover switches the selected segment', async () => {
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Discover'));
    const tabs = allElements().filter((e) => e.props.accessibilityRole === 'tab');
    expect(tabs[1]?.props.accessibilityState).toEqual({ selected: true });
  });
});

describe('Joined segment', () => {
  test('lists only joined projects, each with its ledger last-contribution info', async () => {
    renderScreen();
    await focusScreen();
    const rows = flatListRows();
    expect(rows).toHaveLength(1);
    const text = rows.map(textOf).join('\n');
    expect(text).toContain('Thermal Study');
    expect(text).toContain('1 round contributed');
    expect(text).not.toContain('Open MNIST');
  });

  test('a joined project with no ledger entries reads honestly', async () => {
    mockLedgerList.mockResolvedValue([]);
    renderScreen();
    await focusScreen();
    expect(flatListRows().map(textOf).join('\n')).toContain(
      'No contributions from this device yet',
    );
  });

  test('tapping a joined row opens ProjectDetail for that project', async () => {
    renderScreen();
    await focusScreen();
    const rows = flatListRows();
    await press(pressableByLabel('Thermal Study', rows));
    expect(mockNavigate).toHaveBeenCalledWith('ProjectDetail', { projectId: 'p-joined' });
  });
});

describe('Discover segment', () => {
  test('lists non-joined projects with eligibility markers and failure lines', async () => {
    mockResultFor['p-restricted'] = {
      eligible: false,
      hardFailures: ['Needs 8 GB RAM, device has 3.5 GB'],
      softWarnings: [],
    };
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Discover'));
    const text = flatListRows().map(textOf).join('\n');
    expect(text).toContain('Open MNIST — recommended');
    expect(text).toContain('Hospital ECG — unsupported');
    expect(text).toContain('Needs 8 GB RAM, device has 3.5 GB');
    expect(text).not.toContain('Thermal Study');
  });

  test('the PUBLIC row routes through ProjectDetail; the Join chip is visual only', async () => {
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Discover'));
    const rows = flatListRows();
    // The row is the single touchable — no nested Join pressable duplicating its onPress.
    expect(pressables(rows)).toHaveLength(rows.length);
    await press(pressableByLabel('Open MNIST — recommended', rows));
    expect(mockNavigate).toHaveBeenCalledWith('ProjectDetail', { projectId: 'p-public' });
    expect(mockJoinProject).not.toHaveBeenCalled();
  });

  test('non-PUBLIC rows get no Join chip (approval flows live in ProjectDetail)', async () => {
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Discover'));
    const rows = flatListRows();
    const rowText = (name: string) => rows.map(textOf).find((t) => t.includes(name)) ?? '';
    expect(rowText('Open MNIST')).toContain('Join');
    expect(rowText('Hospital ECG')).not.toContain('Join');
  });

  test('tapping a discover row opens ProjectDetail', async () => {
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Discover'));
    await press(pressableByLabel('Hospital ECG — recommended', flatListRows()));
    expect(mockNavigate).toHaveBeenCalledWith('ProjectDetail', { projectId: 'p-restricted' });
  });
});

describe('load failure', () => {
  test('a failed fetch shows the error + Retry, never the misleading empty-segment copy', async () => {
    mockListProjects.mockRejectedValue(new Error('network down'));
    renderScreen();
    await focusScreen();
    // The list (and with it the "No joined projects" / "Nothing to discover" empty states)
    // must not render — the error banner + Retry replace it.
    expect(() => flatListRows()).toThrow('no FlatList');
    expect(allElements().map((e) => e.props.message)).toContain('network down');
    expect(pressableByLabel('Retry')).toBeTruthy();
  });

  test('Retry reloads and recovers the list', async () => {
    mockListProjects.mockRejectedValueOnce(new Error('network down'));
    renderScreen();
    await focusScreen();
    await press(pressableByLabel('Retry'));
    expect(flatListRows().map(textOf).join('\n')).toContain('Thermal Study');
    expect(allElements().map((e) => e.props.message)).not.toContain('network down');
  });
});
