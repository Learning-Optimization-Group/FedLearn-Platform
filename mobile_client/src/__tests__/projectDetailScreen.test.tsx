// Behavior tests for ProjectDetail (stage 2) via the shared function-call harness.
//
// Pins the contract of the join interstitial:
//   · the privacy label's three sections render unconditionally (it IS the interstitial);
//   · Join executes HERE — REST membership join first for a PUBLIC non-member, then the run
//     join through TrainingContext; a member skips the membership call;
//   · once this project is the joined run: run identity (run id, partition), the live server
//     endpoint inside "Leaves your phone", and Leave as the single action;
//   · a run on another project / a non-PUBLIC non-member / a missing native core all explain
//     instead of offering Join;
//   · per-project contribution history comes from the device-local ledger.
import type * as ReactTypes from 'react';

const mockFocusCallbacks: Array<() => void> = [];
const mockListProjects = jest.fn();
const mockJoinProject = jest.fn();
const mockEntriesForProject = jest.fn();
const mockCaps = jest.fn();
const mockJoin = jest.fn();
const mockStop = jest.fn();
const mockNativeAvailable = jest.fn();
const mockTrainingState: Record<string, unknown> = {};
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
  useFocusEffect: (cb: () => void) => {
    mockFocusCallbacks.push(cb);
  },
}));
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
  contributionLedger: { entriesForProject: (id: string, n: number) => mockEntriesForProject(id, n) },
}));
jest.mock('../lib/nativeCore', () => ({
  isNativeCoreAvailable: () => mockNativeAvailable(),
}));
jest.mock('../state/TrainingContext', () => ({
  useTraining: () => ({
    state: mockTrainingState,
    join: mockJoin,
    startTraining: jest.fn(),
    stopTraining: mockStop,
  }),
}));

import { ProjectDetailScreen } from '../screens/ProjectDetailScreen';
import {
  flush,
  press,
  pressableByLabel,
  pressables,
  renderComponent,
  screenText,
} from '../testUtils/componentHarness';

const P_PUBLIC = {
  projectId: 'p-public',
  name: 'Open MNIST',
  modelType: 'MLP',
  status: 'READY',
  visibility: 'PUBLIC',
  joined: false,
};
const P_MEMBER = {
  projectId: 'p-member',
  name: 'Thermal Study',
  modelType: 'CNN',
  status: 'READY',
  visibility: 'PUBLIC',
  joined: true,
};
const P_RESTRICTED = {
  projectId: 'p-restricted',
  name: 'Hospital ECG',
  modelType: 'TRANSFORMER',
  status: 'READY',
  visibility: 'RESTRICTED',
  joined: false,
};

function setTrainingState(over: Record<string, unknown> = {}): void {
  for (const k of Object.keys(mockTrainingState)) delete mockTrainingState[k];
  Object.assign(
    mockTrainingState,
    {
      machine: 'notJoined',
      joining: false,
      stopping: false,
      error: null,
      joined: null,
      projectName: null,
      logs: [],
      latestRound: null,
      serverStatus: null,
      session: { rounds: 0, scalarsTransmitted: 0, bytesUp: 0, bytesDown: 0, computeMs: 0 },
    },
    over,
  );
}

function joinedRun(projectId: string): Record<string, unknown> {
  return {
    runId: 'run-42',
    projectId,
    partitionId: 3,
    assignedRound: 5,
    grpcEndpoint: '10.0.0.5:50005',
    manifest: {},
    message: 'ok',
  };
}

function renderScreen(projectId: string): void {
  mockFocusCallbacks.length = 0;
  renderComponent(() =>
    (
      ProjectDetailScreen as unknown as (p: {
        route: { params: { projectId: string } };
      }) => ReactTypes.ReactNode
    )({ route: { params: { projectId } } }),
  );
}

async function focusScreen(): Promise<void> {
  mockFocusCallbacks[mockFocusCallbacks.length - 1]?.();
  await flush();
}

function pressableLabels(): unknown[] {
  return pressables().map((e) => e.props.accessibilityLabel);
}

beforeEach(() => {
  jest.clearAllMocks();
  for (const k of Object.keys(mockResultFor)) delete mockResultFor[k];
  setTrainingState();
  mockListProjects.mockResolvedValue([P_PUBLIC, P_MEMBER, P_RESTRICTED]);
  mockCaps.mockResolvedValue({});
  mockEntriesForProject.mockResolvedValue([]);
  mockNativeAvailable.mockReturnValue(true);
  mockJoinProject.mockResolvedValue(undefined);
  mockJoin.mockResolvedValue(undefined);
  mockStop.mockResolvedValue(undefined);
});

describe('privacy label (the join interstitial)', () => {
  test('renders its three sections before any data has loaded', () => {
    renderScreen('p-public');
    const text = screenText();
    expect(text).toContain('Stays on your phone');
    expect(text).toContain('Leaves your phone');
    expect(text).toContain('Never collected');
  });

  test('keeps the three sections once the project has loaded', async () => {
    renderScreen('p-public');
    await focusScreen();
    const text = screenText();
    expect(text).toContain('Stays on your phone');
    expect(text).toContain('Leaves your phone');
    expect(text).toContain('Never collected');
    expect(text).toContain('Open MNIST');
  });

  test('shows the live server endpoint under "Leaves your phone" once joined here', async () => {
    setTrainingState({ joined: joinedRun('p-public'), projectName: 'Open MNIST', machine: 'joined' });
    renderScreen('p-public');
    await focusScreen();
    expect(screenText()).toContain('Training server');
    expect(screenText()).toContain('10.0.0.5:50005');
  });
});

describe('join action', () => {
  test('PUBLIC non-member: membership join first, then the run join, in order', async () => {
    renderScreen('p-public');
    await focusScreen();
    await press(pressableByLabel('Join training run'));
    await flush();
    expect(mockJoinProject).toHaveBeenCalledWith('p-public');
    expect(mockJoin).toHaveBeenCalledWith('p-public', 'Open MNIST');
    expect(mockJoinProject.mock.invocationCallOrder[0]).toBeLessThan(
      mockJoin.mock.invocationCallOrder[0] ?? -1,
    );
  });

  test('existing member: skips the REST membership call', async () => {
    renderScreen('p-member');
    await focusScreen();
    await press(pressableByLabel('Join training run'));
    await flush();
    expect(mockJoinProject).not.toHaveBeenCalled();
    expect(mockJoin).toHaveBeenCalledWith('p-member', 'Thermal Study');
  });

  test('RESTRICTED non-member: approval note instead of a Join button', async () => {
    renderScreen('p-restricted');
    await focusScreen();
    expect(screenText()).toContain('owner approval');
    expect(pressableLabels()).not.toContain('Join training run');
  });

  test('missing native core: explains instead of offering Join', async () => {
    mockNativeAvailable.mockReturnValue(false);
    renderScreen('p-public');
    await focusScreen();
    expect(screenText()).toContain('not available in this build');
    expect(pressableLabels()).not.toContain('Join training run');
  });

  test('device failing hard gates: explains instead of offering Join', async () => {
    mockResultFor['p-public'] = {
      eligible: false,
      hardFailures: ['Needs 8 GB RAM, device has 3.5 GB'],
      softWarnings: [],
    };
    renderScreen('p-public');
    await focusScreen();
    expect(screenText()).toContain('does not meet the project requirements');
    expect(screenText()).toContain('Needs 8 GB RAM, device has 3.5 GB');
    expect(pressableLabels()).not.toContain('Join training run');
  });
});

describe('joined run', () => {
  test('this project joined: run identity + Leave as the single action, no Join', async () => {
    setTrainingState({ joined: joinedRun('p-public'), projectName: 'Open MNIST', machine: 'joined' });
    renderScreen('p-public');
    await focusScreen();
    const text = screenText();
    expect(text).toContain('run-42');
    expect(text).toContain('3'); // partition
    expect(pressableLabels()).toContain('Leave run');
    expect(pressableLabels()).not.toContain('Join training run');
  });

  test('Leave hands off to the shared stopTraining', async () => {
    setTrainingState({ joined: joinedRun('p-public'), projectName: 'Open MNIST', machine: 'joined' });
    renderScreen('p-public');
    await focusScreen();
    await press(pressableByLabel('Leave run'));
    expect(mockStop).toHaveBeenCalledTimes(1);
  });

  test('a run on another project blocks joining and names it', async () => {
    setTrainingState({ joined: joinedRun('p-other'), projectName: 'ECG Study', machine: 'joined' });
    renderScreen('p-public');
    await focusScreen();
    expect(screenText()).toContain('already joined to ECG Study');
    expect(pressableLabels()).not.toContain('Join training run');
    expect(pressableLabels()).not.toContain('Leave run');
  });
});

describe('contribution history', () => {
  test('renders the per-project ledger slice, newest rows as delivered', async () => {
    mockEntriesForProject.mockResolvedValue([
      {
        projectId: 'p-public',
        projectName: 'Open MNIST',
        round: 7,
        wallClockMs: 4000,
        bytesUp: 2048,
        bytesDown: 4096,
        at: '2026-07-10T00:00:00.000Z',
      },
      {
        projectId: 'p-public',
        projectName: 'Open MNIST',
        round: 6,
        wallClockMs: 3000,
        bytesUp: 1024,
        bytesDown: 4096,
        at: '2026-07-09T00:00:00.000Z',
      },
    ]);
    renderScreen('p-public');
    await focusScreen();
    expect(mockEntriesForProject).toHaveBeenCalledWith('p-public', 10);
    const text = screenText();
    expect(text).toContain('round 7');
    expect(text).toContain('round 6');
    expect(text).toContain('Submitted'); // client-side fact wording — never "accepted"
  });

  test('reads honestly when this device has never contributed here', async () => {
    renderScreen('p-public');
    await focusScreen();
    expect(screenText()).toContain('No contributions from this device yet');
  });
});
