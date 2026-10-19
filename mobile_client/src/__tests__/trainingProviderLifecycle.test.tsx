import type ReactTypes from 'react';
import type { TrainingContextValue } from '../state/TrainingContext';
import type { TrainingHooks } from '../lib/training';
import type { JoinedRun } from '../lib/runJoin';

// Exercise the provider's real callbacks and effect cleanups without a native renderer.
// Each mount owns separate hook cells; native, network and storage boundaries are mocked.
let mockRendering: Pick<ProviderMount, 'reducer' | 'ref' | 'effect'>;
jest.mock('react', () => ({
  ...jest.requireActual('react'),
  useReducer: (reducer: (state: unknown, action: unknown) => unknown, initial: unknown) =>
    mockRendering.reducer(reducer, initial),
  useRef: (value: unknown) => mockRendering.ref(value),
  useEffect: (effect: () => void | (() => void), deps: unknown[]) => mockRendering.effect(effect, deps),
  useCallback: (callback: unknown) => callback,
  useMemo: (factory: () => unknown) => factory(),
}));
jest.mock('../lib/runJoin', () => ({ ...jest.requireActual('../lib/runJoin'), joinRun: jest.fn() }));
jest.mock('../lib/nativeCore', () => ({ __esModule: true, default: { stop: jest.fn() } }));
jest.mock('../lib/training', () => ({ ...jest.requireActual('../lib/training'), runTrainingLoop: jest.fn() }));
jest.mock('../lib/stompClient', () => ({ connectStomp: jest.fn() }));
jest.mock('../lib/statusHeartbeat', () => ({ startServerStatusHeartbeat: jest.fn() }));
jest.mock('../lib/foregroundService', () => ({ foregroundService: { start: jest.fn(), stop: jest.fn() } }));
jest.mock('../lib/contributionLedger', () => ({ contributionLedger: { record: jest.fn(async () => {}) } }));
jest.mock('../lib/diagnosticJournal', () => ({ diagnosticJournal: { append: jest.fn(async () => {}) } }));

import { TrainingProvider } from '../state/TrainingContext';
import { joinRun } from '../lib/runJoin';
import nativeCore from '../lib/nativeCore';
import { runTrainingLoop } from '../lib/training';
import { foregroundService } from '../lib/foregroundService';
import { connectStomp } from '../lib/stompClient';
import { startServerStatusHeartbeat } from '../lib/statusHeartbeat';
import { contributionLedger } from '../lib/contributionLedger';

type Effect = { deps: unknown[]; cleanup?: () => void; setup?: () => void | (() => void) };
class ProviderMount {
  private cells: unknown[] = [];
  private effects: Effect[] = [];
  private cursor = 0;
  private effectCursor = 0;
  private mounted = true;
  lateDispatches = 0;

  reducer(reduce: (state: unknown, action: unknown) => unknown, initial: unknown) {
    const index = this.cursor++;
    if (!(index in this.cells)) this.cells[index] = initial;
    return [this.cells[index], (action: unknown) => {
      if (!this.mounted) { this.lateDispatches++; return; }
      this.cells[index] = reduce(this.cells[index], action);
    }];
  }
  ref(value: unknown) {
    const index = this.cursor++;
    if (!(index in this.cells)) this.cells[index] = { current: value };
    return this.cells[index];
  }
  effect(setup: () => void | (() => void), deps: unknown[]) {
    const index = this.effectCursor++;
    const previous = this.effects[index];
    if (!previous || deps.length !== previous.deps.length || deps.some((d, i) => !Object.is(d, previous.deps[i]))) {
      this.effects[index] = { deps, cleanup: previous?.cleanup, setup };
    }
  }
  render(): TrainingContextValue {
    if (!this.mounted) throw new Error('Unmounted provider');
    mockRendering = { reducer: this.reducer.bind(this), ref: this.ref.bind(this), effect: this.effect.bind(this) };
    this.cursor = 0; this.effectCursor = 0;
    const element = TrainingProvider({ children: null }) as ReactTypes.ReactElement<{ value: TrainingContextValue }>;
    for (const effect of this.effects) {
      if (!effect.setup) continue;
      effect.cleanup?.();
      effect.cleanup = effect.setup() || undefined;
      effect.setup = undefined;
    }
    return element.props.value;
  }
  unmount() {
    this.mounted = false;
    for (const effect of this.effects) { effect.cleanup?.(); effect.cleanup = undefined; }
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(r => { resolve = r; });
  return { promise, resolve };
}
async function flush() { for (let i = 0; i < 20; i++) await Promise.resolve(); }
const JOINED = { runId: 'run-a', projectId: 'project-a', manifest: {} } as JoinedRun;
const mockedJoin = jest.mocked(joinRun);
const mockedLoop = jest.mocked(runTrainingLoop);
const deactivate = jest.fn();
const heartbeatStop = jest.fn();
const subscribe = jest.fn();

beforeEach(() => {
  jest.clearAllMocks();
  mockedJoin.mockResolvedValue(JOINED);
  mockedLoop.mockResolvedValue(undefined);
  jest.mocked(nativeCore.stop).mockResolvedValue(undefined);
  jest.mocked(connectStomp).mockResolvedValue({ deactivate, subscribe } as never);
  jest.mocked(startServerStatusHeartbeat).mockReturnValue({ stop: heartbeatStop } as never);
});

test('unmount during training cancels native work, service and polling without late contributions', async () => {
  const training = deferred<void>();
  let hooks!: TrainingHooks;
  mockedLoop.mockImplementation(async (_joined, h) => { hooks = h; await training.promise; });
  const mount = new ProviderMount();
  await mount.render().join('project-a');
  const running = mount.render().startTraining();
  await flush();
  expect(hooks.shouldStop()).toBe(false);
  mount.unmount();
  await flush();
  const cancelled = hooks.shouldStop();
  const aborted = jest.mocked(nativeCore.stop).mock.calls.length;
  const stoppedService = jest.mocked(foregroundService.stop).mock.calls.length;
  hooks.onLog('late native callback');
  hooks.onRound({ round: 1 } as never);
  training.resolve(); await running; await flush();
  expect(cancelled).toBe(true);
  expect(aborted).toBeGreaterThan(0);
  expect(stoppedService).toBeGreaterThan(0);
  expect(deactivate).toHaveBeenCalled();
  expect(heartbeatStop).toHaveBeenCalled();
  expect(contributionLedger.record).not.toHaveBeenCalled();
  expect(mount.lateDispatches).toBe(0);
});

test('a new provider cannot enroll until an unmounted provider finishes its late enrollment and cleanup', async () => {
  const enrollment = deferred<JoinedRun>();
  mockedJoin.mockReturnValueOnce(enrollment.promise);
  const old = new ProviderMount();
  const joining = old.render().join('project-a');
  old.unmount(); await flush();
  const next = new ProviderMount();
  await next.render().join('project-b');
  const callsBeforeOldFinished = mockedJoin.mock.calls.length;
  enrollment.resolve(JOINED); await joining; await flush();
  await next.render().join('project-b');
  next.unmount(); await flush();
  expect(callsBeforeOldFinished).toBe(1);
  expect(mockedJoin).toHaveBeenCalledTimes(2);
  expect(nativeCore.stop).toHaveBeenCalled();
  expect(old.lateDispatches).toBe(0);
});

test('the next login stays excluded through the final native abort after old enrollment settles', async () => {
  const enrollment = deferred<JoinedRun>();
  const finalAbort = deferred<void>();
  mockedJoin.mockReturnValueOnce(enrollment.promise);
  jest.mocked(nativeCore.stop).mockResolvedValueOnce(undefined).mockReturnValueOnce(finalAbort.promise);
  const old = new ProviderMount();
  const joining = old.render().join('project-a');
  old.unmount(); await flush();
  enrollment.resolve(JOINED); await joining; await flush();
  expect(nativeCore.stop).toHaveBeenCalledTimes(2);
  const next = new ProviderMount();
  await next.render().join('project-b');
  const callsBeforeCleanup = mockedJoin.mock.calls.length;
  finalAbort.resolve(); await flush();
  await next.render().join('project-b');
  next.unmount(); await flush();
  expect(callsBeforeCleanup).toBe(1);
  expect(mockedJoin).toHaveBeenCalledTimes(2);
});

test('a normal provider rerender keeps training alive and explicit Stop still drains it', async () => {
  const training = deferred<void>();
  let hooks!: TrainingHooks;
  mockedLoop.mockImplementation(async (_joined, h) => { hooks = h; await training.promise; });
  const mount = new ProviderMount();
  await mount.render().join('project-a');
  const running = mount.render().startTraining();
  const value = mount.render();
  expect(hooks.shouldStop()).toBe(false);
  expect(nativeCore.stop).not.toHaveBeenCalled();
  const stopping = value.stopTraining();
  await flush();
  expect(hooks.shouldStop()).toBe(true);
  training.resolve(); await Promise.all([running, stopping]);
  expect(mount.render().state.joined).toBeNull();
  mount.unmount(); await flush();
});

test('repeated unmount and callbacks retained by the old screen cannot start another operation', async () => {
  const mount = new ProviderMount();
  await mount.render().join('project-a');
  const old = mount.render();
  mount.unmount(); mount.unmount(); await flush();
  const aborts = jest.mocked(nativeCore.stop).mock.calls.length;
  await old.join('project-b'); await old.startTraining(); await old.stopTraining();
  expect(mockedJoin).toHaveBeenCalledTimes(1);
  expect(mockedLoop).not.toHaveBeenCalled();
  expect(nativeCore.stop).toHaveBeenCalledTimes(aborts);
  expect(mount.lateDispatches).toBe(0);
});

test('a queued log message after subscription teardown cannot dispatch into the old provider', async () => {
  const mount = new ProviderMount();
  await mount.render().join('project-a');
  mount.render(); await flush();
  const message = subscribe.mock.calls[0][1] as (body: string) => void;
  mount.unmount();
  message('queued before deactivation completed');
  await flush();
  expect(mount.lateDispatches).toBe(0);
});

test('unmount during explicit Stop suppresses its completion and excludes the next provider until drained', async () => {
  const training = deferred<void>();
  mockedLoop.mockImplementation(async () => { await training.promise; });
  const old = new ProviderMount();
  await old.render().join('project-a');
  const value = old.render();
  const running = value.startTraining();
  const stopping = value.stopTraining();
  old.unmount(); await flush();
  const next = new ProviderMount();
  await next.render().join('project-b');
  const callsBeforeDrained = mockedJoin.mock.calls.length;
  training.resolve(); await Promise.all([running, stopping]); await flush();
  await next.render().join('project-b');
  next.unmount(); await flush();
  expect(callsBeforeDrained).toBe(1);
  expect(mockedJoin).toHaveBeenCalledTimes(2);
  expect(old.lateDispatches).toBe(0);
});
