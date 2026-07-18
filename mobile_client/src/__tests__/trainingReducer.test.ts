// The lifted training state machine (stage 1 of the IA redesign): TrainingScreen's former
// component state as a pure reducer shared by Home/Projects/stage-2 pushes. These tests pin
// the machine transitions AND the behaviors carried over verbatim from the screen: the
// 500-line log ring, the "training error keeps the run joined" rule, and the stop reset
// that clears everything except `error`.
import {
  appendLogLines,
  emptySession,
  initialTrainingState,
  MAX_LOG_LINES,
  trainingReducer,
  type TrainingState,
} from '../state/trainingReducer';
import type { RoundResult } from '../lib/nativeCore';
import type { JoinedRun } from '../lib/runJoin';

function joinedRun(over: Partial<JoinedRun> = {}): JoinedRun {
  return {
    runId: 'run-1',
    projectId: 'proj-1',
    partitionId: 0,
    assignedRound: 1,
    grpcEndpoint: 'host:50051',
    manifest: {
      runId: 'run-1',
      projectId: 'proj-1',
      recipeKey: 'TINYNET_GOLDEN',
      strategy: 'DeComFL',
      numRounds: 10,
      clientsPerRound: 2,
      partitioningMode: 'IID',
      seed: 42,
      torchVersion: '2.0.0',
    },
    message: 'ok',
    ...over,
  };
}

function round(over: Partial<RoundResult> = {}): RoundResult {
  return {
    round: 1,
    loss: 0.5,
    accuracy: -1,
    scalarsTransmitted: 8,
    uplinkBytes: 1024,
    downlinkBytes: 2048,
    computeMs: 1500,
    reverted: false,
    ...over,
  };
}

describe('trainingReducer — join transitions', () => {
  it('starts notJoined with an empty session', () => {
    expect(initialTrainingState.machine).toBe('notJoined');
    expect(initialTrainingState.joined).toBeNull();
    expect(initialTrainingState.session).toEqual(emptySession);
  });

  it('JOIN_START sets joining and clears a previous error (old onJoin behavior)', () => {
    const s = trainingReducer(
      { ...initialTrainingState, error: 'previous' },
      { type: 'JOIN_START' },
    );
    expect(s.joining).toBe(true);
    expect(s.error).toBeNull();
    expect(s.machine).toBe('notJoined');
  });

  it('JOIN_SUCCESS moves to joined and records the project name', () => {
    const s0 = trainingReducer(initialTrainingState, { type: 'JOIN_START' });
    const s = trainingReducer(s0, {
      type: 'JOIN_SUCCESS',
      joined: joinedRun(),
      projectName: 'Pneumonia CNN',
    });
    expect(s.machine).toBe('joined');
    expect(s.joining).toBe(false);
    expect(s.joined?.runId).toBe('run-1');
    expect(s.projectName).toBe('Pneumonia CNN');
  });

  it('JOIN_SUCCESS falls back to the projectId when no name was handed over', () => {
    const s = trainingReducer(initialTrainingState, { type: 'JOIN_SUCCESS', joined: joinedRun() });
    expect(s.projectName).toBe('proj-1');
  });

  it('JOIN_FAILURE is the machine error state (join failed → nothing joined)', () => {
    const s0 = trainingReducer(initialTrainingState, { type: 'JOIN_START' });
    const s = trainingReducer(s0, { type: 'JOIN_FAILURE', error: 'No active run' });
    expect(s.machine).toBe('error');
    expect(s.joining).toBe(false);
    expect(s.error).toBe('No active run');
    expect(s.joined).toBeNull();
  });
});

describe('trainingReducer — training lifecycle', () => {
  const joined: TrainingState = trainingReducer(initialTrainingState, {
    type: 'JOIN_SUCCESS',
    joined: joinedRun(),
  });

  it('TRAINING_START enters training, clears error + latestRound, and restarts the session fold', () => {
    const dirty: TrainingState = {
      ...joined,
      error: 'old',
      latestRound: round(),
      session: { rounds: 3, scalarsTransmitted: 9, bytesUp: 1, bytesDown: 2, computeMs: 3 },
    };
    const s = trainingReducer(dirty, { type: 'TRAINING_START' });
    expect(s.machine).toBe('training');
    expect(s.error).toBeNull();
    expect(s.latestRound).toBeNull();
    expect(s.session).toEqual(emptySession);
  });

  it('ROUND_RESULT sets latestRound and folds the session totals', () => {
    let s = trainingReducer(joined, { type: 'TRAINING_START' });
    s = trainingReducer(s, { type: 'ROUND_RESULT', round: round({ round: 1 }) });
    s = trainingReducer(s, {
      type: 'ROUND_RESULT',
      round: round({ round: 2, loss: 0.4, scalarsTransmitted: 4, uplinkBytes: 100, downlinkBytes: 200, computeMs: 500 }),
    });
    expect(s.latestRound?.round).toBe(2);
    expect(s.session).toEqual({
      rounds: 2,
      scalarsTransmitted: 12,
      bytesUp: 1124,
      bytesDown: 2248,
      computeMs: 2000,
    });
  });

  it('TRAINING_ERROR keeps the run joined (machine stays training here; the old screen kept phase ready) and appends a WARN line', () => {
    let s = trainingReducer(joined, { type: 'TRAINING_START' });
    s = trainingReducer(s, { type: 'TRAINING_ERROR', error: 'gRPC unavailable' });
    expect(s.error).toBe('gRPC unavailable');
    expect(s.joined).not.toBeNull();
    expect(s.machine).not.toBe('error'); // a training failure is NOT the join-error state
    expect(s.logs[s.logs.length - 1]).toEqual({ level: 'WARN', text: 'gRPC unavailable' });
  });

  it('TRAINING_END returns to joined while a run is still joined', () => {
    let s = trainingReducer(joined, { type: 'TRAINING_START' });
    s = trainingReducer(s, { type: 'TRAINING_END' });
    expect(s.machine).toBe('joined');
  });

  it('TRAINING_END after a stop tore the run down stays notJoined (loop finally races STOP_COMPLETE)', () => {
    let s = trainingReducer(joined, { type: 'TRAINING_START' });
    s = trainingReducer(s, { type: 'STOP_START' });
    s = trainingReducer(s, { type: 'STOP_COMPLETE' });
    s = trainingReducer(s, { type: 'TRAINING_END' });
    expect(s.machine).toBe('notJoined');
    expect(s.joined).toBeNull();
  });
});

describe('trainingReducer — stop semantics', () => {
  it('STOP_COMPLETE resets everything except error (the old onStop never cleared it)', () => {
    let s = trainingReducer(initialTrainingState, { type: 'JOIN_SUCCESS', joined: joinedRun() });
    s = trainingReducer(s, { type: 'TRAINING_START' });
    s = trainingReducer(s, { type: 'ROUND_RESULT', round: round() });
    s = trainingReducer(s, { type: 'LOG_APPEND', body: 'line' });
    s = trainingReducer(s, { type: 'SERVER_STATUS', status: {
      serverState: 'TRAINING', currentRound: 2, requiredClientsForRound: 2,
      receivedUpdatesThisRound: 1, activeClients: 2, roundDeadlineUnixMs: 0,
    } });
    s = trainingReducer(s, { type: 'TRAINING_ERROR', error: 'kept' });
    s = trainingReducer(s, { type: 'STOP_START' });
    expect(s.stopping).toBe(true);
    s = trainingReducer(s, { type: 'STOP_COMPLETE' });
    expect(s).toEqual({ ...initialTrainingState, error: 'kept' });
  });
});

describe('trainingReducer — log ring', () => {
  it('LOG_APPEND splits multi-line bodies, drops empties, and tags a level', () => {
    const s = trainingReducer(initialTrainingState, {
      type: 'LOG_APPEND',
      body: 'a\n\nb\n',
      level: 'INFO',
    });
    expect(s.logs).toEqual([
      { level: 'INFO', text: 'a' },
      { level: 'INFO', text: 'b' },
    ]);
  });

  it('LOGS_CLEAR empties the ring (the STOMP effect clears on each new subscription)', () => {
    let s = trainingReducer(initialTrainingState, { type: 'LOG_APPEND', body: 'x' });
    s = trainingReducer(s, { type: 'LOGS_CLEAR' });
    expect(s.logs).toEqual([]);
  });

  it(`caps the ring at ${MAX_LOG_LINES} lines, keeping the newest`, () => {
    const many = Array.from({ length: MAX_LOG_LINES + 20 }, (_, i) => `line-${i}`).join('\n');
    const logs = appendLogLines([], many);
    expect(logs).toHaveLength(MAX_LOG_LINES);
    expect(logs[0]?.text).toBe('line-20'); // the oldest 20 rolled off
    expect(logs[logs.length - 1]?.text).toBe(`line-${MAX_LOG_LINES + 19}`);
  });

  it('appendLogLines returns the previous array untouched for an all-empty body', () => {
    const prev = [{ text: 'keep' }];
    expect(appendLogLines(prev, '\n\n')).toBe(prev);
  });
});

describe('trainingReducer — server status', () => {
  it('SERVER_STATUS stores and clears the heartbeat value', () => {
    const status = {
      serverState: 'TRAINING',
      currentRound: 5,
      requiredClientsForRound: 4,
      receivedUpdatesThisRound: 2,
      activeClients: 3,
      roundDeadlineUnixMs: 123,
    };
    let s = trainingReducer(initialTrainingState, { type: 'SERVER_STATUS', status });
    expect(s.serverStatus).toEqual(status);
    s = trainingReducer(s, { type: 'SERVER_STATUS', status: null });
    expect(s.serverStatus).toBeNull();
  });
});
