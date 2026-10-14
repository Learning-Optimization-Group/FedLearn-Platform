// MO-8: one network blip must not end a client's participation. The round loop previously rethrew any
// non-"STOP:" error, so a single transient getServerStatus/runRound rejection tore down the whole run.
// runResilientRoundLoop wraps each iteration in a bounded retry/backoff, and — when consecutive failures
// exhaust the per-round retry budget — a bounded number of rejoins (re-enroll + re-register), before it
// finally gives up. STOP / terminal / cooperative-stop still end cleanly and are never retried.
//
// These drive the state machine through an injected ops object (no native module, no real timers): the
// blip is a fast Promise rejection, which is the common on-device failure shape (a hang needs MO-2's
// per-RPC deadlines — out of scope here and flagged in the impl).
import {
  ModelExecutionFailedError,
  DEFAULT_RESILIENCE,
  runResilientRoundLoop,
  type RoundOps,
  type ResiliencePolicy,
} from '../lib/training';
import type { RoundConfig, RoundResult, ServerStatus } from '../lib/nativeCore';
import { startServerStatusHeartbeat } from '../lib/statusHeartbeat';
import { RunJoinRefusedError } from '../lib/runJoin';

const CFG: RoundConfig = {
  strategy: 'FedAvg',
  learningRate: 0.001,
  mu: 0.001,
  numPerturbations: 1,
  numLocalSteps: 1,
  gradEstimateMethod: 'forward',
  seed: 0,
  torchVersion: '',
  initialStateSha256: '',
  proximalMu: 0,
  batchSize: 0,
  batchSeed: '',
  optimizer: 'sgd',
  adamBeta1: 0.9,
  adamBeta2: 0.999,
  adamEpsilon: 1e-8,
  dropoutRates: [],
  dropoutSeed: '',
  batchNormFeatures: [],
  batchNormMomentum: [],
  batchNormValuesPerExample: [],
};

const POLICY: ResiliencePolicy = {
  maxRoundRetries: 2,
  maxRejoins: 2,
  baseBackoffMs: 10,
  pacingMs: 0,
  rejoinRecoveryRounds: 3,
};

function round(n: number): RoundResult {
  return {
    round: n,
    loss: 1 / n,
    accuracy: -1,
    scalarsTransmitted: 0,
    uplinkBytes: 0,
    downlinkBytes: 0,
    computeMs: 1,
    reverted: false,
  };
}

function hooks() {
  const logs: string[] = [];
  const rounds: RoundResult[] = [];
  return {
    logs,
    rounds,
    onLog: (l: string) => logs.push(l),
    onRound: (r: RoundResult) => rounds.push(r),
    shouldStop: () => false,
  };
}

function baseOps(over: Partial<RoundOps> = {}): RoundOps {
  return {
    getServerStatus: jest.fn().mockResolvedValue({ serverState: 'TRAINING', currentRound: 1 }),
    runFedAvgRound: jest.fn().mockResolvedValue(round(1)),
    runDeComFLRound: jest.fn().mockResolvedValue(round(1)),
    rejoin: jest.fn().mockResolvedValue({ runId: 'r' }),
    delay: jest.fn().mockResolvedValue(undefined),
    loadSubmittedRound: jest.fn().mockResolvedValue(null),
    saveSubmittedRound: jest.fn().mockResolvedValue(undefined),
    runEnded: jest.fn().mockResolvedValue(false),
    ...over,
  };
}

describe('runResilientRoundLoop (MO-8)', () => {
  it.each(['runFedAvgRound', 'runDeComFLRound'] as const)(
    'routes native heartbeat loss to re-registration before another round call (%s)', async (op) => {
      const h = hooks();
      let registered = false;
      let callsBeforeRegistration = 0;
      let checkpoint: number | null = 7;
      const ops = baseOps({
        getServerStatus: async () => ({serverState: h.rounds.length ? 'COMPLETED' : 'TRAINING', currentRound: 8}),
        [op]: async () => {
          if (!registered) {
            callsBeforeRegistration += 1;
            throw {message: 'RECONNECT_REQUIRED: Heartbeat connection lost'};
          }
          return round(8);
        },
        rejoin: async () => { registered = true; return {runId: 'r'}; },
        loadSubmittedRound: async () => checkpoint,
        saveSubmittedRound: async (_id, value) => { checkpoint = value; },
      });
      await runResilientRoundLoop({runId: 'r', isFedAvg: op === 'runFedAvgRound', cfg: CFG}, ops, POLICY, h);
      expect(callsBeforeRegistration).toBe(1);
      expect(h.rounds.map(r => r.round)).toEqual([8]);
      expect(checkpoint).toBe(8);
    });

  it('retains the accepted-round checkpoint across heartbeat-loss re-registration', async () => {
    const h = hooks();
    let registered = false;
    let submittedRoundObserved = false;
    let serverRound = 7;
    let checkpoint: number | null = 7;
    const trained: number[] = [];
    const ops = baseOps({
      getServerStatus: async () => {
        if (!registered) throw new Error('RECONNECT_REQUIRED: Heartbeat connection lost');
        if (serverRound === 7) submittedRoundObserved = true;
        return {serverState: h.rounds.length ? 'COMPLETED' : 'TRAINING', currentRound: serverRound};
      },
      runFedAvgRound: async () => { trained.push(serverRound); return round(serverRound); },
      rejoin: async () => { registered = true; return {runId: 'r'}; },
      delay: async () => { if (registered && submittedRoundObserved) serverRound = 8; },
      loadSubmittedRound: async () => checkpoint,
      saveSubmittedRound: async (_id, value) => { checkpoint = value; },
    });
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops, POLICY, h);
    expect(trained).toEqual([8]);
    expect(checkpoint).toBe(8);
  });

  it('bounds repeated heartbeat-loss recovery even when every re-registration succeeds', async () => {
    let registrations = 0;
    const h = hooks();
    const ops = baseOps({
      runFedAvgRound: async () => { throw new Error('RECONNECT_REQUIRED: Heartbeat connection lost'); },
      rejoin: async () => { registrations += 1; return {runId: 'r'}; },
    });
    await expect(runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops, POLICY, h))
      .rejects.toThrow(/RECONNECT_REQUIRED/);
    expect(registrations).toBe(2);
    expect(h.rounds).toEqual([]);
  });

  it('honors user Stop while waiting to re-register a lost heartbeat connection', async () => {
    let stop = false;
    let registrations = 0;
    const h = {...hooks(), shouldStop: () => stop};
    const ops = baseOps({
      runFedAvgRound: async () => { throw new Error('RECONNECT_REQUIRED: Heartbeat connection lost'); },
      delay: async () => { stop = true; },
      rejoin: async () => { registrations += 1; return {runId: 'r'}; },
    });
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops, POLICY, h);
    expect(registrations).toBe(0);
    expect(h.rounds).toEqual([]);
  });

  it('keeps a partial upload uncheckpointed and re-registers after its heartbeat dies', async () => {
    const h = hooks();
    let checkpoint: number | null = 7;
    let calls = 0;
    let registered = false;
    const checkpointsAtRegistration: (number | null)[] = [];
    const ops = baseOps({
      getServerStatus: async () => ({serverState: h.rounds.length ? 'COMPLETED' : 'TRAINING', currentRound: 8}),
      runFedAvgRound: async () => {
        calls += 1;
        if (calls === 1) throw new Error('SubmitModelUpdateStream failed: gRPC code 4: Deadline Exceeded');
        if (!registered) throw new Error('RECONNECT_REQUIRED: Heartbeat connection lost');
        return round(8);
      },
      rejoin: async () => {
        checkpointsAtRegistration.push(checkpoint);
        registered = true;
        return {runId: 'r'};
      },
      loadSubmittedRound: async () => checkpoint,
      saveSubmittedRound: async (_id, value) => { checkpoint = value; },
    });
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops, POLICY, h);
    expect(calls).toBe(3); // deadline, heartbeat-loss rejection, successful post-registration round
    expect(checkpointsAtRegistration).toEqual([7]);
    expect(h.rounds.map(r => r.round)).toEqual([8]);
    expect(checkpoint).toBe(8);
  });

  it('retires stale training telemetry after the actual terminal server report', async () => {
    const terminal = {serverState: 'TRAINING_COMPLETE', currentRound: 3,
      requiredClientsForRound: 2, receivedUpdatesThisRound: 2, activeClients: 2,
      roundDeadlineUnixMs: 0};
    let displayedState: string | null = 'TRAINING';
    const h = {...hooks(), onRunEnded: () => {
      displayedState = null;
    }};
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG},
      baseOps({getServerStatus: jest.fn().mockResolvedValue(terminal)}), POLICY, h);
    expect(displayedState).toBeNull();
    expect(h.rounds).toEqual([]);
  });

  it('clears stale training telemetry when the backend confirms a departed server run ended', async () => {
    let displayedState: string | null = 'TRAINING';
    const h = {...hooks(), onRunEnded: () => {
      displayedState = null;
    }};
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG},
      baseOps({getServerStatus: jest.fn().mockRejectedValue(new Error('connection closed')),
        runEnded: jest.fn().mockResolvedValue(true)}), POLICY, h);
    expect(displayedState).toBeNull();
    expect(h.rounds).toEqual([]);
  });
  it.each(['runFedAvgRound', 'runDeComFLRound'] as const)(
    'retires this client telemetry on native STOP without retrying (%s)', async (op) => {
      let displayedState: string | null = 'TRAINING';
      let resolveStatus!: (s: ServerStatus) => void;
      const heartbeat = startServerStatusHeartbeat({
        runId: 'r',
        getStatus: () => new Promise<ServerStatus>((resolve) => { resolveStatus = resolve; }),
        onStatus: (s) => { displayedState = s.serverState; },
      });
      const onRunEnded = jest.fn(() => {
        heartbeat.stop();
        displayedState = null;
      });
      const ops = baseOps({ [op]: jest.fn().mockRejectedValue(
        new Error('STOP: server ended participation')) });
      const h = {...hooks(), onRunEnded};

      await runResilientRoundLoop({runId: 'r', isFedAvg: op === 'runFedAvgRound', cfg: CFG},
        ops, POLICY, h);

      expect(displayedState).toBeNull();
      expect(onRunEnded).toHaveBeenCalledTimes(1);
      expect(ops[op]).toHaveBeenCalledTimes(1);
      expect(ops.delay).not.toHaveBeenCalled();
      expect(ops.rejoin).not.toHaveBeenCalled();
      expect(ops.runEnded).not.toHaveBeenCalled();
      expect(h.rounds).toEqual([]);
      resolveStatus({serverState: 'TRAINING', currentRound: 1,
        requiredClientsForRound: 2, receivedUpdatesThisRound: 1,
        activeClients: 2, roundDeadlineUnixMs: 0});
      await Promise.resolve();
      expect(displayedState).toBeNull();
    },
  );

  it('recovers from twenty seconds of immediate connection failures with bounded rejoin backoff', async () => {
    let elapsed = 0;
    let trained = false;
    let registered = false;
    const ops = baseOps({
      delay: jest.fn(async (ms: number) => { elapsed += ms; }),
      getServerStatus: jest.fn(async () => {
        if (elapsed < 20000) throw {message: 'socket unavailable'};
        return {serverState: trained ? 'COMPLETED' : 'TRAINING', currentRound: 1};
      }),
      rejoin: jest.fn(async () => {
        if (elapsed < 20000) throw {message: 'registration unavailable'};
        registered = true;
        return {runId: 'r'};
      }),
      runFedAvgRound: jest.fn(async () => {
        expect(registered).toBe(true); // accepted native registration restores the heartbeat
        trained = true; return round(1);
      }),
    });
    const h = hooks();
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops, DEFAULT_RESILIENCE, h);
    expect(ops.rejoin).toHaveBeenCalledTimes(2);
    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(h.logs.some(line => line.includes('registration unavailable'))).toBe(true);
    expect(elapsed).toBeGreaterThanOrEqual(20000);
    expect(elapsed).toBeLessThan(40000);
  });

  it('refuses to train when status recovers but every registration still fails', async () => {
    let statusCalls = 0;
    let trained = false;
    const ops = baseOps({
      getServerStatus: jest.fn(async () => {
        if (++statusCalls <= DEFAULT_RESILIENCE.maxRoundRetries + 1) throw new Error('socket unavailable');
        return {serverState: trained ? 'COMPLETED' : 'TRAINING', currentRound: 1};
      }),
      rejoin: jest.fn().mockRejectedValue(new Error('registration unavailable')),
      runFedAvgRound: jest.fn(async () => { trained = true; return round(1); }),
    });
    await expect(runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops,
      DEFAULT_RESILIENCE, hooks())).rejects.toThrow('registration unavailable');
    expect(ops.rejoin).toHaveBeenCalledTimes(DEFAULT_RESILIENCE.maxRejoins);
    expect(ops.runFedAvgRound).not.toHaveBeenCalled();
  });

  it('Stop during the first reconnect wait prevents another registration or round', async () => {
    let stop = false;
    const waits: number[] = [];
    const ops = baseOps({
      getServerStatus: jest.fn().mockRejectedValue(new Error('socket unavailable')),
      delay: jest.fn(async (ms: number) => {
        waits.push(ms);
        if (waits.length > DEFAULT_RESILIENCE.maxRoundRetries) stop = true;
      }),
    });
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops,
      DEFAULT_RESILIENCE, {...hooks(), shouldStop: () => stop});
    expect(waits.slice(-1)).toEqual([1000]);
    expect(ops.rejoin).not.toHaveBeenCalled();
    expect(ops.runFedAvgRound).not.toHaveBeenCalled();
  });

  it('cooperative Stop interrupts failed-rejoin backoff within one native wait', async () => {
    let stop = false;
    const rejoin = jest.fn().mockRejectedValue({message: 'registration unavailable'});
    const waits: number[] = [];
    const ops = baseOps({
      getServerStatus: jest.fn().mockRejectedValue(new Error('socket unavailable')),
      rejoin,
      delay: jest.fn(async (ms: number) => {
        if (rejoin.mock.calls.length) { waits.push(ms); stop = true; }
      }),
    });
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops, DEFAULT_RESILIENCE,
      {...hooks(), shouldStop: () => stop});
    expect(rejoin).toHaveBeenCalledTimes(1);
    expect(waits).toEqual([1000]);
    expect(ops.runFedAvgRound).not.toHaveBeenCalled();
  });

  it('stops without training a replacement run when reconnect returns another run ID', async () => {
    const trainedRuns: string[] = [];
    let statusCalls = 0;
    const ops = baseOps({
      getServerStatus: async () => ({ serverState: ++statusCalls <= 2 ? 'TRAINING' : 'COMPLETED', currentRound: 1 }),
      runFedAvgRound: async (runId) => {
        trainedRuns.push(runId);
        if (runId === 'r') throw new Error('connection lost');
        return round(1);
      },
      rejoin: jest.fn().mockResolvedValue({ runId: 'replacement' }),
    });

    await expect(runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops,
      { ...POLICY, maxRoundRetries: 0 }, hooks())).rejects.toThrow(/run.*changed|different run/i);
    expect(trainedRuns).toEqual(['r']);
    expect(ops.rejoin).toHaveBeenCalledTimes(1);
  });

  it('does not retry a contract refusal during reconnect', async () => {
    let statusCalls = 0;
    const failure = new RunJoinRefusedError('Reconnect contract changed');
    const ops = baseOps({
      getServerStatus: async () => ({ serverState: ++statusCalls <= 2 ? 'TRAINING' : 'COMPLETED', currentRound: 1 }),
      runFedAvgRound: jest.fn().mockRejectedValue(new Error('connection lost')),
      rejoin: jest.fn().mockRejectedValue(failure),
    });
    await expect(runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops,
      { ...POLICY, maxRoundRetries: 0 }, hooks())).rejects.toBe(failure);
    expect(ops.rejoin).toHaveBeenCalledTimes(1);
  });
  it('waits in pre-training and aggregation states even before the first upload', async () => {
    const sequence: string[] = [];
    const statuses = [
      { serverState: 'WAITING_FOR_CLIENTS', currentRound: 1 },
      { serverState: 'AGGREGATING', currentRound: 1 },
      { serverState: 'TRAINING', currentRound: 1 },
      { serverState: 'TRAINING_COMPLETE', currentRound: 1 },
    ];
    const ops = baseOps({
      getServerStatus: jest.fn().mockImplementation(async () => {
        const status = statuses.shift() ?? { serverState: 'TRAINING_COMPLETE', currentRound: 1 };
        sequence.push(status.serverState);
        return status;
      }),
      runFedAvgRound: jest.fn().mockImplementation(async () => {
        sequence.push('native-round');
        return round(1);
      }),
    });
    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks());
    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(ops.delay).toHaveBeenCalledTimes(2);
    expect(sequence).toEqual(['WAITING_FOR_CLIENTS', 'AGGREGATING', 'TRAINING', 'native-round', 'TRAINING_COMPLETE']);
  });

  it('rejects an incomplete server status without starting native training', async () => {
    const ops = baseOps({
      getServerStatus: jest.fn().mockResolvedValue({ serverState: 'TRAINING' }),
      runFedAvgRound: jest.fn().mockRejectedValue(new Error('unexpected native training')),
    });
    await expect(runResilientRoundLoop(
      { runId: 'r', isFedAvg: true, cfg: CFG }, ops,
      { ...POLICY, maxRoundRetries: 0, maxRejoins: 0 }, hooks(),
    )).rejects.toThrow(/round number/i);
    expect(ops.runFedAvgRound).not.toHaveBeenCalled();
  });
  it('uploads once and waits for the server round to advance before training again', async () => {
    const statuses = [
      { serverState: 'TRAINING', currentRound: 1 },
      { serverState: 'AGGREGATING', currentRound: 1 },
      { serverState: 'TRAINING', currentRound: 1 },
      { serverState: 'TRAINING', currentRound: 2 },
      { serverState: 'TRAINING_COMPLETE', currentRound: 2 },
    ];
    const getServerStatus = jest.fn().mockImplementation(async () => statuses.shift());
    const runFedAvgRound = jest.fn().mockResolvedValueOnce(round(1)).mockResolvedValueOnce(round(2));
    const ops = baseOps({ getServerStatus, runFedAvgRound });
    const h = hooks();

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    expect(runFedAvgRound).toHaveBeenCalledTimes(2);
    expect(h.rounds.map((r) => r.round)).toEqual([1, 2]);
    expect(ops.delay).toHaveBeenCalledTimes(2);
  });

  it('does not train a previously submitted round after reconnecting', async () => {
    const statuses = [
      { serverState: 'TRAINING', currentRound: 3 },
      { serverState: 'TRAINING', currentRound: 3 },
      { serverState: 'TRAINING', currentRound: 4 },
      { serverState: 'TRAINING_COMPLETE', currentRound: 4 },
    ];
    const ops = baseOps({
      getServerStatus: jest.fn().mockImplementation(async () => statuses.shift()),
      loadSubmittedRound: jest.fn().mockResolvedValue(3),
      saveSubmittedRound: jest.fn().mockResolvedValue(undefined),
      runFedAvgRound: jest.fn().mockResolvedValue(round(4)),
    });

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks());

    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(ops.saveSubmittedRound).toHaveBeenCalledWith('r', 4);
  });

  it('stops rather than retrying an upload when its durable checkpoint cannot be saved', async () => {
    const ops = baseOps({
      getServerStatus: jest.fn().mockResolvedValue({ serverState: 'TRAINING', currentRound: 1 }),
      saveSubmittedRound: jest.fn().mockRejectedValue(new Error('storage unavailable')),
    });
    await expect(
      runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks()),
    ).rejects.toThrow(/checkpoint/i);
    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(ops.rejoin).not.toHaveBeenCalled();
  });
  it('ends cleanly (no retry) when the server state is terminal', async () => {
    const ops = baseOps({ getServerStatus: jest.fn().mockResolvedValue({ serverState: 'COMPLETED', currentRound: 1 }) });
    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks());
    expect(ops.runFedAvgRound).not.toHaveBeenCalled();
    expect(ops.rejoin).not.toHaveBeenCalled();
  });

  it('recognizes a native plain-object STOP instead of retrying it as a network error', async () => {
    const stopped = jest.fn();
    const ops = baseOps({runFedAvgRound: jest.fn().mockRejectedValue({message: 'STOP: abort during local SGD'})});
    await runResilientRoundLoop({runId: 'r', isFedAvg: true, cfg: CFG}, ops, POLICY,
      {...hooks(), onRunEnded: stopped});
    expect(stopped).toHaveBeenCalledTimes(1);
    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(ops.delay).not.toHaveBeenCalled();
    expect(ops.rejoin).not.toHaveBeenCalled();
  });

  it('ends cleanly on a STOP: round rejection without retrying or rejoining', async () => {
    const ops = baseOps({
      getServerStatus: jest.fn().mockResolvedValue({ serverState: 'TRAINING', currentRound: 1 }),
      runFedAvgRound: jest.fn().mockRejectedValue(new Error('STOP: server ended participation')),
    });
    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks());
    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(ops.rejoin).not.toHaveBeenCalled();
  });

  // Stage 3: ExecuTorch refusing to run the model on this device's data is deterministic (a live phone retried
  // NotSupported for about a minute, then rejoined twice, before giving up). It ends training at once.
  it.each(['runFedAvgRound', 'runDeComFLRound'] as const)(
    'stops at once, without retrying or rejoining, when the model cannot run on this data (%s)', async (op) => {
      const cause = new Error('MODEL_EXECUTION: TrainableExecutorchModel: execute_forward_backward failed (error 16)');
      const ops = baseOps({ [op]: jest.fn().mockRejectedValue(cause) });
      const h = hooks();

      const run = runResilientRoundLoop({ runId: 'r', isFedAvg: op === 'runFedAvgRound', cfg: CFG }, ops, POLICY, h);

      await expect(run).rejects.toBeInstanceOf(ModelExecutionFailedError);
      await expect(run).rejects.toThrow(/execute_forward_backward failed \(error 16\)/);
      expect(ops[op]).toHaveBeenCalledTimes(1);
      expect(ops.rejoin).not.toHaveBeenCalled();
      expect(ops.delay).not.toHaveBeenCalled();
      expect(h.logs.some((l) => l.startsWith('Transient error'))).toBe(false);
    });

  // Stage 4 S6: on the live phone the rejection arrived as a plain object carrying `message`, not an Error instance.
  // The loop read it as "[object Object]", missed the prefix, and retried a deterministic failure three times.
  it('stops at once when the model-execution rejection is not an Error instance', async () => {
    const cause = { code: 'E', message: 'MODEL_EXECUTION: ExecutorchModel: set_input(x) failed (error 16)' };
    const ops = baseOps({ runFedAvgRound: jest.fn().mockRejectedValue(cause) });
    const h = hooks();

    const run = runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    await expect(run).rejects.toBeInstanceOf(ModelExecutionFailedError);
    await expect(run).rejects.toThrow(/set_input\(x\) failed \(error 16\)/);
    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(h.logs.some((l) => l.startsWith('Transient error'))).toBe(false);
  });

  it('cooperative stop ends the loop before any work', async () => {
    const ops = baseOps();
    const h = { ...hooks(), shouldStop: () => true };
    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);
    expect(ops.getServerStatus).not.toHaveBeenCalled();
  });

  it('retries a transient round failure with backoff, then continues (blip does NOT end participation)', async () => {
    let calls = 0;
    const runFedAvgRound = jest.fn().mockImplementation(() => {
      calls += 1;
      if (calls === 1) return Promise.reject(new Error('grpc unavailable'));
      if (calls === 2) return Promise.resolve(round(1));
      // Terminal after one good round so the loop ends.
      return Promise.resolve(round(2));
    });
    const getServerStatus = jest
      .fn()
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValue({ serverState: 'COMPLETED', currentRound: 1 });
    const ops = baseOps({ runFedAvgRound, getServerStatus });
    const h = hooks();

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    // The failed attempt was retried (not fatal) and a real round eventually landed.
    expect(runFedAvgRound.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(h.rounds.map((r) => r.round)).toContain(1);
    expect(ops.delay).toHaveBeenCalled(); // backed off before the retry
    expect(ops.rejoin).not.toHaveBeenCalled(); // one blip never escalates to a rejoin
    // The retry says what failed, so a device's log explains a transient error instead of only counting it.
    expect(h.logs).toContain('Transient error (attempt 1/2): grpc unavailable; retrying in 10ms…');
  });

  it('escalates to a bounded rejoin once consecutive failures exhaust the retry budget', async () => {
    // Always-failing round until after a rejoin, at which point it succeeds then completes.
    let rejoined = false;
    const runFedAvgRound = jest.fn().mockImplementation(() => {
      if (!rejoined) return Promise.reject(new Error('grpc unavailable'));
      return Promise.resolve(round(1));
    });
    const rejoin = jest.fn().mockImplementation(() => {
      rejoined = true;
      return Promise.resolve({ runId: 'r' });
    });
    const getServerStatus = jest
      .fn()
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 }) // pre-rejoin attempts
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 }) // post-rejoin
      .mockResolvedValue({ serverState: 'COMPLETED', currentRound: 1 });
    const ops = baseOps({ runFedAvgRound, rejoin, getServerStatus });
    const h = hooks();

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    expect(rejoin).toHaveBeenCalledTimes(1);
    // Reconnect resumes the same run and its qualified state.
    expect(runFedAvgRound.mock.calls.every((c) => c[0] === 'r')).toBe(true);
  });

  it('gives up (throws) only after the rejoin budget is also exhausted', async () => {
    const ops = baseOps({
      runFedAvgRound: jest.fn().mockRejectedValue(new Error('grpc unavailable')),
      rejoin: jest.fn().mockResolvedValue({ runId: 'r' }),
    });
    await expect(
      runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks()),
    ).rejects.toThrow('grpc unavailable');
    // Bounded: exactly maxRejoins attempts, no infinite loop.
    expect((ops.rejoin as jest.Mock).mock.calls.length).toBe(POLICY.maxRejoins);
  });

  it('restores the rejoin budget after a stable streak, surviving more independent blips than the raw cap', async () => {
    // maxRejoins=2, rejoinRecoveryRounds=3. A pure lifetime cap gives up after 2 rejoins; recovery lets a
    // link that re-stabilises (3 good rounds) rejoin AGAIN — so >2 total rejoins occur and the run still
    // completes instead of throwing. Round outcomes come in bursts: 3 fails (→rejoin), then 3 good, repeat.
    let failsLeft = 3;
    let goods = 0;
    let totalGoods = 0;
    const rejoin = jest.fn().mockResolvedValue({ runId: 'r' });
    const runFedAvgRound = jest.fn().mockImplementation(() => {
      if (failsLeft > 0) {
        failsLeft -= 1;
        return Promise.reject(new Error('blip'));
      }
      goods += 1;
      totalGoods += 1;
      if (goods >= 3) {
        goods = 0;
        failsLeft = 3; // after a stable streak, the next blip burst begins
      }
      return Promise.resolve(round(totalGoods));
    });
    // End the run once we've observed the 3rd rejoin (impossible under a lifetime cap of 2).
    const getServerStatus = jest.fn().mockImplementation(() =>
      Promise.resolve({ serverState: rejoin.mock.calls.length >= 3 ? 'COMPLETED' : 'TRAINING',
        currentRound: totalGoods + 1 }),
    );
    const ops = baseOps({ runFedAvgRound, rejoin, getServerStatus });

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks());

    expect(rejoin.mock.calls.length).toBeGreaterThanOrEqual(3);
  });

  it('resets the failure counter after a good round (an early blip does not doom a later one)', async () => {
    // fail, succeed(round1), fail, succeed(round2), then COMPLETED. With maxRoundRetries=2 the two
    // isolated failures must never accumulate into a rejoin.
    const seq = [
      () => Promise.reject(new Error('blip')),
      () => Promise.resolve(round(1)),
      () => Promise.reject(new Error('blip')),
      () => Promise.resolve(round(2)),
    ];
    let i = 0;
    const runFedAvgRound = jest.fn().mockImplementation(() => {
      const step = seq[i];
      i += 1;
      return step ? step() : Promise.resolve(round(3));
    });
    const getServerStatus = jest
      .fn()
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 1 })
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 2 })
      .mockResolvedValueOnce({ serverState: 'TRAINING', currentRound: 2 })
      .mockResolvedValue({ serverState: 'COMPLETED', currentRound: 2 });
    const ops = baseOps({ runFedAvgRound, getServerStatus });
    const h = hooks();

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    expect(ops.rejoin).not.toHaveBeenCalled();
    expect(h.rounds.map((r) => r.round)).toEqual([1, 2]);
  });

  describe('a server gone because the run finished (Stage 4 S7)', () => {
    // The ResNet-18 run's last round returned ~53 s after its upload (the phone evaluated its images), after the server
    // had reported completion for its 3 s drain and exited. The phone saw UNAVAILABLE, rejoined into a 403 and showed a
    // completed run as an error. A failure now first asks the backend whether this run has ended.
    const gone = () => {
      const statuses = [{ serverState: 'TRAINING', currentRound: 2 }];
      return jest.fn().mockImplementation(async () => {
        const next = statuses.shift();
        if (next) return next;
        throw new Error('GetServerStatus failed: gRPC code 14: UNAVAILABLE');
      });
    };

    it('ends cleanly when the backend says the run ended', async () => {
      const ops = baseOps({ getServerStatus: gone(), runFedAvgRound: jest.fn().mockResolvedValue(round(2)),
                            runEnded: jest.fn().mockResolvedValue(true) });
      const h = hooks();
      await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);
      expect(h.rounds.map((r) => r.round)).toEqual([2]);
      expect(ops.runEnded).toHaveBeenCalledWith('r');
      expect(ops.rejoin).not.toHaveBeenCalled();
      expect(h.logs).toContain('Run ended.');
    });

    it('retries as before while the run is still going', async () => {
      const ops = baseOps({ getServerStatus: gone(), runFedAvgRound: jest.fn().mockResolvedValue(round(2)) });
      await expect(runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks()))
        .rejects.toThrow('UNAVAILABLE');
      expect(ops.rejoin).toHaveBeenCalledTimes(POLICY.maxRejoins);
    });

    it('retries as before when the backend cannot be asked', async () => {
      const ops = baseOps({ getServerStatus: gone(), runFedAvgRound: jest.fn().mockResolvedValue(round(2)),
                            runEnded: jest.fn().mockRejectedValue(new Error('network down')) });
      await expect(runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks()))
        .rejects.toThrow('UNAVAILABLE');
      expect(ops.rejoin).toHaveBeenCalledTimes(POLICY.maxRejoins);
    });
  });
});
