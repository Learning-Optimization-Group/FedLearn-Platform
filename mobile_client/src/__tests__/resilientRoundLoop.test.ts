// MO-8: one network blip must not end a client's participation. The round loop previously rethrew any
// non-"STOP:" error, so a single transient getServerStatus/runRound rejection tore down the whole run.
// runResilientRoundLoop wraps each iteration in a bounded retry/backoff, and — when consecutive failures
// exhaust the per-round retry budget — a bounded number of rejoins (re-enroll + re-register), before it
// finally gives up. STOP / terminal / cooperative-stop still end cleanly and are never retried.
//
// These drive the state machine through an injected ops object (no native module, no real timers): the
// blip is a fast Promise rejection, which is the common on-device failure shape (a hang needs MO-2's
// per-RPC deadlines — out of scope here and flagged in the impl).
import { runResilientRoundLoop, type RoundOps, type ResiliencePolicy } from '../lib/training';
import type { RoundConfig, RoundResult } from '../lib/nativeCore';

const CFG: RoundConfig = {
  strategy: 'FedAvg',
  learningRate: 0.001,
  mu: 0.001,
  numPerturbations: 1,
  numLocalSteps: 1,
  gradEstimateMethod: 'forward',
  seed: 0,
  torchVersion: '',
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
    getServerStatus: jest.fn().mockResolvedValue({ serverState: 'TRAINING' }),
    runFedAvgRound: jest.fn().mockResolvedValue(round(1)),
    runDeComFLRound: jest.fn().mockResolvedValue(round(1)),
    rejoin: jest.fn().mockResolvedValue({ runId: 'run-rejoined' }),
    delay: jest.fn().mockResolvedValue(undefined),
    ...over,
  };
}

describe('runResilientRoundLoop (MO-8)', () => {
  it('ends cleanly (no retry) when the server state is terminal', async () => {
    const ops = baseOps({ getServerStatus: jest.fn().mockResolvedValue({ serverState: 'COMPLETED' }) });
    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks());
    expect(ops.runFedAvgRound).not.toHaveBeenCalled();
    expect(ops.rejoin).not.toHaveBeenCalled();
  });

  it('ends cleanly on a STOP: round rejection without retrying or rejoining', async () => {
    const ops = baseOps({
      getServerStatus: jest.fn().mockResolvedValue({ serverState: 'TRAINING' }),
      runFedAvgRound: jest.fn().mockRejectedValue(new Error('STOP: server ended participation')),
    });
    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, hooks());
    expect(ops.runFedAvgRound).toHaveBeenCalledTimes(1);
    expect(ops.rejoin).not.toHaveBeenCalled();
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
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValue({ serverState: 'COMPLETED' });
    const ops = baseOps({ runFedAvgRound, getServerStatus });
    const h = hooks();

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    // The failed attempt was retried (not fatal) and a real round eventually landed.
    expect(runFedAvgRound.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(h.rounds.map((r) => r.round)).toContain(1);
    expect(ops.delay).toHaveBeenCalled(); // backed off before the retry
    expect(ops.rejoin).not.toHaveBeenCalled(); // one blip never escalates to a rejoin
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
      return Promise.resolve({ runId: 'run-2' });
    });
    const getServerStatus = jest
      .fn()
      .mockResolvedValueOnce({ serverState: 'TRAINING' }) // pre-rejoin attempts
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValueOnce({ serverState: 'TRAINING' }) // post-rejoin
      .mockResolvedValue({ serverState: 'COMPLETED' });
    const ops = baseOps({ runFedAvgRound, rejoin, getServerStatus });
    const h = hooks();

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    expect(rejoin).toHaveBeenCalledTimes(1);
    // After the rejoin, the subsequent status/round calls target the NEW run id.
    expect(runFedAvgRound.mock.calls.some((c) => c[0] === 'run-2')).toBe(true);
  });

  it('gives up (throws) only after the rejoin budget is also exhausted', async () => {
    const ops = baseOps({
      runFedAvgRound: jest.fn().mockRejectedValue(new Error('grpc unavailable')),
      rejoin: jest.fn().mockResolvedValue({ runId: 'run-x' }),
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
    const rejoin = jest.fn().mockResolvedValue({ runId: 'run-n' });
    const runFedAvgRound = jest.fn().mockImplementation(() => {
      if (failsLeft > 0) {
        failsLeft -= 1;
        return Promise.reject(new Error('blip'));
      }
      goods += 1;
      const n = goods;
      if (goods >= 3) {
        goods = 0;
        failsLeft = 3; // after a stable streak, the next blip burst begins
      }
      return Promise.resolve(round(n));
    });
    // End the run once we've observed the 3rd rejoin (impossible under a lifetime cap of 2).
    const getServerStatus = jest.fn().mockImplementation(() =>
      Promise.resolve({ serverState: rejoin.mock.calls.length >= 3 ? 'COMPLETED' : 'TRAINING' }),
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
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValueOnce({ serverState: 'TRAINING' })
      .mockResolvedValue({ serverState: 'COMPLETED' });
    const ops = baseOps({ runFedAvgRound, getServerStatus });
    const h = hooks();

    await runResilientRoundLoop({ runId: 'r', isFedAvg: true, cfg: CFG }, ops, POLICY, h);

    expect(ops.rejoin).not.toHaveBeenCalled();
    expect(h.rounds.map((r) => r.round)).toEqual([1, 2]);
  });
});
