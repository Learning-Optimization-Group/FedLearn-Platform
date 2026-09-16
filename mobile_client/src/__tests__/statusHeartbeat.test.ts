// MO-3: the training screen must show the server's LIVE round number + round deadline, refreshed on a
// heartbeat that ticks independently of the on-device round cadence (a DeComFL round can take many
// seconds, during which the server may advance rounds / the deadline counts down). The heartbeat polls
// getServerStatus on its own interval, surfaces every result, and — critically — keeps polling after a
// transient failure (a blip must not silently freeze the live view). These tests pin that timer
// behavior with fake timers + an injected getStatus, and the pure deadline formatter's edges.
import { startServerStatusHeartbeat, formatRoundDeadline } from '../lib/statusHeartbeat';
import type { ServerStatus } from '../lib/nativeCore';

function status(over: Partial<ServerStatus> = {}): ServerStatus {
  return {
    serverState: 'TRAINING',
    currentRound: 3,
    requiredClientsForRound: 4,
    receivedUpdatesThisRound: 1,
    activeClients: 2,
    roundDeadlineUnixMs: 0,
    ...over,
  };
}

describe('startServerStatusHeartbeat (MO-3)', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('polls immediately on start, then again every intervalMs (independent of any round loop)', async () => {
    const getStatus = jest.fn().mockResolvedValue(status());
    const onStatus = jest.fn();

    const hb = startServerStatusHeartbeat({ runId: 'run-1', intervalMs: 3000, getStatus, onStatus });

    // Immediate first poll — not gated on the interval.
    expect(getStatus).toHaveBeenCalledTimes(1);
    await jest.advanceTimersByTimeAsync(0);
    expect(onStatus).toHaveBeenCalledTimes(1);
    expect(onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ currentRound: 3 }));

    await jest.advanceTimersByTimeAsync(3000);
    expect(getStatus).toHaveBeenCalledTimes(2);
    await jest.advanceTimersByTimeAsync(3000);
    expect(getStatus).toHaveBeenCalledTimes(3);

    hb.stop();
  });

  it('keeps polling after a transient error (one blip must not end the heartbeat)', async () => {
    const getStatus = jest
      .fn()
      .mockRejectedValueOnce(new Error('network blip'))
      .mockResolvedValue(status({ currentRound: 7 }));
    const onStatus = jest.fn();
    const onError = jest.fn();

    const hb = startServerStatusHeartbeat({ runId: 'run-1', intervalMs: 1000, getStatus, onStatus, onError });

    await jest.advanceTimersByTimeAsync(0);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onStatus).not.toHaveBeenCalled(); // first tick failed

    // The next tick must still be scheduled despite the failure.
    await jest.advanceTimersByTimeAsync(1000);
    expect(getStatus).toHaveBeenCalledTimes(2);
    expect(onStatus).toHaveBeenCalledWith(expect.objectContaining({ currentRound: 7 }));

    hb.stop();
  });

  it('stop() halts further polling and suppresses late callbacks', async () => {
    const getStatus = jest.fn().mockResolvedValue(status());
    const onStatus = jest.fn();

    const hb = startServerStatusHeartbeat({ runId: 'run-1', intervalMs: 1000, getStatus, onStatus });
    expect(getStatus).toHaveBeenCalledTimes(1);

    hb.stop();
    await jest.advanceTimersByTimeAsync(5000);
    // No further polls after stop.
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  it('a status arriving after stop() is not delivered to onStatus', async () => {
    let resolve!: (s: ServerStatus) => void;
    const getStatus = jest.fn().mockReturnValue(new Promise<ServerStatus>((r) => { resolve = r; }));
    const onStatus = jest.fn();

    const hb = startServerStatusHeartbeat({ runId: 'run-1', intervalMs: 1000, getStatus, onStatus });
    hb.stop();
    resolve(status()); // in-flight poll settles after stop
    await jest.advanceTimersByTimeAsync(0);
    expect(onStatus).not.toHaveBeenCalled();
  });
});

describe('formatRoundDeadline (MO-3)', () => {
  it('renders an em dash when there is no deadline', () => {
    expect(formatRoundDeadline(0, 1_000_000)).toBe('—');
  });

  it('renders a seconds countdown under a minute', () => {
    expect(formatRoundDeadline(1_012_000, 1_000_000)).toBe('closes in 12s');
  });

  it('renders minutes + seconds at or over a minute', () => {
    expect(formatRoundDeadline(1_090_000, 1_000_000)).toBe('closes in 1m 30s');
  });

  it('renders a closing state once the deadline has passed', () => {
    expect(formatRoundDeadline(999_000, 1_000_000)).toBe('closing…');
  });
});
