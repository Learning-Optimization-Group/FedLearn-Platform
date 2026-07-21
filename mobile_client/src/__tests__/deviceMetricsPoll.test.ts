// The Home screen's device-metrics poll: the seam that finally feeds DeviceBanner real data
// (it previously received a hard-coded null and could never render). Mirrors the MO-3
// heartbeat contract: immediate first poll, fixed cadence via recursive setTimeout, a failed
// poll delivers null (hide, don't go stale) and never ends polling, stop() suppresses late
// callbacks.
import { startDeviceMetricsPoll } from '../lib/deviceMetricsPoll';
import type { DeviceMetrics } from '../lib/nativeCore';

function metrics(over: Partial<DeviceMetrics> = {}): DeviceMetrics {
  return {
    peakRssBytes: 128 * 1024 * 1024,
    thermalState: 'NOMINAL',
    batteryLevel: 0.8,
    batteryCharging: false,
    ...over,
  };
}

describe('startDeviceMetricsPoll', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('polls immediately, then on the configured interval', async () => {
    const getMetrics = jest.fn().mockResolvedValue(metrics());
    const onMetrics = jest.fn();

    const poll = startDeviceMetricsPoll({ onMetrics, intervalMs: 15000, getMetrics });

    expect(getMetrics).toHaveBeenCalledTimes(1); // immediate, not gated on the interval
    await jest.advanceTimersByTimeAsync(0);
    expect(onMetrics).toHaveBeenLastCalledWith(expect.objectContaining({ thermalState: 'NOMINAL' }));

    await jest.advanceTimersByTimeAsync(15000);
    expect(getMetrics).toHaveBeenCalledTimes(2);
    await jest.advanceTimersByTimeAsync(15000);
    expect(getMetrics).toHaveBeenCalledTimes(3);

    poll.stop();
  });

  it('delivers null on a failed poll and keeps polling (banner hides rather than staling)', async () => {
    const getMetrics = jest
      .fn()
      .mockRejectedValueOnce(new Error('bridge busy'))
      .mockResolvedValue(metrics({ thermalState: 'SERIOUS' }));
    const onMetrics = jest.fn();

    const poll = startDeviceMetricsPoll({ onMetrics, intervalMs: 1000, getMetrics });

    await jest.advanceTimersByTimeAsync(0);
    expect(onMetrics).toHaveBeenLastCalledWith(null);

    await jest.advanceTimersByTimeAsync(1000);
    expect(getMetrics).toHaveBeenCalledTimes(2);
    expect(onMetrics).toHaveBeenLastCalledWith(expect.objectContaining({ thermalState: 'SERIOUS' }));

    poll.stop();
  });

  it('stop() halts polling and suppresses a poll that settles late', async () => {
    let resolve!: (m: DeviceMetrics) => void;
    const getMetrics = jest.fn().mockReturnValue(new Promise<DeviceMetrics>((r) => { resolve = r; }));
    const onMetrics = jest.fn();

    const poll = startDeviceMetricsPoll({ onMetrics, intervalMs: 1000, getMetrics });
    poll.stop();
    resolve(metrics()); // in-flight poll settles after stop
    await jest.advanceTimersByTimeAsync(5000);

    expect(onMetrics).not.toHaveBeenCalled();
    expect(getMetrics).toHaveBeenCalledTimes(1); // never rescheduled
  });
});
