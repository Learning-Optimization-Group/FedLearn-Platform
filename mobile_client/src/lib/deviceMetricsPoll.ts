// Live device-metrics poll for the Home screen's DeviceBanner (thermal / battery).
//
// The native TurboModule DOES expose a metrics API — getDeviceMetrics() in
// bridge/specs/NativeFedLearnCore.ts (peakRssBytes, thermalState, batteryLevel,
// batteryCharging) — but until now the only caller was the one-shot eligibility probe
// (deviceClass.collectDeviceCapabilities), and the training screen rendered
// <DeviceBanner metrics={null} />, so the banner was dead code in the live UI.
//
// This poller mirrors statusHeartbeat.ts: immediate first poll, then a recursive setTimeout
// (a slow poll can never overlap the next), best-effort — a failed poll delivers `null`
// (the banner hides rather than showing stale data) and polling continues. Injectable
// getMetrics seam for unit tests; stop() suppresses late callbacks.
import nativeCore, { type DeviceMetrics } from './nativeCore';

export interface DeviceMetricsPollOptions {
  /** Delivered every poll: the fresh metrics, or null when the poll failed. */
  onMetrics: (metrics: DeviceMetrics | null) => void;
  /** Poll cadence in ms (default 15000 — device state moves slowly; keep the bridge quiet). */
  intervalMs?: number;
  /** Injectable metrics source (defaults to the native TurboModule call). */
  getMetrics?: () => Promise<DeviceMetrics>;
}

export interface DeviceMetricsPollHandle {
  /** Halt polling; suppresses any in-flight poll's callback. Safe to call more than once. */
  stop: () => void;
}

export const DEFAULT_METRICS_INTERVAL_MS = 15000;

/**
 * Start polling device metrics on a fixed interval. Polls immediately, then every
 * `intervalMs`. The caller gates this on isNativeCoreAvailable() — on builds without the
 * native core every poll would just reject (and deliver null).
 */
export function startDeviceMetricsPoll(opts: DeviceMetricsPollOptions): DeviceMetricsPollHandle {
  const intervalMs = opts.intervalMs ?? DEFAULT_METRICS_INTERVAL_MS;
  const getMetrics = opts.getMetrics ?? (() => nativeCore.getDeviceMetrics());

  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const tick = async (): Promise<void> => {
    let metrics: DeviceMetrics | null = null;
    try {
      metrics = await getMetrics();
    } catch {
      metrics = null; // best-effort: hide the banner rather than show stale state
    }
    if (stopped) return; // a poll settling after stop() must not deliver or reschedule
    opts.onMetrics(metrics);
    timer = setTimeout(() => {
      void tick();
    }, intervalMs);
  };

  void tick();

  return {
    stop() {
      stopped = true;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    },
  };
}
