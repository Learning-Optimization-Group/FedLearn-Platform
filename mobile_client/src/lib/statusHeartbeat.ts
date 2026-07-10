// MO-3: a live server-status heartbeat, decoupled from the on-device training round loop.
//
// runTrainingLoop (training.ts) polls getServerStatus only ONCE per round iteration, and only to check
// for a terminal serverState — the currentRound / roundDeadlineUnixMs it returns are discarded, and a
// DeComFL round can occupy the loop for many seconds. So the UI's only "round" signal was the client's
// last-COMPLETED local round, which freezes during a long round and never shows the server's deadline.
//
// This heartbeat ticks on its OWN interval (independent of the round/send cadence), surfacing the
// server's live currentRound + roundDeadlineUnixMs (+ participation counts) so the training screen can
// render an honest, continuously-updating view. It is best-effort: a transient getServerStatus failure
// is reported but never ends the heartbeat — the next tick is always scheduled — so one network blip
// doesn't silently freeze the live view.
import nativeCore, { type ServerStatus } from './nativeCore';

export interface ServerStatusHeartbeatOptions {
  runId: string;
  /** Delivered every successful poll. */
  onStatus: (status: ServerStatus) => void;
  /** Delivered on a poll failure. The heartbeat keeps running regardless. */
  onError?: (error: unknown) => void;
  /** Poll cadence in ms (default 3000). Independent of the training round cadence. */
  intervalMs?: number;
  /** Injectable poll (defaults to the native gRPC status RPC) — the seam unit tests drive. */
  getStatus?: (runId: string) => Promise<ServerStatus>;
}

export interface ServerStatusHeartbeatHandle {
  /** Halt polling; suppresses any in-flight poll's callback. Safe to call more than once. */
  stop: () => void;
}

const DEFAULT_INTERVAL_MS = 3000;

/**
 * Start polling the server run's status on a fixed interval, independent of the training loop.
 * Polls immediately (so the first status isn't gated on the interval), then every `intervalMs`.
 * Returns a handle whose `stop()` clears the timer and prevents any late callback from firing.
 */
export function startServerStatusHeartbeat(
  opts: ServerStatusHeartbeatOptions,
): ServerStatusHeartbeatHandle {
  const intervalMs = opts.intervalMs ?? DEFAULT_INTERVAL_MS;
  const getStatus = opts.getStatus ?? ((runId: string) => nativeCore.getServerStatus(runId));

  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const tick = async (): Promise<void> => {
    let status: ServerStatus | null = null;
    try {
      status = await getStatus(opts.runId);
    } catch (error) {
      // A blip is expected and non-fatal — report it, but do NOT stop the heartbeat.
      if (!stopped) opts.onError?.(error);
    }
    // A poll that settles after stop() must not deliver a stale callback or reschedule.
    if (stopped) return;
    if (status) opts.onStatus(status);
    // Recursive setTimeout (not setInterval) so a slow poll can't overlap the next one.
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

/**
 * Format the live round-deadline countdown for display. Pure (takes `nowMs`) so it's testable and so a
 * 1s UI ticker can re-render it. Returns an em dash when there is no deadline, "closing…" once it has
 * passed, and a "closes in …" countdown otherwise.
 */
export function formatRoundDeadline(deadlineUnixMs: number, nowMs: number): string {
  if (!deadlineUnixMs || deadlineUnixMs <= 0) return '—';
  const remainingMs = deadlineUnixMs - nowMs;
  if (remainingMs <= 0) return 'closing…';
  const totalSeconds = Math.ceil(remainingMs / 1000);
  if (totalSeconds < 60) return `closes in ${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `closes in ${minutes}m ${seconds}s`;
}
