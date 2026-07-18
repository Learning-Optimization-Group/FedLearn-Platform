// =============================================================================
// FedLearn Desktop — Run-outcome notifications (renderer-only)
// =============================================================================
// Fires an HTML5 Notification when a training run transitions to completed or
// failed. Pure transition classification + a guarded, injectable notifier so
// the node-env jest suite can drive it with a mocked Notification constructor.
// No main-process involvement.
// =============================================================================

import { ACTIVE_STATUSES, type TrainRunStatus } from './trainFlow';

export type RunOutcome = 'completed' | 'failed';

/**
 * A run "completes" or "fails" only when leaving an in-flight status — App can
 * also set 'error' during setup (start rejected before 'pulling'), and status
 * polling may repeat terminal states; neither should notify.
 */
export function classifyRunTransition(prev: TrainRunStatus, next: TrainRunStatus): RunOutcome | null {
  if (prev === next || !ACTIVE_STATUSES.has(prev)) return null;
  if (next === 'completed') return 'completed';
  if (next === 'error') return 'failed';
  return null;
}

/** The subset of the HTML5 Notification API this module touches. */
export interface NotificationCtorLike {
  new (title: string, options?: { body?: string; tag?: string }): unknown;
  permission: string;
  requestPermission?: () => Promise<string>;
}

export function runOutcomeTitle(outcome: RunOutcome, projectName: string): string {
  return outcome === 'completed'
    ? `FedLearn — training completed: ${projectName}`
    : `FedLearn — training failed: ${projectName}`;
}

/**
 * Best-effort desktop notification, guarded by Notification.permission:
 * granted → fire; denied → stay silent; default → ask once, fire if granted.
 * Never throws (notifications must not break the training UI).
 */
export function notifyRunOutcome(
  outcome: RunOutcome,
  projectName: string,
  ctor?: NotificationCtorLike,
): void {
  const N =
    ctor ??
    (typeof Notification !== 'undefined' ? (Notification as unknown as NotificationCtorLike) : undefined);
  if (!N) return;

  const title = runOutcomeTitle(outcome, projectName);
  try {
    if (N.permission === 'granted') {
      new N(title, { tag: 'fedlearn-run-outcome' });
      return;
    }
    if (N.permission !== 'denied' && typeof N.requestPermission === 'function') {
      N.requestPermission()
        .then((p) => {
          if (p === 'granted') new N(title, { tag: 'fedlearn-run-outcome' });
        })
        .catch(() => {
          /* best-effort */
        });
    }
  } catch {
    /* best-effort */
  }
}
