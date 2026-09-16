// Tests for the training-outcome desktop notifications: transition
// classification plus the permission-guarded notifier, driven with a mocked
// Notification constructor (there is no real Notification in node-env jest).

import {
  classifyRunTransition,
  notifyRunOutcome,
  runOutcomeTitle,
  type NotificationCtorLike,
} from '../renderer/components/runNotifications';
import type { TrainRunStatus } from '../renderer/components/trainFlow';

function makeMockNotification(permission: string, requestResult?: string) {
  const created: string[] = [];
  const requestPermission =
    requestResult === undefined ? undefined : jest.fn(async () => requestResult);

  class Mock {
    static permission = permission;
    static requestPermission = requestPermission;
    constructor(title: string) {
      created.push(title);
    }
  }
  return { ctor: Mock as unknown as NotificationCtorLike, created, requestPermission };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

describe('classifyRunTransition', () => {
  it.each<[TrainRunStatus, TrainRunStatus, string | null]>([
    ['running', 'completed', 'completed'],
    ['running', 'error', 'failed'],
    ['pulling', 'error', 'failed'],
    ['restarting', 'completed', 'completed'],
    ['paused', 'error', 'failed'],
    // Not transitions out of an in-flight run:
    ['idle', 'completed', null],
    ['idle', 'error', null],
    ['stopped', 'error', null],
    ['completed', 'completed', null],
    ['error', 'error', null],
    // In-flight but not a terminal outcome:
    ['running', 'stopped', null],
    ['running', 'idle', null],
    ['pulling', 'running', null],
  ])('%s → %s yields %s', (prev, next, expected) => {
    expect(classifyRunTransition(prev, next)).toBe(expected);
  });
});

describe('runOutcomeTitle', () => {
  it('formats the completed and failed titles', () => {
    expect(runOutcomeTitle('completed', 'Pneumonia CNN')).toBe(
      'FedLearn — training completed: Pneumonia CNN',
    );
    expect(runOutcomeTitle('failed', 'ECG')).toBe('FedLearn — training failed: ECG');
  });
});

describe('notifyRunOutcome', () => {
  it('fires immediately when permission is granted', () => {
    const { ctor, created } = makeMockNotification('granted');
    notifyRunOutcome('completed', 'Pneumonia CNN', ctor);
    expect(created).toEqual(['FedLearn — training completed: Pneumonia CNN']);
  });

  it('stays silent when permission is denied and never re-asks', () => {
    const { ctor, created, requestPermission } = makeMockNotification('denied', 'granted');
    notifyRunOutcome('failed', 'ECG', ctor);
    expect(created).toEqual([]);
    expect(requestPermission).not.toHaveBeenCalled();
  });

  it('asks once on default permission and fires when granted', async () => {
    const { ctor, created, requestPermission } = makeMockNotification('default', 'granted');
    notifyRunOutcome('failed', 'ECG', ctor);
    await flush();
    expect(requestPermission).toHaveBeenCalledTimes(1);
    expect(created).toEqual(['FedLearn — training failed: ECG']);
  });

  it('asks on default permission but stays silent when refused', async () => {
    const { ctor, created } = makeMockNotification('default', 'denied');
    notifyRunOutcome('completed', 'X', ctor);
    await flush();
    expect(created).toEqual([]);
  });

  it('is a no-op without a Notification implementation (no throw)', () => {
    expect(() => notifyRunOutcome('completed', 'X')).not.toThrow();
  });

  it('falls back to the global Notification when no ctor is injected', () => {
    const { ctor, created } = makeMockNotification('granted');
    (globalThis as Record<string, unknown>).Notification = ctor;
    try {
      notifyRunOutcome('completed', 'Global');
      expect(created).toEqual(['FedLearn — training completed: Global']);
    } finally {
      delete (globalThis as Record<string, unknown>).Notification;
    }
  });

  it('never throws even if the constructor itself throws', () => {
    const Throwing = class {
      static permission = 'granted';
      constructor() {
        throw new Error('renderer refused');
      }
    } as unknown as NotificationCtorLike;
    expect(() => notifyRunOutcome('completed', 'X', Throwing)).not.toThrow();
  });
});

describe('transition → notification pipeline (mirrors the TrainSection effect)', () => {
  it('fires exactly once across a full run, including repeated terminal polls', () => {
    const { ctor, created } = makeMockNotification('granted');
    const sequence: TrainRunStatus[] = ['idle', 'pulling', 'running', 'completed', 'completed', 'idle'];

    let prev = sequence[0];
    for (const next of sequence.slice(1)) {
      const outcome = classifyRunTransition(prev, next);
      prev = next;
      if (outcome) notifyRunOutcome(outcome, 'Pneumonia CNN', ctor);
    }

    expect(created).toEqual(['FedLearn — training completed: Pneumonia CNN']);
  });

  it('reports a failure transition as failed', () => {
    const { ctor, created } = makeMockNotification('granted');
    const outcome = classifyRunTransition('pulling', 'error');
    if (outcome) notifyRunOutcome(outcome, 'ECG', ctor);
    expect(created).toEqual(['FedLearn — training failed: ECG']);
  });
});
