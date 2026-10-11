// Prevent duplicate UI events from starting a second training loop while one is already active.
export class SingleFlight {
  private active = false;
  private idleWaiters: Array<() => void> = [];

  async run(task: () => Promise<void>, onBusy?: () => void): Promise<void> {
    if (this.active) {
      onBusy?.();
      return;
    }
    this.active = true;
    try {
      await task();
    } finally {
      this.active = false;
      const waiters = this.idleWaiters;
      this.idleWaiters = [];
      waiters.forEach(resolve => resolve());
    }
  }

  /** Request cancellation immediately, then wait for the current operation to release its state. */
  async abortAndWait(requestAbort: () => Promise<void>): Promise<void> {
    try {
      await requestAbort();
    } finally {
      await new Promise<void>(resolve => {
        if (this.active) this.idleWaiters.push(resolve);
        else resolve();
      });
    }
  }
}
