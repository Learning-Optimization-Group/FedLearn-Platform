// Prevent duplicate UI events from starting a second training loop while one is already active.
export class SingleFlight {
  private active = false;
  private stopFlight: Promise<void> | null = null;
  private idleWaiters: Array<() => void> = [];

  async run(task: () => Promise<void>, onBusy?: () => void): Promise<void> {
    if (this.active || this.stopFlight) {
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

  /** Cancel promptly, then clean up any native client created while the operation unwinds. */
  abortAndWait(requestAbort: () => Promise<void>): Promise<void> {
    if (!this.stopFlight) this.stopFlight = this.stopAndWait(requestAbort);
    return this.stopFlight;
  }

  private async stopAndWait(requestAbort: () => Promise<void>): Promise<void> {
    try {
      await requestAbort();
    } finally {
      try {
        await new Promise<void>(resolve => {
          if (this.active) this.idleWaiters.push(resolve);
          else resolve();
        });
        // Enrollment or a reconnect may have replaced the native client after the first abort.
        await requestAbort();
      } finally {
        this.stopFlight = null;
      }
    }
  }
}
