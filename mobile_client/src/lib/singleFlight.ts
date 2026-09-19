// Prevent duplicate UI events from starting a second training loop while one is already active.
export class SingleFlight {
  private active = false;

  async run(task: () => Promise<void>): Promise<void> {
    if (this.active) return;
    this.active = true;
    try {
      await task();
    } finally {
      this.active = false;
    }
  }
}
