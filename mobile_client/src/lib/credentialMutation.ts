let pending: Promise<void> = Promise.resolve();

/** Keep conditional reads and writes together across asynchronous encrypted-storage calls. */
export function mutateCredentials<T>(action: () => Promise<T>): Promise<T> {
  const result = pending.then(action);
  pending = result.then(() => undefined, () => undefined);
  return result;
}

export function waitForCredentialMutations(): Promise<void> { return pending; }
