// Minimal in-memory stand-in for `electron-store` used under Jest.
//
// The real package resolves its on-disk `cwd` via Electron's `app.getPath`,
// which doesn't exist outside a running Electron process — instantiating it
// under Jest would throw (or, if patched, write real files to the test
// runner's disk). AuthService is the sole consumer (get/set/delete on a
// couple of top-level keys), so a Map-backed default export preserves that
// contract with no disk I/O and no cross-instance leakage between tests.
export default class Store<T extends Record<string, unknown> = Record<string, unknown>> {
  private readonly data = new Map<string, unknown>();

  constructor(_options?: unknown) {}

  get<K extends keyof T & string>(key: K): T[K] | undefined {
    return this.data.get(key) as T[K] | undefined;
  }

  set<K extends keyof T & string>(key: K, value: T[K]): void {
    this.data.set(key, value);
  }

  delete<K extends keyof T & string>(key: K): void {
    this.data.delete(key);
  }
}
