// FE-5: shared Vitest setup, loaded before every spec (vitest.config.ts setupFiles).
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Unmount React trees between tests so queries don't leak across specs (globals are off, so RTL's
// auto-cleanup isn't registered — do it explicitly).
afterEach(() => {
  cleanup();
});

// jsdom omits a few browser APIs that responsive/chart components (recharts) touch on mount. Stub
// them so rendering a real dashboard doesn't throw before we can assert anything.
if (typeof window.matchMedia !== 'function') {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

if (typeof window.ResizeObserver === 'undefined') {
  class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  window.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
}
