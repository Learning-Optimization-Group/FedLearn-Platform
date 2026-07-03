import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// FE-5: Vitest runs the React 19 component/unit suite. Kept separate from vite.config.ts so tests
// don't pull in the dev proxy / tailwind pipeline — only the React plugin (for JSX + Fast Refresh
// transform parity) is needed. jsdom gives components a DOM; CSS is stubbed (we assert behaviour,
// not styles). Test globals are NOT injected — specs import { describe, it, expect, vi } from
// 'vitest' explicitly, so the app tsconfig and eslint flat config need no test-only changes.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    globals: false,
    css: false,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    clearMocks: true,
    restoreMocks: true,
  },
});
