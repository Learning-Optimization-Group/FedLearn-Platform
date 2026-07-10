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
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'lcov'],
      include: ['src/**/*.{ts,tsx}'],
      // Excluded from the denominator: specs + their fixtures, config/entry files, and
      // type-only declarations carry no runtime behaviour worth a coverage number.
      exclude: [
        'src/**/*.{test,spec}.{ts,tsx}',
        'src/test/**',
        'src/**/*.d.ts',
        'src/main.tsx',
        'src/vite-env.d.ts',
        '**/*.config.*',
      ],
      // Regression floor — set a few points below the measured baseline
      // (stmts/lines 58.08%, branches 70.95%, funcs 38.07% as of this commit).
      // Green on current code; the intent is to catch coverage drops, not to
      // pin an aspirational target. Ratchet up as the suite grows.
      thresholds: {
        lines: 54,
        statements: 54,
        functions: 34,
        branches: 66,
      },
    },
  },
});
