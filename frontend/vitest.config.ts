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
      // Vitest 4's AST remapping changes the baseline on the same 64 files / 138 tests:
      // branches 723/959 (v3) -> 804/1856 (v4); statements now 52.94%, branches 43.31%.
      // https://v4.vitest.dev/guide/migration.html#v8-code-coverage-major-changes
      // Recalibrate those two floors; retain the existing line/function floors.
      thresholds: {
        lines: 54,
        statements: 52,
        functions: 34,
        branches: 42,
      },
    },
  },
});
