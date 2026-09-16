import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';

// ESLint v9 flat config for the Vite + React 19 + TypeScript dashboard.
// Referenced by package.json `lint` ("eslint .") and gated in CI (.github/workflows/ci.yml).
export default tseslint.config(
  { ignores: ['dist', 'node_modules', 'coverage', 'vite.config.ts', 'vitest.config.ts'] },
  {
    files: ['**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: { ...globals.browser, ...globals.es2022 },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // FE-6: promoted from 'warn' to 'error'. These are correctness/hygiene rules, not
      // stylistic ones — `any` erases type safety, unused vars hide typos/half-done refactors,
      // and a non-component export silently breaks fast-refresh. The one legitimate exception
      // (the co-located useAuth hook in AuthContext) carries a scoped, documented eslint-disable.
      'react-refresh/only-export-components': ['error', { allowConstantExport: true }],
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
);
