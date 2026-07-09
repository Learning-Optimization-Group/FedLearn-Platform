import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

// ESLint v9 flat config for the Electron main/preload/renderer TypeScript app.
// Referenced by package.json `lint` ("eslint .") and gated in CI (.github/workflows/ci.yml).
//
// TE-9: unlike frontend/eslint.config.mjs (which softens these two rules to `warn` to keep an
// existing warn-level baseline non-blocking), this config keeps @typescript-eslint/no-explicit-any
// and @typescript-eslint/no-unused-vars at the typescript-eslint "recommended" default of `error` —
// this is a fresh gate with no legacy baseline to preserve, so it starts strict.
export default tseslint.config(
  { ignores: ['dist', 'node_modules', 'coverage', 'release'] },
  {
    files: ['src/**/*.{ts,tsx}'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: { ...globals.es2022 },
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
  {
    // Main + preload run under Node (Electron main process / contextBridge preload).
    files: ['src/main/**/*.ts', 'src/preload/**/*.ts'],
    languageOptions: {
      globals: { ...globals.node },
    },
  },
  {
    // Renderer runs in the Chromium window — browser globals, plus React hooks correctness rules.
    files: ['src/renderer/**/*.{ts,tsx}'],
    languageOptions: {
      globals: { ...globals.browser },
    },
    plugins: {
      'react-hooks': reactHooks,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
    },
  },
  {
    // Shared modules are imported from both Main and Renderer (and from Jest) — see
    // src/shared/urlSecurity.ts's own "must stay importable from every process" contract.
    files: ['src/shared/**/*.ts'],
    languageOptions: {
      globals: { ...globals.node, ...globals.browser },
    },
  },
  {
    // Tests + manual mocks run under Jest (node test environment; see jest.config.js).
    files: ['src/__tests__/**/*.ts', 'src/__mocks__/**/*.ts'],
    languageOptions: {
      globals: { ...globals.node, ...globals.jest },
    },
  },
);
