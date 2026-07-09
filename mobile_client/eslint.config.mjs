import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

// ESLint v9 flat config for the React Native (Hermes) TypeScript app.
// Referenced by package.json `lint` ("eslint .") and gated in CI (.github/workflows/ci.yml, mobile-js job).
//
// TE-9: unlike frontend/eslint.config.mjs (which softens these two rules to `warn` to keep an
// existing warn-level baseline non-blocking), this config keeps @typescript-eslint/no-explicit-any
// and @typescript-eslint/no-unused-vars at the typescript-eslint "recommended" default of `error` —
// this is a fresh gate with no legacy baseline to preserve, so it starts strict.
export default tseslint.config(
  {
    ignores: [
      'node_modules',
      'android',
      'ios',
      'build', // native CMake build dir (shared/, scripts/pte_export.py etc.), not JS output
      'scratch',
      'coverage',
    ],
  },
  {
    files: ['src/**/*.{ts,tsx}', 'bridge/specs/**/*.ts'],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: {
        // RN's Hermes runtime exposes the same fetch/WebSocket/console/timer surface as a
        // browser (globals.browser); __DEV__ and `global` are RN-specific additions on top.
        ...globals.browser,
        __DEV__: 'readonly',
        global: 'readonly',
      },
    },
    plugins: {
      'react-hooks': reactHooks,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    },
  },
  {
    // Tests run under the `react-native` Jest preset (see package.json `jest` block).
    files: ['src/__tests__/**/*.ts'],
    languageOptions: {
      globals: { ...globals.jest },
    },
  },
  {
    // TE-9 follow-up: root-level CommonJS build/config files (babel/metro/react-native/tailwind
    // config) previously matched no `files` block above and so got zero rule enforcement. These
    // are plain Node CommonJS, not TypeScript — recommended JS rules only, with Node globals
    // (module/require/__dirname are already in `globals.node`).
    files: ['*.{js,cjs}', 'scripts/**/*.js'],   // include scripts/** for parity with desktop (future-proof)
    extends: [js.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'commonjs',
      globals: { ...globals.node },
    },
  },
  {
    // index.js (RN entry point) and the flat config file itself use `import`/`export` — ESM —
    // even though they're plain `.js`/`.mjs`, not TypeScript, so override sourceType back to
    // 'module' for just these two.
    files: ['index.js', '*.mjs'],
    extends: [js.configs.recommended],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: { ...globals.node },
    },
  },
);
