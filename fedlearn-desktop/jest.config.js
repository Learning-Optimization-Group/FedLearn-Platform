// fedlearn-desktop/jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  globals: {
    'ts-jest': {
      tsconfig: '<rootDir>/tsconfig.test.json',
    },
  },
  moduleNameMapper: {
    // Stub out Electron modules — they can't run in Jest
    'electron': '<rootDir>/src/__mocks__/electron.ts',
    'electron-log': '<rootDir>/src/__mocks__/electron-log.ts',
    // Real electron-store resolves its on-disk cwd via Electron's `app.getPath`,
    // which doesn't exist under Jest — stand in with an in-memory Map so
    // AuthService (the only consumer) stays unit-testable with no disk I/O.
    'electron-store': '<rootDir>/src/__mocks__/electron-store.ts',
  },
  // Instrument only the TS sources the `node`-env suite can actually exercise.
  // The renderer `.tsx` components have no jsdom/RTL harness here, so they are
  // intentionally out of scope (tracked separately in the frontend unit).
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/__tests__/**',
    '!src/__mocks__/**',
  ],
  coveragePathIgnorePatterns: [
    '/node_modules/',
    '/dist/',
    'webpack',
    '/__mocks__/',
    '/__tests__/',
  ],
  // Modest regression floor: set a few points below the measured baseline
  // (stmts 36.7 / branch 35.0 / funcs 28.9 / lines 36.7) so it guards against
  // regressions without failing on current code.
  coverageThreshold: {
    global: {
      statements: 33,
      branches: 31,
      functions: 25,
      lines: 33,
    },
  },
};
