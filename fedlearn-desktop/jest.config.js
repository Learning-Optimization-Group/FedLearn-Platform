// fedlearn-desktop/jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.test.ts'],
  moduleNameMapper: {
    // Stub out Electron modules — they can't run in Jest
    'electron': '<rootDir>/src/__mocks__/electron.ts',
    'electron-log': '<rootDir>/src/__mocks__/electron-log.ts',
  },
};
