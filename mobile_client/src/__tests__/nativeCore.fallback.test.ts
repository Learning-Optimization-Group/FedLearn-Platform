// MO-5: the native FL core TurboModule is absent on any build that didn't compile it (the iOS scaffold).
// Importing the spec must NOT throw at load — it must resolve to a typed fallback whose methods reject
// with a clear message only when actually invoked. Regression test for the launch crash caused by the
// old `TurboModuleRegistry.getEnforcing(...)` (which throws synchronously at module load when absent).
//
// Mock react-native's TurboModuleRegistry.get so we can drive "module absent" (null) vs "registered".
const mockGet = jest.fn();
jest.mock('react-native', () => ({
  TurboModuleRegistry: {
    get: mockGet,
    // Present only to assert we never fall back to the throwing resolver.
    getEnforcing: jest.fn(() => {
      throw new Error('getEnforcing must not be used — it crashes the JS bundle when absent');
    }),
  },
}));

const A_ROUND_CONFIG = {
  strategy: 'DeComFL' as const,
  learningRate: 0.001,
  mu: 0.001,
  numPerturbations: 1,
  numLocalSteps: 1,
  gradEstimateMethod: 'forward' as const,
  seed: 0,
  torchVersion: '',
};

// Load a FRESH copy of the spec so its module-level TurboModuleRegistry.get() re-runs with the mock
// value set for the current test (the resolution happens once, at import).
function loadSpec(): typeof import('@spec/NativeFedLearnCore') {
  let mod: typeof import('@spec/NativeFedLearnCore') | undefined;
  jest.isolateModules(() => {
    // jest.isolateModules swaps the module registry synchronously around this callback, so the
    // re-import must be a synchronous require() — a static import or dynamic import() can't be
    // scoped inside it and would resolve against the wrong (already-restored) registry.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    mod = require('@spec/NativeFedLearnCore');
  });
  return mod as typeof import('@spec/NativeFedLearnCore');
}

describe('NativeFedLearnCore resolution (MO-5)', () => {
  beforeEach(() => {
    mockGet.mockReset();
  });

  test('importing the spec with no native module registered does not throw at load', () => {
    mockGet.mockReturnValue(null);
    let spec: typeof import('@spec/NativeFedLearnCore') | undefined;
    expect(() => {
      spec = loadSpec();
    }).not.toThrow();
    expect(mockGet).toHaveBeenCalledWith('NativeFedLearnCore');
    expect(spec?.isNativeCoreAvailable()).toBe(false);
  });

  test('calling a training method on the fallback rejects with the clear error', async () => {
    mockGet.mockReturnValue(null);
    const core = loadSpec().default;
    await expect(core.runDeComFLRound('run-1', A_ROUND_CONFIG)).rejects.toThrow(
      'native FL core unavailable on this platform',
    );
  });

  test('fallback preserves the Promise<T> contract so .catch(...) callers degrade gracefully', async () => {
    mockGet.mockReturnValue(null);
    const core = loadSpec().default;
    // deviceClass.collectDeviceCapabilities relies on this: the method returns a rejected promise
    // (never a synchronous throw), so `.catch()` runs.
    await expect(core.getDeviceMetrics().catch(() => 'handled')).resolves.toBe('handled');
  });

  test('resolves to the real native module when one is registered', () => {
    const fakeNative = { registerClient: jest.fn() };
    mockGet.mockReturnValue(fakeNative);
    const spec = loadSpec();
    expect(spec.isNativeCoreAvailable()).toBe(true);
    expect(spec.default).toBe(fakeNative);
  });
});
