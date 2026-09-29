import { assertNativeCompatibility, NativeCompatibilityError } from '../lib/nativeCompatibility';
import type { RuntimeCompatibility } from '../lib/nativeCore';

describe('native execution compatibility', () => {
  const compatible = { bridgeAbiVersion: 3, protocolVersion: 2 };

  test('accepts the compiled bridge and protocol expected by this JavaScript bundle', async () => {
    await expect(assertNativeCompatibility({ getRuntimeCompatibility: async () => compatible })).resolves.toBeUndefined();
  });

  test.each([
    ['missing method', undefined],
    ['old bridge', async () => ({ bridgeAbiVersion: 0, protocolVersion: 2 })],
    // ABI 1 cannot take the round config's minibatch fields (batchSize, batchSeed).
    ['bridge without minibatching', async () => ({ bridgeAbiVersion: 1, protocolVersion: 2 })],
    // ABI 2 has no qualification probe (qualifyTrainable).
    ['bridge without the qualification probe', async () => ({ bridgeAbiVersion: 2, protocolVersion: 2 })],
    ['different protocol', async () => ({ bridgeAbiVersion: 3, protocolVersion: 3 })],
    ['incomplete reply', async () => ({ bridgeAbiVersion: 3 } as RuntimeCompatibility)],
  ])('rejects %s with a rebuild instruction', async (_description, probe) => {
    await expect(assertNativeCompatibility({ getRuntimeCompatibility: probe })).rejects.toThrow(
      NativeCompatibilityError,
    );
    await expect(assertNativeCompatibility({ getRuntimeCompatibility: probe })).rejects.toThrow(/rebuild.*app/i);
  });

  test('wraps a native probe failure without silently provisioning', async () => {
    await expect(
      assertNativeCompatibility({ getRuntimeCompatibility: async () => { throw new Error('native unavailable'); } }),
    ).rejects.toThrow(NativeCompatibilityError);
  });
});
