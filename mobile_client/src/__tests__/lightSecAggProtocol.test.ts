import {
  LIGHTSECAGG_PRIME,
  mod,
  modInverse,
  quantize,
  dequantize,
  computeLightSecAggShares,
  aggregateLightSecAggShares,
  reconstructSecret,
} from '../lib/lightSecAgg';

describe('LightSecAgg Mobile Protocol Unit Tests', () => {
  it('correctly normalizes modulo p = 2^31 - 1', () => {
    expect(LIGHTSECAGG_PRIME).toBe(2147483647n);
    expect(mod(5n)).toBe(5n);
    expect(mod(-1n)).toBe(2147483646n);
    expect(mod(2147483647n)).toBe(0n);
    expect(mod(2147483648n)).toBe(1n);
  });

  it('correctly computes modular inverse via Fermat Little Theorem', () => {
    const a = 12345n;
    const inv = modInverse(a);
    expect(mod(a * inv)).toBe(1n);

    expect(() => modInverse(0n)).toThrow('Zero has no modular inverse');
  });

  it('preserves precision through quantize and dequantize', () => {
    const val = 3.14159;
    const q = quantize(val);
    const recovered = dequantize(q);
    expect(recovered).toBeCloseTo(val, 5);

    const negativeVal = -2.71828;
    const qNeg = quantize(negativeVal);
    const recoveredNeg = dequantize(qNeg);
    expect(recoveredNeg).toBeCloseTo(negativeVal, 5);
  });

  it('correctly splits secret into Shamir shares and reconstructs at threshold', async () => {
    const secret = [0.123, -0.456, 0.789, 0.0];
    const numShares = 5;
    const threshold = 3;

    const shares = await computeLightSecAggShares(secret, numShares, threshold);
    expect(shares.length).toBe(numShares);
    expect(shares[0].length).toBe(secret.length);

    // Any subset of threshold shares reconstructs the secret
    const subsetXs = [1, 2, 4];
    const subsetShares = [shares[0], shares[1], shares[3]];

    const reconstructed = reconstructSecret(subsetXs, subsetShares, threshold);
    for (let i = 0; i < secret.length; i++) {
      expect(reconstructed[i]).toBeCloseTo(secret[i], 4);
    }
  });

  it('aggregates shares across clients coordinate-wise', async () => {
    const sharesA = [[10, 20, 30]];
    const sharesB = [[5, 15, 25]];
    const aggregated = await aggregateLightSecAggShares([sharesA[0], sharesB[0]]);
    expect(aggregated).toEqual([15, 35, 55]);
  });

  it('rejects invalid threshold configurations', async () => {
    await expect(computeLightSecAggShares([1.0], 3, 4)).rejects.toThrow('Invalid threshold');
    await expect(computeLightSecAggShares([1.0], 3, 0)).rejects.toThrow('Invalid threshold');
  });
});
