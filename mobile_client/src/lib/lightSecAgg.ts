/**
 * LightSecAgg client protocol library for mobile federated learning.
 *
 * Implements modular arithmetic modulo the Mersenne prime p = 2^31 - 1,
 * Shamir t-out-of-N secret sharing, aggregate share computation,
 * and perturbation vector masking.
 */
import nativeCore, { isNativeCoreAvailable } from './nativeCore';

export const LIGHTSECAGG_PRIME = 2147483647n; // 2^31 - 1
export const DEFAULT_SCALE = 1000000n; // 1e6

export function mod(val: bigint, p: bigint = LIGHTSECAGG_PRIME): bigint {
  const r = val % p;
  return r < 0n ? r + p : r;
}

export function modInverse(a: bigint, p: bigint = LIGHTSECAGG_PRIME): bigint {
  const aNorm = mod(a, p);
  if (aNorm === 0n) {
    throw new Error('Zero has no modular inverse');
  }
  // Fermat's Little Theorem: a^(p-2) mod p
  let exp = p - 2n;
  let base = aNorm;
  let result = 1n;
  while (exp > 0n) {
    if (exp % 2n === 1n) {
      result = (result * base) % p;
    }
    base = (base * base) % p;
    exp = exp / 2n;
  }
  return result;
}

export function quantize(value: number, scale: bigint = DEFAULT_SCALE, p: bigint = LIGHTSECAGG_PRIME): bigint {
  const scaled = BigInt(Math.round(value * Number(scale)));
  return mod(scaled, p);
}

export function dequantize(val: bigint, scale: bigint = DEFAULT_SCALE, p: bigint = LIGHTSECAGG_PRIME): number {
  const normalized = mod(val, p);
  const signed = normalized > p / 2n ? normalized - p : normalized;
  return Number(signed) / Number(scale);
}

function getSecureRandomFieldElement(p: bigint): bigint {
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const buf = new Uint32Array(1);
    const limit = (0x100000000n / p) * p;
    while (true) {
      crypto.getRandomValues(buf);
      const val = BigInt(buf[0]);
      if (val < limit) {
        return (val % (p - 1n)) + 1n; // Uniform in [1, p-1]
      }
    }
  }
  return BigInt(Math.floor(Math.random() * (Number(p) - 2))) + 1n;
}

/**
 * Computes Shamir t-out-of-N secret shares for a vector of secrets.
 * Uses native C++ core if registered; falls back to pure TypeScript implementation.
 */
export async function computeLightSecAggShares(
  secret: number[],
  numShares: number,
  threshold: number,
  scale: number = Number(DEFAULT_SCALE),
): Promise<number[][]> {
  if (threshold < 1 || threshold > numShares) {
    throw new Error(`Invalid threshold ${threshold} for ${numShares} shares`);
  }
  if (secret.length === 0) {
    return Array.from({ length: numShares }, () => []);
  }

  // Try native core if available
  if (isNativeCoreAvailable() && typeof nativeCore.computeLightSecAggShares === 'function') {
    try {
      const shares = await nativeCore.computeLightSecAggShares(secret, numShares, threshold, scale);
      return shares as number[][];
    } catch {
      // Fall through to TypeScript reference on bridge error
    }
  }

  // Pure TS Reference Implementation
  const p = LIGHTSECAGG_PRIME;
  const bigScale = BigInt(scale);
  const len = secret.length;
  const quantized = secret.map(s => quantize(s, bigScale, p));

  // Generate random polynomial coefficients for degree (threshold - 1)
  const shares: bigint[][] = Array.from({ length: numShares }, () => new Array(len));

  for (let d = 0; d < len; d++) {
    const coeffs: bigint[] = [quantized[d]];
    for (let deg = 1; deg < threshold; deg++) {
      coeffs.push(getSecureRandomFieldElement(p));
    }

    // Evaluate f(x) for x in 1..numShares using Horner's method
    for (let i = 1; i <= numShares; i++) {
      const x = BigInt(i);
      let y = 0n;
      for (let k = coeffs.length - 1; k >= 0; k--) {
        y = mod(y * x + coeffs[k], p);
      }
      shares[i - 1][d] = y;
    }
  }

  return shares.map(s => s.map(v => Number(v)));
}

/**
 * Aggregates shares across clients coordinate-wise modulo p.
 */
export async function aggregateLightSecAggShares(
  shares: number[][],
): Promise<number[]> {
  if (shares.length === 0 || shares[0].length === 0) {
    return [];
  }

  if (isNativeCoreAvailable() && typeof nativeCore.aggregateLightSecAggShares === 'function') {
    try {
      const agg = await nativeCore.aggregateLightSecAggShares(shares);
      return agg as number[];
    } catch {
      // Fall through
    }
  }

  const p = LIGHTSECAGG_PRIME;
  const len = shares[0].length;
  const out = new Array<number>(len).fill(0);

  for (let d = 0; d < len; d++) {
    let sum = 0n;
    for (let i = 0; i < shares.length; i++) {
      sum = mod(sum + BigInt(shares[i][d]), p);
    }
    out[d] = Number(sum);
  }

  return out;
}

/**
 * Reconstructs the secret vector at x=0 using Lagrange basis polynomials.
 */
export function reconstructSecret(
  xs: number[],
  shares: number[][],
  threshold: number,
  scale: number = Number(DEFAULT_SCALE),
): number[] {
  if (xs.length < threshold || shares.length < threshold) {
    throw new Error(`Insufficient shares (${xs.length}) for threshold ${threshold}`);
  }
  const p = LIGHTSECAGG_PRIME;
  const bigScale = BigInt(scale);
  const len = shares[0].length;
  const result: number[] = new Array(len);

  // Compute Lagrange basis coefficients lambda_i(0) = prod_{j != i} (0 - x_j)/(x_i - x_j)
  const lambdas: bigint[] = [];
  for (let i = 0; i < threshold; i++) {
    let num = 1n;
    let den = 1n;
    for (let j = 0; j < threshold; j++) {
      if (i === j) continue;
      const xi = BigInt(xs[i]);
      const xj = BigInt(xs[j]);
      num = mod(num * (-xj), p);
      den = mod(den * (xi - xj), p);
    }
    const lambda = mod(num * modInverse(den, p), p);
    lambdas.push(lambda);
  }

  // Linear combination of shares
  for (let d = 0; d < len; d++) {
    let secretInt = 0n;
    for (let i = 0; i < threshold; i++) {
      secretInt = mod(secretInt + lambdas[i] * BigInt(shares[i][d]), p);
    }
    result[d] = dequantize(secretInt, bigScale, p);
  }

  return result;
}
