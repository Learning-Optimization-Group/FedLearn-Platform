// FL-boundary auth activation: the desktop must hand the backend-minted connection
// token to the spawned client so it can authenticate to the (now fail-closed) FL
// server. The framework client reads FEDLEARN_CONNECTION_TOKEN straight from its
// process environment, so the token travels as an env var in BOTH launch paths —
// container env (docker) and spawn env (native). These pure builders pin that.

import { buildContainerEnv, withConnectionTokenEnv, TrainingConfig } from '../main/docker.service';

const baseConfig: TrainingConfig = {
  hardwareProfile: 'cpu',
  projectId: 'proj-1',
  serverAddress: 'host:50001',
  partitionId: '2',
  modelType: 'CNN',
  datasetPath: '/data',
};

describe('buildContainerEnv (docker path)', () => {
  test('injects FEDLEARN_CONNECTION_TOKEN when a token is present', () => {
    const env = buildContainerEnv({ ...baseConfig, connectionToken: 'tok-abc.def.ghi' });
    expect(env).toContain('FEDLEARN_CONNECTION_TOKEN=tok-abc.def.ghi');
    // Existing contract still carried.
    expect(env).toContain('SERVER_ADDRESS=host:50001');
    expect(env).toContain('PARTITION_ID=2');
  });

  test('omits the token var entirely when no token is set (legacy gate-off flow)', () => {
    const env = buildContainerEnv(baseConfig);
    expect(env.some((e) => e.startsWith('FEDLEARN_CONNECTION_TOKEN'))).toBe(false);
  });
});

describe('withConnectionTokenEnv (native path)', () => {
  test('adds the token to the spawn env without dropping the base env', () => {
    const base = { PATH: '/usr/bin', PYTHONUNBUFFERED: '1' };
    const env = withConnectionTokenEnv(base, { ...baseConfig, connectionToken: 'tok-xyz' });
    expect(env.FEDLEARN_CONNECTION_TOKEN).toBe('tok-xyz');
    expect(env.PATH).toBe('/usr/bin');
    expect(env.PYTHONUNBUFFERED).toBe('1');
  });

  test('returns the base env unchanged when no token is set', () => {
    const base = { PATH: '/usr/bin' };
    const env = withConnectionTokenEnv(base, baseConfig);
    expect(env.FEDLEARN_CONNECTION_TOKEN).toBeUndefined();
    expect(env.PATH).toBe('/usr/bin');
  });
});
