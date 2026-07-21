// Unit tests for the TrainSection state-flow logic (pure helpers).

import {
  derivePhase,
  deriveReadiness,
  describeDetection,
  formatElapsed,
  isReadyToStart,
  type ReadinessInput,
  type TrainRunStatus,
} from '../renderer/components/trainFlow';

describe('derivePhase', () => {
  it.each<[TrainRunStatus, string]>([
    ['idle', 'setup'],
    ['stopped', 'setup'],
    ['pulling', 'running'],
    ['running', 'running'],
    ['restarting', 'running'],
    ['paused', 'running'],
    ['completed', 'completed'],
    ['error', 'error'],
  ])('maps %s → %s', (status, phase) => {
    expect(derivePhase(status)).toBe(phase);
  });
});

describe('describeDetection', () => {
  it('names Apple Silicon for darwin/arm64', () => {
    expect(
      describeDetection({
        platform: 'darwin',
        arch: 'arm64',
        recommendedProfile: 'mps',
        nativeBundleAvailable: true,
        cudaAvailable: false,
      }),
    ).toBe('Apple Silicon');
  });

  it('appends CUDA info on Windows', () => {
    expect(
      describeDetection({
        platform: 'win32',
        arch: 'x64',
        recommendedProfile: 'discrete',
        nativeBundleAvailable: true,
        cudaAvailable: true,
        cudaInfo: 'RTX 4090',
      }),
    ).toBe('Windows x64 · CUDA — RTX 4090');
  });

  it('falls back to platform/arch and flags a missing native bundle', () => {
    expect(
      describeDetection({
        platform: 'linux',
        arch: 'x64',
        recommendedProfile: 'cpu',
        nativeBundleAvailable: false,
        cudaAvailable: false,
      }),
    ).toBe('linux/x64 · native client bundle missing — reinstall to enable training');
  });
});

// A fully-ready baseline the individual cases perturb.
function readyInput(overrides: Partial<ReadinessInput> = {}): ReadinessInput {
  return {
    projectsLoading: false,
    projectsError: '',
    hasProjects: true,
    selectedProject: { name: 'Pneumonia CNN', status: 'RUNNING' },
    eligibility: { eligible: true, lines: [] },
    detection: { done: true, failed: false, nativeBundleMissing: false, summary: 'Apple Silicon' },
    datasetPath: '',
    datasetSkipped: true,
    ...overrides,
  };
}

describe('deriveReadiness + isReadyToStart', () => {
  it('is ready when server ok, project running, hardware detected, dataset skipped', () => {
    const items = deriveReadiness(readyInput());
    expect(items.map((i) => [i.id, i.state])).toEqual([
      ['server', 'ok'],
      ['project', 'ok'],
      ['hardware', 'ok'],
      ['dataset', 'ok'],
    ]);
    expect(isReadyToStart(items)).toBe(true);
  });

  it('is fully pending (not ready) before any fetch resolves', () => {
    const items = deriveReadiness(
      readyInput({
        projectsLoading: true,
        selectedProject: null,
        eligibility: null,
        detection: { done: false, failed: false, nativeBundleMissing: false, summary: '' },
        datasetSkipped: false,
      }),
    );
    expect(items.every((i) => i.state === 'pending')).toBe(true);
    expect(isReadyToStart(items)).toBe(false);
  });

  it('blocks on a server error and leaves the project row pending', () => {
    const items = deriveReadiness(readyInput({ projectsError: 'connect ECONNREFUSED' }));
    expect(items[0].state).toBe('blocked');
    expect(items[0].detail).toContain('ECONNREFUSED');
    expect(items[1].state).toBe('pending');
    expect(isReadyToStart(items)).toBe(false);
  });

  it('blocks when the selected project is not accepting clients', () => {
    const items = deriveReadiness(
      readyInput({ selectedProject: { name: 'ECG', status: 'CREATED' } }),
    );
    expect(items[1].state).toBe('blocked');
    expect(items[1].detail).toContain('not accepting clients');
    expect(isReadyToStart(items)).toBe(false);
  });

  it('blocks when no projects exist at all', () => {
    const items = deriveReadiness(readyInput({ hasProjects: false, selectedProject: null }));
    expect(items[1].state).toBe('blocked');
    expect(isReadyToStart(items)).toBe(false);
  });

  it('keeps eligibility advisory: soft warnings warn but never gate start', () => {
    const items = deriveReadiness(
      readyInput({ eligibility: { eligible: true, lines: ['No NPU detected — training may be slow'] } }),
    );
    expect(items[1].state).toBe('warn');
    expect(isReadyToStart(items)).toBe(true);
  });

  it('keeps hardware advisory: missing native bundle warns but never gates start', () => {
    const items = deriveReadiness(
      readyInput({ detection: { done: true, failed: false, nativeBundleMissing: true, summary: 'x' } }),
    );
    expect(items[2].state).toBe('warn');
    expect(isReadyToStart(items)).toBe(true);
  });

  it('warns (without gating) when detection itself failed', () => {
    const items = deriveReadiness(
      readyInput({ detection: { done: true, failed: true, nativeBundleMissing: false, summary: '' } }),
    );
    expect(items[2].state).toBe('warn');
    expect(isReadyToStart(items)).toBe(true);
  });

  it('requires the dataset to be chosen or explicitly skipped', () => {
    const neither = deriveReadiness(readyInput({ datasetPath: '', datasetSkipped: false }));
    expect(neither[3].state).toBe('pending');
    expect(isReadyToStart(neither)).toBe(false);

    const chosen = deriveReadiness(readyInput({ datasetPath: '/data/xray', datasetSkipped: false }));
    expect(chosen[3].state).toBe('ok');
    expect(chosen[3].detail).toBe('/data/xray');
    expect(isReadyToStart(chosen)).toBe(true);
  });
});

describe('formatElapsed', () => {
  it.each<[number, string]>([
    [0, '0:00'],
    [-500, '0:00'],
    [7_000, '0:07'],
    [65_000, '1:05'],
    [3_599_000, '59:59'],
    [3_723_000, '1:02:03'],
  ])('formats %d ms as %s', (ms, expected) => {
    expect(formatElapsed(ms)).toBe(expected);
  });
});
