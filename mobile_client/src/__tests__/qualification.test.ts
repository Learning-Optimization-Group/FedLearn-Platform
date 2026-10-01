// Stage 3 D2: before a device first trains with a trainable program it runs the exporter's probe on it, once per
// (device, app build, bridge, program). A pass is cached; a failure refuses and quarantines that key until the app
// or the program changes, so a device never retries a program it cannot run.
jest.mock('react-native-device-info', () => ({
  getModel: () => 'vivo 1805', getSystemVersion: () => '9', getVersion: () => '2.1.0', getBuildNumber: () => '7',
}));

import { qualificationKey, qualifyTrainableProgram, QualificationStore } from '../lib/qualification';
import type { QualificationReport, TrainableProbe } from '../lib/nativeCore';

const PROBE: TrainableProbe = {
  rows: 8, width: 4, classes: 3, learningRate: 0.1, lossStep1: 1.1054, lossStep2: 1.0776, lossTolerance: 1e-4,
  maxProbeMs: 2000,
};
const SHA = 'c'.repeat(64);
const PASS: QualificationReport = { passed: true, failedCheck: '', detail: '', lossStep1: 1.1054, lossStep2: 1.0776,
  wallMs: 3 };
const FAIL: QualificationReport = { ...PASS, passed: false, failedCheck: 'LOSS_MISMATCH', detail: 'step 1 loss 2.0' };

function memory() {
  const data = new Map<string, string>();
  return { getItem: async (k: string) => data.get(k) ?? null, setItem: async (k: string, v: string) => { data.set(k, v); } };
}

function setup(report: QualificationReport = PASS) {
  const store = new QualificationStore(memory());
  const native = { qualifyTrainable: jest.fn().mockResolvedValue(report) };
  return { store, native, run: () => qualifyTrainableProgram(SHA, PROBE, { store, native }) };
}

test('runs the probe on a program this device has not qualified, and remembers a pass', async () => {
  const { native, run } = setup();
  await expect(run()).resolves.toMatchObject({ passed: true });
  await expect(run()).resolves.toMatchObject({ passed: true });
  expect(native.qualifyTrainable).toHaveBeenCalledTimes(1);
  expect(native.qualifyTrainable).toHaveBeenCalledWith(PROBE);
});

test('a failure is remembered too: the device does not rerun a program it cannot run', async () => {
  const { native, run } = setup(FAIL);
  await expect(run()).resolves.toMatchObject({ passed: false, failedCheck: 'LOSS_MISMATCH' });
  await expect(run()).resolves.toMatchObject({ passed: false, failedCheck: 'LOSS_MISMATCH' });
  expect(native.qualifyTrainable).toHaveBeenCalledTimes(1);
});

test('another program, or another app build, is qualified afresh', () => {
  const key = qualificationKey(SHA);
  expect(qualificationKey('d'.repeat(64))).not.toBe(key);
  expect(key).toContain('vivo 1805');
  expect(key).toContain('2.1.0+7');
  expect(key).toContain(SHA);
});

test('a program with no probe does not qualify', async () => {
  const store = new QualificationStore(memory());
  const native = { qualifyTrainable: jest.fn() };
  await expect(qualifyTrainableProgram(SHA, undefined, { store, native }))
    .resolves.toMatchObject({ passed: false, failedCheck: 'NO_PROBE' });
  expect(native.qualifyTrainable).not.toHaveBeenCalled();
});

test('a native error is a failed qualification, not a pass', async () => {
  const store = new QualificationStore(memory());
  const native = { qualifyTrainable: jest.fn().mockRejectedValue(new Error('boom')) };
  await expect(qualifyTrainableProgram(SHA, PROBE, { store, native }))
    .resolves.toMatchObject({ passed: false, failedCheck: 'NATIVE_ERROR', detail: 'boom' });
});

test('a corrupt cache is treated as empty rather than trusted', async () => {
  const storage = memory();
  await storage.setItem('fedlearn.qualification.v1', '{not json');
  const native = { qualifyTrainable: jest.fn().mockResolvedValue(PASS) };
  await expect(qualifyTrainableProgram(SHA, PROBE, { store: new QualificationStore(storage), native }))
    .resolves.toMatchObject({ passed: true });
  expect(native.qualifyTrainable).toHaveBeenCalledTimes(1);
});

test('an image program\'s probe reaches the device with each example\'s shape', async () => {
  // Stage 4 S6: the CNN takes [rows, 3, 32, 32]; the native probe builds that batch from the stated shape.
  const store = new QualificationStore(memory());
  const native = { qualifyTrainable: jest.fn().mockResolvedValue(PASS) };
  const image: TrainableProbe = { ...PROBE, width: 3072, classes: 10, inputShape: [3, 32, 32] };
  await qualifyTrainableProgram('d'.repeat(64), image, { store, native });
  expect(native.qualifyTrainable).toHaveBeenCalledWith(expect.objectContaining({ inputShape: [3, 32, 32] }));
});
