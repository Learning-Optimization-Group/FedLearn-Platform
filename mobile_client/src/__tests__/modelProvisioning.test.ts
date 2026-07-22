import { api } from '../lib/restClient';
import nativeCore from '../lib/nativeCore';
import { provisionTrainingBundle, ModelDeliveryUnavailableError } from '../lib/modelProvisioning';

jest.mock('../lib/restClient', () => ({ api: { get: jest.fn() } }));
jest.mock('../lib/nativeCore', () => ({ __esModule: true, default: { stageBundleFile: jest.fn() } }));

const mApi = api as unknown as { get: jest.Mock };
const mCore = nativeCore as unknown as { stageBundleFile: jest.Mock };

const DTO = {
  runId: 'r1',
  paramLayout: [{ name: 'fc1.weight', shape: [5, 4] }, { name: 'fc1.bias', shape: [5] }],
  totalParamCount: 43,
  lossPteUrl: '/api/runs/r1/files/loss.pte', lossSha256: 'losssha',
  inferPteUrl: '/api/runs/r1/files/infer.pte', inferSha256: 'infersha',
  inputsUrl: '/api/runs/r1/files/inputs.f32', inputsSha256: 'insha', inputShape: [8, 4],
  targetsUrl: '/api/runs/r1/files/targets.i64', targetsSha256: 'tgtsha',
};

describe('provisionTrainingBundle (P4 fetch + stage)', () => {
  beforeEach(() => jest.clearAllMocks());

  test('fetches the bundle, base64-stages each file, and maps to a ModelBundle with staged paths', async () => {
    mApi.get.mockImplementation((url: string) => {
      if (url === '/api/runs/r1/model-bundle') return Promise.resolve({ data: DTO });
      if (url.startsWith('/api/runs/r1/files/')) {
        return Promise.resolve({ data: new Uint8Array([1, 2, 3, 4]).buffer }); // -> base64 "AQIDBA=="
      }
      return Promise.reject(new Error('unexpected GET ' + url));
    });
    mCore.stageBundleFile.mockImplementation((name: string) =>
      Promise.resolve('/data/app/files/bundle/' + name));

    const b = await provisionTrainingBundle('r1');

    // Files were downloaded as arraybuffer and staged by name, each with ITS OWN declared sha256
    // so the native layer can verify the decoded bytes before writing (MO-7).
    expect(mApi.get).toHaveBeenCalledWith('/api/runs/r1/files/loss.pte', { responseType: 'arraybuffer' });
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('loss.pte', 'AQIDBA==', 'losssha');
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('infer.pte', 'AQIDBA==', 'infersha');
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('inputs.f32', 'AQIDBA==', 'insha');
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('targets.i64', 'AQIDBA==', 'tgtsha');
    expect(mCore.stageBundleFile).toHaveBeenCalledTimes(4);

    // Bundle carries the staged local paths + the DTO's shas/shape/layout.
    expect(b.lossPtePath).toBe('/data/app/files/bundle/loss.pte');
    expect(b.lossSha256).toBe('losssha');
    expect(b.inputsF32Path).toBe('/data/app/files/bundle/inputs.f32');
    expect(b.targetsI64Path).toBe('/data/app/files/bundle/targets.i64');
    expect(b.inputShape).toEqual([8, 4]);
    expect(b.manifest.inferPtePath).toBe('/data/app/files/bundle/infer.pte'); // rewritten to staged path
    expect(b.manifest.inferSha256).toBe('infersha');
    expect(b.manifest.totalParamCount).toBe(43);
    expect(b.manifest.paramLayout).toEqual(DTO.paramLayout);
  });

  test('throws ModelDeliveryUnavailableError with a clear message when no bundle is staged (404)', async () => {
    mApi.get.mockRejectedValue({ response: { status: 404 } });
    await expect(provisionTrainingBundle('r1')).rejects.toBeInstanceOf(ModelDeliveryUnavailableError);
  });

  test('surfaces a native sha256-mismatch rejection as ModelDeliveryUnavailableError naming the file', async () => {
    mApi.get.mockImplementation((url: string) => {
      if (url === '/api/runs/r1/model-bundle') return Promise.resolve({ data: DTO });
      return Promise.resolve({ data: new Uint8Array([1, 2, 3, 4]).buffer });
    });
    mCore.stageBundleFile.mockImplementation((name: string) =>
      name === 'inputs.f32'
        ? Promise.reject(new Error('stageBundleFile: sha256 mismatch for inputs.f32 (expected insha, got deadbeef)'))
        : Promise.resolve('/data/app/files/bundle/' + name));

    const p = provisionTrainingBundle('r1');
    await expect(p).rejects.toBeInstanceOf(ModelDeliveryUnavailableError);
    await expect(p).rejects.toThrow(/inputs\.f32.*sha256 mismatch/);
  });

  test('stages trainable.pte and populates the manifest when the run advertises first-order', async () => {
    const trainableDto = {
      ...DTO,
      trainablePteUrl: '/api/runs/r1/files/trainable.pte',
      trainableSha256: 'trainsha',
      trainableParamNames: ['base.fc1.weight', 'base.fc1.bias'],
    };
    mApi.get.mockImplementation((url: string) => {
      if (url === '/api/runs/r1/model-bundle') return Promise.resolve({ data: trainableDto });
      return Promise.resolve({ data: new Uint8Array([1, 2, 3, 4]).buffer });
    });
    mCore.stageBundleFile.mockImplementation((name: string) =>
      Promise.resolve('/data/app/files/bundle/' + name));

    const b = await provisionTrainingBundle('r1');

    // The trainable graph is fetched + sha256-verified + staged like every other bundle file...
    expect(mApi.get).toHaveBeenCalledWith('/api/runs/r1/files/trainable.pte', { responseType: 'arraybuffer' });
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('trainable.pte', 'AQIDBA==', 'trainsha');
    // ...and its LOCAL staged path + canonical names land on the manifest the native core loads.
    expect(b.manifest.trainablePtePath).toBe('/data/app/files/bundle/trainable.pte');
    expect(b.manifest.trainableSha256).toBe('trainsha');
    expect(b.manifest.trainableParamNames).toEqual(['base.fc1.weight', 'base.fc1.bias']);
  });

  test('leaves the manifest DeComFL-only (no trainablePtePath) when no trainable bundle is advertised', async () => {
    mApi.get.mockImplementation((url: string) => {
      if (url === '/api/runs/r1/model-bundle') return Promise.resolve({ data: DTO });
      return Promise.resolve({ data: new Uint8Array([1, 2, 3, 4]).buffer });
    });
    mCore.stageBundleFile.mockImplementation((name: string) =>
      Promise.resolve('/data/app/files/bundle/' + name));

    const b = await provisionTrainingBundle('r1');

    expect(b.manifest.trainablePtePath).toBeUndefined();
    const staged = mCore.stageBundleFile.mock.calls.map((c: unknown[]) => c[0]);
    expect(staged).not.toContain('trainable.pte'); // nothing trainable staged for a DeComFL-only run
  });

  test.each(['lossSha256', 'inferSha256', 'inputsSha256', 'targetsSha256'] as const)(
    'refuses to stage when the bundle omits %s (no unverified file reaches the native layer)',
    async (missing) => {
      const dto = { ...DTO, [missing]: '' };
      mApi.get.mockImplementation((url: string) => {
        if (url === '/api/runs/r1/model-bundle') return Promise.resolve({ data: dto });
        return Promise.resolve({ data: new Uint8Array([1, 2, 3, 4]).buffer });
      });
      mCore.stageBundleFile.mockImplementation((name: string) =>
        Promise.resolve('/data/app/files/bundle/' + name));

      await expect(provisionTrainingBundle('r1')).rejects.toBeInstanceOf(ModelDeliveryUnavailableError);
      // The file whose hash is missing must never have been handed to the native stager.
      const fileForSha: Record<typeof missing, string> = {
        lossSha256: 'loss.pte', inferSha256: 'infer.pte',
        inputsSha256: 'inputs.f32', targetsSha256: 'targets.i64',
      };
      const staged = mCore.stageBundleFile.mock.calls.map((c: unknown[]) => c[0]);
      expect(staged).not.toContain(fileForSha[missing]);
    },
  );
});
