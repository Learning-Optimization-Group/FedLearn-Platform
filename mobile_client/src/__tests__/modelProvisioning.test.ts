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

    // Files were downloaded as arraybuffer and staged by name.
    expect(mApi.get).toHaveBeenCalledWith('/api/runs/r1/files/loss.pte', { responseType: 'arraybuffer' });
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('loss.pte', 'AQIDBA==');
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('targets.i64', 'AQIDBA==');
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
});
