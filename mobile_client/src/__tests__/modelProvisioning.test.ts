import { api } from '../lib/restClient';
import nativeCore from '../lib/nativeCore';
import { fetchArtifact } from '../lib/artifactService';
import { provisionTrainingBundle, ModelDeliveryUnavailableError, type ContractProgram } from '../lib/modelProvisioning';

jest.mock('../lib/restClient', () => ({ api: { get: jest.fn(), defaults: { baseURL: 'http://127.0.0.1:8082' } } }));
jest.mock('../lib/nativeCore', () => ({ __esModule: true, default: { stageBundleFile: jest.fn() } }));
jest.mock('../lib/artifactService', () => ({ fetchArtifact: jest.fn() }));

const mApi = api as unknown as { get: jest.Mock };
const mCore = nativeCore as unknown as { stageBundleFile: jest.Mock };
const mFetch = fetchArtifact as unknown as jest.Mock;

const LOSS = 'a'.repeat(64);
const INFER = 'b'.repeat(64);
const TRAIN = 'c'.repeat(64);

// The model programs the run's execution contract lists (its portable CPU variant).
const PROGRAMS: ContractProgram[] = [
  { relativePath: 'loss.pte', sha256: LOSS, byteSize: 1000 },
  { relativePath: 'infer.pte', sha256: INFER, byteSize: 2000 },
];

const DTO = {
  runId: 'r1',
  paramLayout: [{ name: 'fc1.weight', shape: [5, 4] }, { name: 'fc1.bias', shape: [5] }],
  totalParamCount: 43,
  lossPteUrl: '/api/runs/r1/files/loss.pte', lossSha256: 'legacy-loss',
  inferPteUrl: '/api/runs/r1/files/infer.pte', inferSha256: 'legacy-infer',
  inputsUrl: '/api/runs/r1/files/inputs.f32', inputsSha256: 'insha', inputShape: [8, 4],
  targetsUrl: '/api/runs/r1/files/targets.i64', targetsSha256: 'tgtsha',
  trainableParamNames: ['base.fc1.weight', 'base.fc1.bias'],
  classNames: ['c0', 'c1', 'c2'],
};

function serve(dto: object = DTO) {
  mApi.get.mockImplementation((url: string) => {
    if (url === '/api/runs/r1/model-bundle') return Promise.resolve({ data: dto });
    if (url.startsWith('/api/runs/r1/files/')) return Promise.resolve({ data: new Uint8Array([1, 2, 3, 4]).buffer });
    return Promise.reject(new Error('unexpected GET ' + url));
  });
  mCore.stageBundleFile.mockImplementation((name: string) => Promise.resolve('/data/files/bundle/' + name));
  mFetch.mockImplementation((_url: string, sha: string) => Promise.resolve('/data/files/artifacts/' + sha));
}

describe('provisionTrainingBundle', () => {
  beforeEach(() => jest.clearAllMocks());

  test('downloads each program the contract lists by its hash and size, streamed to disk natively', async () => {
    serve();

    const b = await provisionTrainingBundle('r1', PROGRAMS);

    expect(mFetch).toHaveBeenCalledWith(`http://127.0.0.1:8082/api/runs/r1/artifacts/${LOSS}`, LOSS, 1000);
    expect(mFetch).toHaveBeenCalledWith(`http://127.0.0.1:8082/api/runs/r1/artifacts/${INFER}`, INFER, 2000);
    expect(b.lossPtePath).toBe(`/data/files/artifacts/${LOSS}`);
    expect(b.lossSha256).toBe(LOSS);
    expect(b.manifest.inferPtePath).toBe(`/data/files/artifacts/${INFER}`);
    expect(b.manifest.inferSha256).toBe(INFER);
    expect(b.manifest.paramLayout).toEqual(DTO.paramLayout);
    expect(b.manifest.totalParamCount).toBe(43);
    // No program goes through the base64 bridge any more.
    const staged = mCore.stageBundleFile.mock.calls.map((c: unknown[]) => c[0]);
    expect(staged).not.toContain('loss.pte');
    expect(staged).not.toContain('infer.pte');
  });

  test('carries the run\'s class names, in label order, for importing the device\'s own data', async () => {
    serve();

    const b = await provisionTrainingBundle('r1', PROGRAMS);

    expect(b.classNames).toEqual(['c0', 'c1', 'c2']);
  });

  test('stages the on-device data files as before, each verified against its declared hash', async () => {
    serve();

    const b = await provisionTrainingBundle('r1', PROGRAMS);

    expect(mCore.stageBundleFile).toHaveBeenCalledWith('inputs.f32', 'AQIDBA==', 'insha');
    expect(mCore.stageBundleFile).toHaveBeenCalledWith('targets.i64', 'AQIDBA==', 'tgtsha');
    expect(b.inputsF32Path).toBe('/data/files/bundle/inputs.f32');
    expect(b.inputShape).toEqual([8, 4]);
  });

  test('downloads the trainable program when the contract lists one, with the bundle\'s parameter names', async () => {
    serve();

    const b = await provisionTrainingBundle('r1', [...PROGRAMS,
      { relativePath: 'trainable.pte', sha256: TRAIN, byteSize: 3000 }]);

    expect(mFetch).toHaveBeenCalledWith(`http://127.0.0.1:8082/api/runs/r1/artifacts/${TRAIN}`, TRAIN, 3000);
    expect(b.manifest.trainablePtePath).toBe(`/data/files/artifacts/${TRAIN}`);
    expect(b.manifest.trainableSha256).toBe(TRAIN);
    expect(b.manifest.trainableParamNames).toEqual(['base.fc1.weight', 'base.fc1.bias']);
  });

  test('leaves the manifest without a trainable program when the contract lists none', async () => {
    serve();

    const b = await provisionTrainingBundle('r1', PROGRAMS);

    expect(b.manifest.trainablePtePath).toBeUndefined();
  });

  test.each(['loss.pte', 'infer.pte'])('refuses when the contract lists no %s', async (missing) => {
    serve();

    const p = provisionTrainingBundle('r1', PROGRAMS.filter(f => f.relativePath !== missing));

    await expect(p).rejects.toBeInstanceOf(ModelDeliveryUnavailableError);
    await expect(p).rejects.toThrow(missing);
  });

  test('surfaces a failed native download naming the program and its reason', async () => {
    serve();
    mFetch.mockImplementation((_url: string, sha: string) =>
      sha === INFER ? Promise.reject(new Error('ARTIFACT_HASH_MISMATCH')) : Promise.resolve('/p/' + sha));

    const p = provisionTrainingBundle('r1', PROGRAMS);

    await expect(p).rejects.toBeInstanceOf(ModelDeliveryUnavailableError);
    await expect(p).rejects.toThrow(/infer\.pte.*ARTIFACT_HASH_MISMATCH/);
  });

  test('throws ModelDeliveryUnavailableError when no bundle is staged (404)', async () => {
    mApi.get.mockRejectedValue({ response: { status: 404 } });
    await expect(provisionTrainingBundle('r1', PROGRAMS)).rejects.toBeInstanceOf(ModelDeliveryUnavailableError);
  });

  test.each(['inputsSha256', 'targetsSha256'] as const)(
    'refuses to stage data when the bundle omits %s', async (missing) => {
      serve({ ...DTO, [missing]: '' });
      await expect(provisionTrainingBundle('r1', PROGRAMS)).rejects.toBeInstanceOf(ModelDeliveryUnavailableError);
    });
});
