// On-device model + training-data provisioning for a run. The model programs (the ExecuTorch .pte graphs) are the
// ones the run's execution contract lists; each is streamed natively to app-private storage and verified against the
// contract's size and SHA-256 (artifactService). The bundle endpoint (GET /api/runs/{runId}/model-bundle) supplies
// the parameter layout and the on-device data files, which are still staged through nativeCore.stageBundleFile
// (sha256-verified before writing, MO-7). Returns the local paths training.ts feeds to loadModel /
// setTrainingDataFromFiles.
import { api } from './restClient';
import nativeCore, { type ModelManifest, type ParamSpec } from './nativeCore';
import { readError } from './errors';
import { fetchArtifact } from './artifactService';

export interface ModelBundle {
  manifest: ModelManifest; // paramLayout + totalParamCount + inferPtePath/inferSha256
  lossPtePath: string; // forward(flat,x,y) -> loss graph (weights-free .pte)
  lossSha256: string;
  // The run's fixture data; absent for a run that trains the device's own dataset snapshot.
  inputsF32Path?: string; // row-major float32, shape = inputShape
  inputShape?: number[];
  targetsI64Path?: string; // int64 labels
  classNames: string[]; // the run's classes in label order; a device's own dataset is imported against them
}

// The backend ModelBundleDto (RunController#modelBundle). File fields are URLs under /api/runs/{id}/files.
interface ModelBundleDto {
  runId: string;
  paramLayout: ParamSpec[];
  totalParamCount: number;
  lossPteUrl: string;
  lossSha256: string;
  inferPteUrl: string;
  inferSha256: string;
  // The run's fixture batch; null (and an empty shape) for a run that trains on each device's own dataset.
  inputsUrl: string | null;
  inputsSha256: string | null;
  inputShape: number[];
  targetsUrl: string | null;
  targetsSha256: string | null;
  // First-order (FedAvg) trainable graph — present only when the backend staged one for this run.
  // Absent => DeComFL-only: the manifest omits trainablePtePath and the native core runs zeroth-order.
  trainablePteUrl?: string | null;
  trainableSha256?: string | null;
  trainableParamNames?: string[] | null;
  classNames?: string[] | null;
}

/** Thrown when the model/data bundle can't be fetched/staged (distinguished so the UI can show a precise
 *  message rather than a generic failure). */
export class ModelDeliveryUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelDeliveryUnavailableError';
  }
}

/**
 * Download one bundle binary and stage it into app-private storage; returns its local path.
 * expectedSha256 is the backend-declared hash: the native layer verifies the decoded bytes against it
 * before writing, so a tampered/corrupted file is rejected at staging time (MO-7). A bundle that
 * doesn't declare a hash for the file is refused outright — nothing unverifiable gets staged.
 */
async function fetchAndStage(url: string | null, filename: string, expectedSha256: string | null): Promise<string> {
  if (!url) {
    throw new ModelDeliveryUnavailableError(`The model bundle offers no ${filename}; this run has no fixture data.`);
  }
  if (!expectedSha256) {
    throw new ModelDeliveryUnavailableError(
      `The model bundle did not declare a sha256 for ${filename}; refusing to stage an unverifiable file.`,
    );
  }
  const res = await api.get(url, { responseType: 'arraybuffer' });
  const base64 = arrayBufferToBase64(res.data as ArrayBuffer);
  try {
    return await nativeCore.stageBundleFile(filename, base64, expectedSha256);
  } catch (e: unknown) {
    // Most commonly a sha256 mismatch (tampered/corrupted download); surface it as the delivery
    // error the UI already knows how to present, keeping the native detail in the message.
    throw new ModelDeliveryUnavailableError(`Could not stage ${filename}: ${readError(e)}`);
  }
}

/**
 * Fetch + stage the model bundle and on-device training partition for a run. Every staged file
 * (loss.pte, infer.pte, inputs.f32, targets.i64) is sha256-verified against the bundle's declared
 * hashes at staging time; loadModel additionally re-verifies the two .pte graphs on load.
 */
/** A model program the run's execution contract lists: its file name, digest and size. */
export interface ContractProgram {
  relativePath: string;
  sha256: string;
  byteSize: number;
}

/**
 * Download one model program the contract lists, streamed natively to app-private storage and verified against the
 * contract's size and digest. Returns its local path.
 */
async function fetchProgram(runId: string, program: ContractProgram): Promise<string> {
  const url = `${api.defaults.baseURL ?? ''}/api/runs/${runId}/artifacts/${program.sha256}`;
  try {
    return await fetchArtifact(url, program.sha256, program.byteSize);
  } catch (e) {
    throw new ModelDeliveryUnavailableError(`Could not download ${program.relativePath}: ${readError(e)}`);
  }
}

/**
 * Provision a run's on-device training bundle. The model programs come from the execution contract, by digest
 * (`programs`, its portable CPU variant): the loss and inference graphs, and the trainable graph when listed. The
 * bundle endpoint still supplies the parameter layout, the trainable parameter names and the on-device data files,
 * which are staged as before until datasets become on-device imports (Stage 3, section B).
 */
export async function provisionTrainingBundle(
  runId: string,
  programs: ContractProgram[],
  options: { fixtureData?: boolean } = {},
): Promise<ModelBundle> {
  const fixtureData = options.fixtureData ?? true;
  const byName = new Map(programs.map(p => [p.relativePath, p]));
  const loss = byName.get('loss.pte');
  const infer = byName.get('infer.pte');
  for (const [name, program] of [['loss.pte', loss], ['infer.pte', infer]] as const) {
    if (!program) {
      throw new ModelDeliveryUnavailableError(`This run's execution contract lists no ${name}.`);
    }
  }
  let dto: ModelBundleDto;
  try {
    const res = await api.get<ModelBundleDto>(`/api/runs/${runId}/model-bundle`);
    dto = res.data;
  } catch (e) {
    throw new ModelDeliveryUnavailableError(`Could not fetch the model bundle: ${readError(e)}`);
  }

  // A run on the device's own dataset takes no training data from the server.
  const [lossPtePath, inferPtePath, inputsF32Path, targetsI64Path] = await Promise.all([
    fetchProgram(runId, loss!),
    fetchProgram(runId, infer!),
    fixtureData ? fetchAndStage(dto.inputsUrl, 'inputs.f32', dto.inputsSha256) : Promise.resolve(undefined),
    fixtureData ? fetchAndStage(dto.targetsUrl, 'targets.i64', dto.targetsSha256) : Promise.resolve(undefined),
  ]);

  const manifest: ModelManifest = {
    paramLayout: dto.paramLayout,
    totalParamCount: dto.totalParamCount,
    inferPtePath,
    inferSha256: infer!.sha256,
  };

  // First-order runs list a trainable graph; its local path and the canonical trainable-parameter names go to the
  // native core through the manifest. Without one the native core runs zeroth-order (DeComFL).
  const trainable = byName.get('trainable.pte');
  if (trainable) {
    manifest.trainablePtePath = await fetchProgram(runId, trainable);
    manifest.trainableSha256 = trainable.sha256;
    manifest.trainableParamNames = dto.trainableParamNames ?? [];
  }

  return {
    manifest,
    lossPtePath,
    lossSha256: loss!.sha256,
    inputsF32Path,
    inputShape: fixtureData ? dto.inputShape : undefined,
    targetsI64Path,
    classNames: dto.classNames ?? [],
  };
}

// ArrayBuffer -> base64 (RN Hermes has no btoa/Buffer). Small + correct; the MVP bundle is tiny (a real
// multi-MB model should stream to a file instead of base64-through-JSI — noted for post-MVP).
const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const at = (j: number) => bytes[j] ?? 0;
  let out = '';
  let i = 0;
  for (; i + 2 < bytes.length; i += 3) {
    const n = (at(i) << 16) | (at(i + 1) << 8) | at(i + 2);
    out += B64[(n >> 18) & 63]! + B64[(n >> 12) & 63]! + B64[(n >> 6) & 63]! + B64[n & 63]!;
  }
  const rem = bytes.length - i;
  if (rem === 1) {
    const n = at(i) << 16;
    out += B64[(n >> 18) & 63]! + B64[(n >> 12) & 63]! + '==';
  } else if (rem === 2) {
    const n = (at(i) << 16) | (at(i + 1) << 8);
    out += B64[(n >> 18) & 63]! + B64[(n >> 12) & 63]! + B64[(n >> 6) & 63]! + '=';
  }
  return out;
}
