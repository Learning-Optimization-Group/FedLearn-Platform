// On-device model + training-data provisioning for a run. Fetches the run's bundle from the backend
// (GET /api/runs/{runId}/model-bundle), downloads each binary (the ExecuTorch loss/infer .pte graphs +
// the on-device data partition), and stages them into app-private storage via the native TurboModule
// (nativeCore.stageBundleFile). Returns the local paths training.ts feeds to loadModel /
// setTrainingDataFromFiles. Raw features/labels are read from these local files by the native core and
// never leave the device.
import { api } from './restClient';
import nativeCore, { type ModelManifest, type ParamSpec } from './nativeCore';

export interface ModelBundle {
  manifest: ModelManifest; // paramLayout + totalParamCount + inferPtePath/inferSha256
  lossPtePath: string; // forward(flat,x,y) -> loss graph (weights-free .pte)
  lossSha256: string;
  inputsF32Path: string; // row-major float32, shape = inputShape
  inputShape: number[];
  targetsI64Path: string; // int64 labels
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
  inputsUrl: string;
  inputsSha256: string;
  inputShape: number[];
  targetsUrl: string;
  targetsSha256: string;
}

/** Thrown when the model/data bundle can't be fetched/staged (distinguished so the UI can show a precise
 *  message rather than a generic failure). */
export class ModelDeliveryUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelDeliveryUnavailableError';
  }
}

/** Download one bundle binary and stage it into app-private storage; returns its local path. */
async function fetchAndStage(url: string, filename: string): Promise<string> {
  const res = await api.get(url, { responseType: 'arraybuffer' });
  const base64 = arrayBufferToBase64(res.data as ArrayBuffer);
  return nativeCore.stageBundleFile(filename, base64);
}

/**
 * Fetch + stage the model bundle and on-device training partition for a run. The native core sha256-
 * verifies loss.pte (loadModel) and infer.pte, so a corrupted download is rejected on load.
 */
export async function provisionTrainingBundle(runId: string): Promise<ModelBundle> {
  let dto: ModelBundleDto;
  try {
    const res = await api.get<ModelBundleDto>(`/api/runs/${runId}/model-bundle`);
    dto = res.data;
  } catch (e: unknown) {
    const status = (e as { response?: { status?: number } })?.response?.status;
    if (status === 404) {
      throw new ModelDeliveryUnavailableError('No model bundle is staged for this run yet.');
    }
    throw new ModelDeliveryUnavailableError(`Could not fetch the model bundle: ${readError(e)}`);
  }

  const [lossPtePath, inferPtePath, inputsF32Path, targetsI64Path] = await Promise.all([
    fetchAndStage(dto.lossPteUrl, 'loss.pte'),
    fetchAndStage(dto.inferPteUrl, 'infer.pte'),
    fetchAndStage(dto.inputsUrl, 'inputs.f32'),
    fetchAndStage(dto.targetsUrl, 'targets.i64'),
  ]);

  const manifest: ModelManifest = {
    paramLayout: dto.paramLayout,
    totalParamCount: dto.totalParamCount,
    inferPtePath, // rewritten to the staged local path
    inferSha256: dto.inferSha256,
  };
  return {
    manifest,
    lossPtePath,
    lossSha256: dto.lossSha256,
    inputsF32Path,
    inputShape: dto.inputShape,
    targetsI64Path,
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

function readError(e: unknown): string {
  const err = e as { response?: { data?: { message?: string } }; message?: string };
  return err?.response?.data?.message ?? err?.message ?? String(e);
}
