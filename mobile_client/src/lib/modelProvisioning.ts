// On-device model + training-data provisioning for a run. This is the client-side SEAM the training
// loop depends on: it must stage the ExecuTorch loss/infer .pte graphs, the sidecar manifest, and the
// device's local training partition (float32 inputs + int64 targets) into app-private files, and return
// their paths. Raw features/labels are read from these local files by the native core and never leave
// the device.
//
// Not wired end-to-end yet: it needs (1) the v2 server's model-delivery endpoint and (2) an on-device
// file-staging layer (a filesystem module such as react-native-fs, or the native GetGlobalModelStream
// path) to write the binaries. Until both land this throws a clear, catchable error — the training loop
// (training.ts) is fully wired and runs the moment provisioning returns a bundle.
import type { ModelManifest } from './nativeCore';

export interface ModelBundle {
  manifest: ModelManifest; // paramLayout + totalParamCount + inferPtePath/inferSha256
  lossPtePath: string; // forward(flat,x,y) -> loss graph (weights-free .pte)
  lossSha256: string;
  inputsF32Path: string; // row-major float32, shape = inputShape
  inputShape: number[];
  targetsI64Path: string; // int64 labels
}

/** Thrown when the model/data delivery pipeline isn't available yet (distinguished so the UI can show
 *  a precise message rather than a generic failure). */
export class ModelDeliveryUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelDeliveryUnavailableError';
  }
}

/**
 * Fetch + stage the model bundle and on-device training partition for a run.
 *
 * CONTRACT (to be provided by the v2 server + a file-staging layer):
 *   GET /api/runs/{runId}/model-bundle  ->
 *     { manifest, lossPte:{url,sha256}, inferPte:{url,sha256}, dataset:{inputsUrl, inputShape, targetsUrl} }
 *   then each binary is written to app-private storage and this returns the local paths above.
 */
export async function provisionTrainingBundle(_runId: string): Promise<ModelBundle> {
  throw new ModelDeliveryUnavailableError(
    'On-device model delivery is not available yet. It requires the v2 server model-bundle endpoint ' +
      'and on-device file staging (the mobile client speaks fedlearn.v2; the current FL server is v1). ' +
      'The on-device training loop is wired and will run automatically once a bundle can be staged.',
  );
}
