// React Native TurboModule spec (codegen source of truth) — 15-LLD-mobile.md §5.1.
// Every method is async (Promise) and returns a TYPED object — never a hand-built JSON string
// (fixes A6 §L3, the v1 stringly-typed bridge). RN codegen consumes THIS file to generate the
// native interface; the C++ impl is bridge/common/FedLearnCoreModule.{h,cpp}.
import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export type Strategy = 'DeComFL' | 'FedAvg';
export type GradEstimateMethod = 'forward' | 'central';

export interface RegisterResult {
  accepted: boolean;
  message: string;
  assignedRound: number; // RegisterClientResponse.assigned_round (late-joiner round)
  serverProtocolVersion: number;
}

export interface ServerStatus {
  serverState: string; // GetServerStatusResponse.ServerState name
  currentRound: number;
  requiredClientsForRound: number;
  receivedUpdatesThisRound: number;
  activeClients: number;
  roundDeadlineUnixMs: number;
}

export interface RoundConfig {
  strategy: Strategy;
  learningRate: number; // eta
  mu: number; // DeComFL ZO smoothing radius (double on the C++ side)
  numPerturbations: number; // P  (1..256, 04 §4.2)
  numLocalSteps: number; // K  (1..1000)
  gradEstimateMethod: GradEstimateMethod; // default 'forward' (B1-H2)
  seed: number; // optimizer seed (distinct from data seed)
  torchVersion: string; // must match server's GetDeComFLConfigResponse.torch_version
}

export interface RoundResult {
  round: number;
  loss: number;
  accuracy: number; // -1.0 if not evaluated this round
  scalarsTransmitted: number; // K*P (DeComFL) ; 0 for FedAvg
  uplinkBytes: number;
  downlinkBytes: number;
  computeMs: number;
  reverted: boolean; // model restored to pre-round snapshot (DeComFL invariant)
}

export interface DeviceMetrics {
  peakRssBytes: number; // native RSS sample (replaces broken resourceMonitor.js, M-H4)
  thermalState: string; // 'NOMINAL'|'FAIR'|'SERIOUS'|'CRITICAL' (platform thermal API)
  batteryLevel: number; // 0.0..1.0
  batteryCharging: boolean;
}

export interface ModelInfo {
  paramCount: number; // model dimension d
  trainableParamCount: number; // requires_grad-filtered count (must match server's P-dim, M-C2)
  sha256: string; // integrity hash of the .pt (verified before jit::load, M-C4)
  tier: '1M' | '10M' | '100M';
}

export interface InferResult {
  logits: number[];
  probabilities: number[];
  argmax: number;
}

// One trainable tensor's flat-order layout (name -> shape). numel = product(shape).
export interface ParamSpec {
  name: string;
  shape: number[]; // int64 on the native side; JS numbers are cast to int64
}

// The model's ExecuTorch "weights-as-inputs" sidecar manifest (written by scripts/pte_export.py):
// the trainable param layout, total param count (incl. frozen, for the tier), and the SEPARATE
// infer graph forward(flat,x)->logits (its own .pte + sha). Must be set before loadModel().
export interface ModelManifest {
  paramLayout: ParamSpec[];
  totalParamCount: number;
  inferPtePath: string;
  inferSha256: string;
  // First-order (FedAvg) path — OPTIONAL. When the backend provisions a TRAINABLE .pte (forward+backward
  // graph), these carry its staged path, sha256, and the trainable param names in canonical flat order.
  // Present => the native FedAvg round uses real backprop + a weight-blob upload; absent => ZO fallback.
  trainablePtePath?: string;
  trainableSha256?: string;
  trainableParamNames?: string[];
}

export interface Spec extends TurboModule {
  // ---- gRPC lifecycle ----
  registerClient(
    serverAddress: string,
    runId: string,
    clientId: string,
    enrollmentToken: string,
    useTls: boolean,
  ): Promise<RegisterResult>;
  getServerStatus(runId: string): Promise<ServerStatus>;
  stop(): Promise<void>; // sets the abort flag; joins threads

  // ---- model ----
  // The ExecuTorch loss graph is weights-free, so the trainable-param layout + the infer graph come
  // from the sidecar manifest; set it before loadModel(). (15-LLD §13 task 14 — model delivery.)
  setModelManifest(manifest: ModelManifest): Promise<void>;
  loadModel(modelPath: string, expectedSha256: string): Promise<ModelInfo>; // integrity-checked
  // On-device training data: float32 inputs (row-major, shape inputShape) + int64 targets, read from
  // app-private files. Raw features/labels live ONLY on the device and never enter any server table.
  setTrainingDataFromFiles(
    inputsF32Path: string,
    inputShape: number[],
    targetsI64Path: string,
  ): Promise<void>;
  // Write a downloaded bundle file (base64) into app-private storage (dataDir/bundle) and return its
  // absolute local path — used by provisionTrainingBundle to stage the .pte graphs + on-device data
  // before loadModel / setTrainingDataFromFiles. filename is basename-sanitised; data stays on-device.
  // expectedSha256 is the backend-declared hash of the file: the native side sha256-verifies the
  // DECODED bytes and rejects on mismatch BEFORE anything is written, so a tampered/corrupted bundle
  // file is never staged (MO-7 — covers inputs.f32/targets.i64, which loadModel never re-checks).
  stageBundleFile(filename: string, base64Data: string, expectedSha256: string): Promise<string>;

  // ---- one round (the bridge runs ONE round per call; the RN layer loops + checks deadline) ----
  runDeComFLRound(runId: string, config: RoundConfig): Promise<RoundResult>;
  runFedAvgRound(runId: string, config: RoundConfig): Promise<RoundResult>;

  // ---- inference (Model Testing screen) — REAL softmax, not exp(-loss) (C5 §3) ----
  infer(inputJson: string): Promise<InferResult>;

  // ---- telemetry ----
  getDeviceMetrics(): Promise<DeviceMetrics>;
}

// The native C++ TurboModule is only registered on builds that actually compiled the FL core (Android
// with LIBTORCH_DIR, or a future real iOS port — MO-14). getEnforcing() throws SYNCHRONOUSLY at module
// load when the module is absent, which crashed the whole JS bundle at launch on the iOS scaffold. get()
// returns null instead, so importing this module is always safe; availability is surfaced via
// isNativeCoreAvailable() so the UI can disable training instead of crashing (MO-5).
const nativeModule: Spec | null = TurboModuleRegistry.get<Spec>('NativeFedLearnCore');

// Single, clear, actionable message rejected by every fallback method.
export const NATIVE_CORE_UNAVAILABLE_MESSAGE = 'native FL core unavailable on this platform';

// True only when the native FL core is registered for this platform/build. The app gates its training
// entry point(s) on this (re-exported via src/lib/nativeCore.ts) rather than calling into the fallback.
export function isNativeCoreAvailable(): boolean {
  return nativeModule != null;
}

// Every fallback method REJECTS (never throws synchronously) so the `Promise<T>` contract holds and
// existing `.catch(...)` callers (e.g. deviceClass.collectDeviceCapabilities) still degrade gracefully.
// The rejection only happens when a method is actually invoked — importing this module never throws.
function unavailable(): Promise<never> {
  return Promise.reject(new Error(NATIVE_CORE_UNAVAILABLE_MESSAGE));
}

// Typed no-op core used when the native module is absent. Keeps the default-export type exactly `Spec`
// so callers and tsc are unaffected; only actual training/gRPC calls fail (loudly, with the message).
const fallbackCore: Spec = {
  registerClient: unavailable,
  getServerStatus: unavailable,
  stop: unavailable,
  setModelManifest: unavailable,
  loadModel: unavailable,
  setTrainingDataFromFiles: unavailable,
  stageBundleFile: unavailable,
  runDeComFLRound: unavailable,
  runFedAvgRound: unavailable,
  infer: unavailable,
  getDeviceMetrics: unavailable,
};

export default nativeModule ?? fallbackCore;
