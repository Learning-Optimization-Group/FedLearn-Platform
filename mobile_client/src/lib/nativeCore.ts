// Typed wrapper over the NativeFedLearnCore TurboModule (bridge/specs/NativeFedLearnCore.ts).
// All app code talks to the native C++ FL core through this module — never the raw TurboModule.
import NativeFedLearnCore, { isNativeCoreAvailable } from '../../bridge/specs/NativeFedLearnCore';

// Re-exported so screens can gate the training entry point without importing the raw spec.
export { isNativeCoreAvailable };

export type {
  Strategy,
  GradEstimateMethod,
  RegisterResult,
  ServerStatus,
  RoundConfig,
  RoundResult,
  DeviceMetrics,
  ModelInfo,
  InferResult,
  ParamSpec,
  ModelManifest,
} from '../../bridge/specs/NativeFedLearnCore';

export const nativeCore = NativeFedLearnCore;
export default nativeCore;
