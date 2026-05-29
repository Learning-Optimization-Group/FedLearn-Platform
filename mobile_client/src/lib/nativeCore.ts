// Typed wrapper over the NativeFedLearnCore TurboModule (bridge/specs/NativeFedLearnCore.ts).
// All app code talks to the native C++ FL core through this module — never the raw TurboModule.
import NativeFedLearnCore from '../../bridge/specs/NativeFedLearnCore';

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
} from '../../bridge/specs/NativeFedLearnCore';

export const nativeCore = NativeFedLearnCore;
export default nativeCore;
