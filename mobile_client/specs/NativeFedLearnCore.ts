import { TurboModule, TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  // Model lifecycle
  loadModel(modelPath: string): boolean;
  getModelInfo(): { numParams: number; sizeBytes: number };

  // Local training (Phase 6)
  trainStep(
    inputPath: string,
    numEpochs: number,
    lr: number,
  ): { loss: number; accuracy: number };

  // Federated learning (Phase 7)
  connect(serverAddress: string, clientId: string): boolean;
  disconnect(): void;
  startFedAvgLoop(configJson: string): void;
  startDeComFLLoop(configJson: string): void;
  stopTraining(): void;
  getStatus(): string;

  // ZO config (Phase 8)
  setZOConfig(configJson: string): void;

  // Log retrieval
  getRecentLogs(): string;
}

export default TurboModuleRegistry.getEnforcing<Spec>('NativeFedLearnCore');
