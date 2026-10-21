import type { SimulationRun, SimulationRound, SimulationWorkload } from '../services/simulationService';
export const runA = '11111111-1111-4111-8111-111111111111';
export const runB = '22222222-2222-4222-8222-222222222222';
export const recipe = {
  workloadId: 'nih-qualified', head: 'linear', numClients: 10, rounds: 3, localEpochs: 1,
  aggregation: 'mean', clipNorm: .05, privacy: 'none', epsilon: 4, behavior: 'clean', seed: 42,
  evaluationSplit: 'validation', featureReportSha256: 'a'.repeat(64), learningRate: .003,
  batchSize: 256, parameters: 1026, delta: .00001, trimRatio: null, maliciousFraction: 0,
  researchRecordDpSafe: false,
};
export const limits = { deadlineSeconds: 180, cpuThreads: 4, observedRssQualificationBytes: 2147483648,
  activeWorkers: 1, scope: 'Exploratory validation; modeled deployment costs.' };
export const workload: SimulationWorkload = {
  id: 'nih-qualified', ready: true, reason: null, label: 'No Finding (0) versus any finding (1)',
  representation: 'Frozen pretrained ResNet-18 features; validation only', reportSha256: 'a'.repeat(64),
  splitSummary: { train: { images: 68650, patients: 22399 }, validation: { images: 17763, patients: 5600 } },
  supportedChoices: { head: ['linear', 'mlp8', 'mlp128', 'mlp512'], privacy: ['none', 'central'] }, limits,
};
export function round(number: number, auc = .61): SimulationRound {
  return { round: number, auc, accuracy: .6, loss: .7, balanced_accuracy: .58,
    sensitivity: .5, specificity: .66, ece: .1,
    comm: { up_bytes: 12000, down_bytes: 16000 },
    privacy: { epsilon_spent: null, clip_rate: .3, noise_std: 0, sensitivity: .01 },
    participants: { sampled: 10, submitted: 10, admitted: 10, accepted: 10, aggregated: 10, dropped: 0, straggled: 0 },
    cpuSeconds: .2, simulatedRoundSeconds: .5 };
}
export function simulationRun(over: Partial<SimulationRun> = {}): SimulationRun {
  return { id: runA, status: 'COMPLETED', configHash: 'b'.repeat(64), recipe,
    createdAt: '2026-10-20T20:00:00Z', startedAt: '2026-10-20T20:00:01Z', endedAt: '2026-10-20T20:00:04Z',
    completedRounds: 3, requestedRounds: 3, rounds: [round(1,.92), round(2,.7), round(3,.61)],
    finalSummary: round(3,.61), qualification: { scope: 'simulation_execution_only', research_record_dp_safe: false },
    recordAvailable: true, recordEncoding: 'json', errorCode: null, errorMessage: null,
    scope: 'Exploratory validation simulation. Research records are not DP-safe.', ...over };
}
