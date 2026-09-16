/**
 * P1-5: the training arm reaches the client, so a FROZEN_HEAD project actually federates head-only.
 *
 * P1-2 taught the SERVER to filter its initial parameters to the arm's trainable subset, and P1-4
 * let a user choose the arm. Nothing carried that choice to the client. `fl-runtime/client.py`
 * accepts `--training-arm`, but neither launch path set it, so the client kept
 * `TRAINING_ARM = "FULL"`, `USE_DERIVED` stayed false, and `get_parameters()` uploaded the FULL
 * state dict against a server expecting the head only.
 *
 * That is the same class of defect the `strategy` field already fixed here — "a non-MLP DeComFL
 * project otherwise runs a FedAvg-path client that silently mismatches the server" — so the arm
 * follows exactly that precedent, through both launch paths:
 *
 *   Docker: TRAINING_ARM env  -> entrypoint.sh -> --training-arm
 *   Native: --training-arm argv
 *
 * Both must carry it, or the arm works in one deployment and silently corrupts the other.
 */
import { buildContainerEnv, type TrainingConfig } from '../main/docker.service';

const BASE: TrainingConfig = {
  projectId: 'p-1',
  serverAddress: 'host:50000',
  partitionId: '0',
  modelType: 'PNEUMONIA_CNN',
} as TrainingConfig;

describe('training arm — Docker path', () => {
  it('passes the arm to the container when the project declares one', () => {
    const env = buildContainerEnv({ ...BASE, trainingArm: 'FROZEN_HEAD' } as TrainingConfig);
    expect(env).toContain('TRAINING_ARM=FROZEN_HEAD');
  });

  it('omits the arm when the connection payload has none', () => {
    // A backend that predates P1 sends no arm; the client must then behave exactly as before
    // rather than receive an empty value it has to interpret.
    const env = buildContainerEnv(BASE);
    expect(env.some((e) => e.startsWith('TRAINING_ARM='))).toBe(false);
  });

  it('passes FULL explicitly when the project is a full fine-tune', () => {
    // Not an omission: the server was told FULL, and the client must be told the same thing rather
    // than inferring it from silence.
    const env = buildContainerEnv({ ...BASE, trainingArm: 'FULL' } as TrainingConfig);
    expect(env).toContain('TRAINING_ARM=FULL');
  });

  it('still carries the other launch config alongside it', () => {
    const env = buildContainerEnv({
      ...BASE, trainingArm: 'FROZEN_HEAD', strategy: 'FedAvg', connectionToken: 'tok',
    } as TrainingConfig);
    expect(env).toEqual(expect.arrayContaining([
      'PROJECT_ID=p-1', 'SERVER_ADDRESS=host:50000', 'PARTITION_ID=0',
      'MODEL_TYPE=PNEUMONIA_CNN', 'STRATEGY=FedAvg',
      'FEDLEARN_CONNECTION_TOKEN=tok', 'TRAINING_ARM=FROZEN_HEAD',
    ]));
  });
});
