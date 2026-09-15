import { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { Project, StartServerData, errorMessage } from '../../services/apiServices';
import { createLogger } from '../../lib/logger';
import { Modal, Input, Select, Button, FormField } from '../ui';

const log = createLogger('StartProjectModal');

interface StartProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onClose: () => void;
  onSubmit: (projectId: string, config: StartServerData) => Promise<void>;
}

// Plain-language descriptions for each training method (values stay as-is).
const STRATEGIES: { value: string; label: string }[] = [
  { value: 'FedAvg', label: 'Standard — averages everyone\'s learning (recommended)' },
  { value: 'FedProx', label: 'Stable — keeps clients close to the shared model on uneven data' },
  { value: 'DeComFL', label: 'Low-bandwidth — sends tiny updates' },
  { value: 'FedOpt', label: 'Adaptive — faster convergence on varied data' },
  { value: 'Robust', label: 'Robust — resists a few bad or noisy clients' },
  { value: 'FoT', label: 'For text models' },
];

type RobustParam = 'fraction' | 'trim' | 'tau' | null;

// Byzantine-robust aggregation rules the server implements (value = backend RobustMethod name). The help
// lines state what the FR-12 breakdown sweeps measured, not only the textbook bound: several rules refuse
// to run below a cohort size, and the server rejects those configurations at start.
const ROBUST_RULES: { value: string; label: string; help: string; param: RobustParam }[] = [
  { value: 'MEDIAN', label: 'Median — middle value for each weight', param: null,
    help: 'Tolerates up to half of devices being bad. The simplest choice.' },
  { value: 'TRIMMED_MEAN', label: 'Trimmed mean — drops the extremes', param: 'trim',
    help: 'Tolerates bad devices up to the share it trims from each end.' },
  { value: 'KRUM', label: 'Krum — keeps the single most typical update', param: 'fraction',
    help: 'Needs at least 3 devices, and more as the expected bad share grows. Uses one device’s update per round.' },
  { value: 'MULTI_KRUM', label: 'Multi-Krum — averages the most typical updates', param: 'fraction',
    help: 'Needs at least 3 devices, and more as the expected bad share grows.' },
  { value: 'BULYAN', label: 'Bulyan — strictest filtering', param: 'fraction',
    help: 'Needs many devices: fewer than a quarter can be bad, and 20 devices support about 1 in 5.' },
  { value: 'CENTERED_CLIP', label: 'Centered clipping — caps how far one update can pull', param: 'tau',
    help: 'Bounds each update instead of discarding any. The radius should match a typical update size.' },
];

export function StartProjectModal({ isOpen, project, onClose, onSubmit }: StartProjectModalProps) {
  const [strategy, setStrategy] = useState('FedAvg');
  const [numRounds, setNumRounds] = useState(5);
  const [minClients, setMinClients] = useState(2);
  const [robustMethod, setRobustMethod] = useState('MEDIAN');
  const [byzantineFraction, setByzantineFraction] = useState(0.1);
  const [trimRatio, setTrimRatio] = useState(0.1);
  const [centeredClipTau, setCenteredClipTau] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  if (!project) return null;

  const isLlmLora = (project?.modelType ?? '').toUpperCase() === 'LLM_LORA';
  const showRobust = !isLlmLora && strategy === 'Robust';
  const selectedRule = ROBUST_RULES.find((r) => r.value === robustMethod);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      setIsLoading(true);
      const config: StartServerData = {
        strategy,
        numRounds: Number(numRounds),
        minClients: Number(minClients),
      };
      // Robust fields are added only for the Robust method, and only the parameter the chosen rule
      // reads. The backend rejects robust fields on any other method, so leftovers from a rule the user
      // looked at and then switched away from must never ride along.
      if (showRobust && selectedRule) {
        config.robustMethod = selectedRule.value;
        if (selectedRule.param === 'fraction') config.byzantineFraction = Number(byzantineFraction);
        if (selectedRule.param === 'trim') config.trimRatio = Number(trimRatio);
        if (selectedRule.param === 'tau') config.centeredClipTau = Number(centeredClipTau);
      }
      await onSubmit(project.id, config);
      // Reset form defaults upon success
      setStrategy('FedAvg');
      setNumRounds(5);
      setMinClients(2);
      setRobustMethod('MEDIAN');
      setByzantineFraction(0.1);
      setTrimRatio(0.1);
      setCenteredClipTau(1);
    } catch (err) {
      // Keep the modal open and surface the backend detail inline, so the
      // failure isn't hidden behind the modal on the route beneath it.
      setError(errorMessage(err, 'Could not start training. Please try again.'));
      log.error('startProject submit failed', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title="Start training"
      footer={
        <>
          <Button type="button" variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button type="submit" form="start-project-form" disabled={isLoading}>
            {isLoading ? 'Starting…' : 'Start training'}
          </Button>
        </>
      }
    >
      <p className="-mt-1 mb-5 text-body text-fg-muted">
        Set up this training run for <strong className="font-medium text-fg">{project.name}</strong>.
        You can change these any time you start again.
      </p>

      <form id="start-project-form" onSubmit={handleSubmit} className="flex flex-col gap-5">
        {error && (
          <p className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2.5 text-label text-danger">
            <AlertCircle className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
            {error}
          </p>
        )}

        {/* Strategy */}
        {isLlmLora ? (
          <FormField label="Training method">
            <div className="rounded-md border border-hairline bg-surface-2 px-3 py-2 text-caption text-fg-muted">
              FedLoRA (automatic for LoRA fine-tuning)
            </div>
          </FormField>
        ) : (
          <FormField label="Training method">
            <Select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              {STRATEGIES.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </Select>
          </FormField>
        )}

        {showRobust && (
          <>
            <FormField label="Aggregation rule" help={selectedRule?.help}>
              <Select value={robustMethod} onChange={(e) => setRobustMethod(e.target.value)}>
                {ROBUST_RULES.map((r) => (
                  <option key={r.value} value={r.value}>{r.label}</option>
                ))}
              </Select>
            </FormField>
            {selectedRule?.param === 'fraction' && (
              <FormField
                label="Share of devices that may be malicious"
                help="Your estimate, from 0 to 0.49 — 0.1 means about 1 in 10. It decides how many updates the rule sets aside."
              >
                <Input
                  type="number"
                  min="0"
                  max="0.49"
                  step="0.01"
                  value={byzantineFraction}
                  onChange={(e) => setByzantineFraction(Number(e.target.value))}
                  required
                />
              </FormField>
            )}
            {selectedRule?.param === 'trim' && (
              <FormField
                label="Share trimmed from each end"
                help="From 0 to 0.49. This is also the largest share of bad devices the rule tolerates."
              >
                <Input
                  type="number"
                  min="0"
                  max="0.49"
                  step="0.01"
                  value={trimRatio}
                  onChange={(e) => setTrimRatio(Number(e.target.value))}
                  required
                />
              </FormField>
            )}
            {selectedRule?.param === 'tau' && (
              <FormField label="Clipping radius" help="How far a single update may pull the model. Must be above 0.">
                <Input
                  type="number"
                  min="0.0001"
                  step="any"
                  value={centeredClipTau}
                  onChange={(e) => setCenteredClipTau(Number(e.target.value))}
                  required
                />
              </FormField>
            )}
          </>
        )}

        <div className="grid grid-cols-2 gap-4">
          <FormField label="Training rounds" help="How many times devices share progress.">
            <Input
              type="number"
              min="1"
              value={numRounds}
              onChange={(e) => setNumRounds(Number(e.target.value))}
              required
            />
          </FormField>
          <FormField label="Devices needed to start" help="Training begins once this many join.">
            <Input
              type="number"
              min="1"
              value={minClients}
              onChange={(e) => setMinClients(Number(e.target.value))}
              required
            />
          </FormField>
        </div>
      </form>
    </Modal>
  );
}
