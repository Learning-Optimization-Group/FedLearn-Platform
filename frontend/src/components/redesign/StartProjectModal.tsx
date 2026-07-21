import { useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { Project, errorMessage } from '../../services/apiServices';
import { createLogger } from '../../lib/logger';
import { Modal, Input, Select, Button, FormField } from '../ui';

const log = createLogger('StartProjectModal');

interface StartProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onClose: () => void;
  onSubmit: (projectId: string, config: { strategy: string; numRounds: number; minClients: number }) => Promise<void>;
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

export function StartProjectModal({ isOpen, project, onClose, onSubmit }: StartProjectModalProps) {
  const [strategy, setStrategy] = useState('FedAvg');
  const [numRounds, setNumRounds] = useState(5);
  const [minClients, setMinClients] = useState(2);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  if (!project) return null;

  const isLlmLora = (project?.modelType ?? '').toUpperCase() === 'LLM_LORA';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      setIsLoading(true);
      await onSubmit(project.id, {
        strategy,
        numRounds: Number(numRounds),
        minClients: Number(minClients),
      });
      // Reset form defaults upon success
      setStrategy('FedAvg');
      setNumRounds(5);
      setMinClients(2);
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
