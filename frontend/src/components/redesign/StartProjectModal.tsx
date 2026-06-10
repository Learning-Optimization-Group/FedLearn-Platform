import { useState } from 'react';
import { Play } from 'lucide-react';
import { Project } from '../../services/apiServices';
import { createLogger } from '../../lib/logger';
import { Modal, Input, Select, Button } from '../ui';

const log = createLogger('StartProjectModal');

interface StartProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onClose: () => void;
  onSubmit: (projectId: string, config: { strategy: string; numRounds: number; minClients: number }) => Promise<void>;
}

const labelClass = 'text-label font-medium text-fg';
const helpClass = 'text-caption text-fg-subtle';

// Plain-language descriptions for each training method (values stay as-is).
const STRATEGIES: { value: string; label: string }[] = [
  { value: 'FedAvg', label: 'Standard — averages everyone\'s learning (recommended)' },
  { value: 'DeComFL', label: 'Low-bandwidth — sends tiny updates' },
  { value: 'FoT', label: 'For text models' },
];

export function StartProjectModal({ isOpen, project, onClose, onSubmit }: StartProjectModalProps) {
  const [strategy, setStrategy] = useState('FedAvg');
  const [numRounds, setNumRounds] = useState(5);
  const [minClients, setMinClients] = useState(2);
  const [isLoading, setIsLoading] = useState(false);

  if (!project) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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
      log.error('startProject submit failed', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <Play strokeWidth={1.5} className="h-5 w-5 text-accent" />
          Start training
        </span>
      }
    >
      <p className="-mt-1 mb-5 text-body text-fg-muted">
        Set up this training run for <strong className="font-medium text-fg">{project.name}</strong>.
        You can change these any time you start again.
      </p>

      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        {/* Strategy */}
        <div className="flex flex-col gap-1.5">
          <label className={labelClass}>Training method</label>
          <Select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {STRATEGIES.map((s) => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </Select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Rounds */}
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Training rounds</label>
            <Input
              type="number"
              min="1"
              value={numRounds}
              onChange={(e) => setNumRounds(Number(e.target.value))}
              required
            />
            <span className={helpClass}>How many times devices share progress.</span>
          </div>
          {/* Min Clients */}
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Devices needed to start</label>
            <Input
              type="number"
              min="1"
              value={minClients}
              onChange={(e) => setMinClients(Number(e.target.value))}
              required
            />
            <span className={helpClass}>Training begins once this many join.</span>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex gap-3 mt-2 pt-4 border-t border-hairline">
          <Button
            type="button"
            variant="secondary"
            onClick={onClose}
            disabled={isLoading}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={isLoading}
            className="flex-1"
          >
            {isLoading ? 'Starting…' : <><Play strokeWidth={2} className="h-4 w-4 fill-current" /> Start training</>}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
