import { useState } from 'react';
import { Play, X } from 'lucide-react';
import { Project } from '../../services/apiServices';
import { createLogger } from '../../lib/logger';

const log = createLogger('StartProjectModal');

interface StartProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onClose: () => void;
  onSubmit: (
    projectId: string,
    config: { strategy: string; numRounds: number; minClients: number }
  ) => Promise<void>;
}

export function StartProjectModal({ isOpen, project, onClose, onSubmit }: StartProjectModalProps) {
  const [strategy, setStrategy] = useState('FedAvg');
  const [numRounds, setNumRounds] = useState(5);
  const [minClients, setMinClients] = useState(2);
  const [isLoading, setIsLoading] = useState(false);

  if (!isOpen || !project) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setIsLoading(true);
      await onSubmit(project.id, {
        strategy,
        numRounds: Number(numRounds),
        minClients: Number(minClients),
      });
      setStrategy('FedAvg');
      setNumRounds(5);
      setMinClients(2);
    } catch (err) {
      log.error('startProject submit failed', err);
    } finally {
      setIsLoading(false);
    }
  };

  const fieldStyle: React.CSSProperties = {
    backgroundColor: 'var(--input-background)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-color)',
  };
  const fieldClass =
    'w-full rounded-lg px-4 py-3 text-[14px] outline-none transition-colors focus:border-(--accent-primary)';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-sm font-sans"
      style={{ backgroundColor: 'oklch(0 0 0 / 0.5)' }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="w-full max-w-lg rounded-2xl shadow-2xl flex flex-col overflow-hidden"
        style={{
          background: 'var(--background-card)',
          border: '1px solid var(--border-color)',
          boxShadow: 'var(--shadow-strong)',
        }}
      >
        <div
          className="flex items-center justify-between p-5 pb-4"
          style={{ borderBottom: '1px solid var(--border-color)' }}
        >
          <div className="flex items-center gap-2">
            <Play className="w-5 h-5 text-(--accent-primary)" />
            <h2 className="text-[18px] font-display font-medium tracking-tight text-(--text-primary) m-0">
              Start Federation Server
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg transition-colors hover:bg-(--accent)"
            style={{ color: 'var(--text-secondary)' }}
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <form
          onSubmit={handleSubmit}
          className="p-6 flex flex-col gap-5"
          style={{ backgroundColor: 'var(--background-primary)' }}
        >
          <div className="flex flex-col gap-1 mb-1">
            <h3 className="text-[14px] font-medium text-(--text-primary)">
              Configure Run Parameters
            </h3>
            <p className="text-[13px] text-(--text-secondary)">
              Settings for orchestrating clients in federation{' '}
              <strong className="text-(--text-primary)">{project.name}</strong>.
            </p>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
              Aggregation Strategy
            </label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className={`${fieldClass} cursor-pointer appearance-none`}
              style={fieldStyle}
            >
              <option value="FedAvg">FedAvg</option>
              <option value="FedAdam">FedAdam</option>
              <option value="FedAdagrad">FedAdagrad</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
                Total Rounds
              </label>
              <input
                type="number"
                min="1"
                value={numRounds}
                onChange={(e) => setNumRounds(Number(e.target.value))}
                className={fieldClass}
                style={fieldStyle}
                required
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
                Min. Clients
              </label>
              <input
                type="number"
                min="1"
                value={minClients}
                onChange={(e) => setMinClients(Number(e.target.value))}
                className={fieldClass}
                style={fieldStyle}
                required
              />
            </div>
          </div>

          <div
            className="flex gap-3 mt-2 pt-4"
            style={{ borderTop: '1px solid var(--border-color)' }}
          >
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="flex-1 py-3 rounded-lg text-[14px] font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed hover:brightness-95"
              style={{
                backgroundColor: 'var(--background-secondary)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
              }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="flex-1 py-3 rounded-lg text-[14px] font-semibold transition-all hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:brightness-100 flex items-center justify-center gap-2"
              style={{
                backgroundColor: 'var(--accent-primary)',
                color: 'var(--primary-foreground)',
              }}
            >
              {isLoading ? (
                'Starting…'
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" /> Start Server
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
