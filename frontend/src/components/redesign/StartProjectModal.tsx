import { useState } from 'react';
import { Play, Sparkles, X } from 'lucide-react';
import { Project } from '../../services/apiServices';

interface StartProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onClose: () => void;
  onSubmit: (projectId: string, config: { strategy: string; numRounds: number; minClients: number }) => Promise<void>;
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
      // Reset form defaults upon success
      setStrategy('FedAvg');
      setNumRounds(5);
      setMinClients(2);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const selectClass =
    "w-full bg-slate-900 border border-slate-700/50 rounded-md px-4 py-3 text-[14px] text-slate-200 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500/50 transition-all appearance-none cursor-pointer";
  const inputClass =
    "w-full bg-slate-900 border border-slate-700/50 rounded-md px-4 py-3 text-[14px] text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/30 focus:border-cyan-500/50 transition-all";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/60 backdrop-blur-sm font-sans"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-slate-900 border border-slate-700 w-full max-w-lg rounded-md shadow-2xl shadow-cyan-900/10 flex flex-col overflow-hidden text-slate-200">
        {/* Header */}
        <div className="flex items-center justify-between p-5 pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Play className="w-5 h-5 text-cyan-400" />
            <h2 className="text-[18px] font-semibold tracking-tight text-slate-100">Start Project Server</h2>
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center text-slate-400 bg-slate-800 hover:bg-slate-700 rounded-sm transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 flex flex-col gap-5 bg-slate-800/20">
          <div className="flex flex-col gap-1 mb-2">
            <h3 className="text-[14px] font-medium text-slate-200">Configure Run Parameter</h3>
            <p className="text-[13px] text-slate-400">Settings for orchestrating clients in project <strong className="text-slate-300">{project.name}</strong>.</p>
          </div>

          {/* Strategy */}
          <div className="flex flex-col gap-2">
            <label className="text-[12px] font-semibold text-slate-400 uppercase tracking-widest">Aggregation Strategy</label>
            <select value={strategy} onChange={(e) => setStrategy(e.target.value)} className={selectClass}>
              <option value="FedAvg">FedAvg</option>
              <option value="FedAdam">FedAdam</option>
              <option value="FedAdagrad">FedAdagrad</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* Rounds */}
            <div className="flex flex-col gap-2">
              <label className="text-[12px] font-semibold text-slate-400 uppercase tracking-widest">Total Rounds</label>
              <input
                type="number"
                min="1"
                value={numRounds}
                onChange={(e) => setNumRounds(Number(e.target.value))}
                className={inputClass}
                required
              />
            </div>
            {/* Min Clients */}
            <div className="flex flex-col gap-2">
              <label className="text-[12px] font-semibold text-slate-400 uppercase tracking-widest">Min. Clients</label>
              <input
                type="number"
                min="1"
                value={minClients}
                onChange={(e) => setMinClients(Number(e.target.value))}
                className={inputClass}
                required
              />
            </div>
          </div>

          {/* Buttons */}
          <div className="flex gap-3 mt-4 pt-4 border-t border-slate-700/50">
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-200 py-2.5 rounded-md text-[14px] font-medium tracking-tight border border-slate-600 transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="flex-1 bg-cyan-500 hover:bg-cyan-400 text-slate-950 py-2.5 rounded-md text-[14px] font-semibold tracking-tight transition-all duration-200 shadow-[0_0_15px_rgba(6,182,212,0.3)] disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? 'Starting...' : <><Play className="w-4 h-4 fill-current" /> Start Server</>}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
