// =============================================================================
// FedLearn Frontend — CreateProjectModal V2 (Apple-inspired)
// =============================================================================

import { useState, useEffect } from 'react';
import { X, Sparkles } from 'lucide-react';

const modelOptions = {
  CNN: {
    models: ['net', 'ResNet', 'VGGNet', 'AlexNet'],
    optimizers: ['Adam', 'SGD', 'RMSprop', 'AdamW'],
  },
  Transformer: {
    models: ['opt-125m', 'bert-tiny'],
    optimizers: ['AdamW', 'Adam'],
  },
};

type ModelType = keyof typeof modelOptions;

interface CreateProjectModalProps {
  isOpen: boolean;
  onSubmit: (data: {
    name: string;
    modelType: string;
    modelName: string;
    optimizer: string;
    pretrainEpochs: number;
  }) => void;
  onClose: () => void;
  isLoading?: boolean;
}

export function CreateProjectModalV2({ isOpen, onSubmit, onClose, isLoading = false }: CreateProjectModalProps) {
  const [name, setName] = useState('');
  const [modelType, setModelType] = useState<ModelType>('CNN');
  const [modelName, setModelName] = useState(modelOptions.CNN.models[0]);
  const [optimizer, setOptimizer] = useState(modelOptions.CNN.optimizers[0]);
  const [pretrainEpochs, setPretrainEpochs] = useState(0);

  useEffect(() => {
    setModelName(modelOptions[modelType].models[0]);
    setOptimizer(modelOptions[modelType].optimizers[0]);
  }, [modelType]);

  // Reset on close
  useEffect(() => {
    if (!isOpen) {
      setName('');
      setModelType('CNN');
      setPretrainEpochs(0);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ name, modelType, modelName, optimizer, pretrainEpochs });
  };

  const selectClass =
    "w-full bg-[#1c1c1e] border border-[rgba(255,255,255,0.1)] rounded-xl px-4 py-3 text-[15px] text-[#f5f5f7] focus:outline-none focus:ring-[3px] focus:ring-[#0a84ff]/30 focus:border-[#0a84ff]/50 transition-all appearance-none cursor-pointer";

  const inputClass =
    "w-full bg-[#1c1c1e] border border-[rgba(255,255,255,0.1)] rounded-xl px-4 py-3 text-[15px] text-[#f5f5f7] placeholder-[#86868b] focus:outline-none focus:ring-[3px] focus:ring-[#0a84ff]/30 focus:border-[#0a84ff]/50 transition-all";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-xl font-sans"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-[rgba(28,28,30,0.92)] border border-[rgba(255,255,255,0.1)] w-full max-w-lg rounded-[32px] shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden text-[#f5f5f7]">
        {/* Header */}
        <div className="flex items-center justify-between p-6 pb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#0a84ff]" />
            <h2 className="text-[22px] font-semibold tracking-tight">New Project</h2>
          </div>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center text-[#86868b] bg-[#3a3a3c] hover:bg-[rgba(255,255,255,0.2)] rounded-full transition-colors">
            <X className="w-[18px] h-[18px]" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="px-6 pb-6 flex flex-col gap-5">
          {/* Project Name */}
          <div className="flex flex-col gap-2">
            <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Project Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. ResNet50 Imaging"
              className={inputClass}
              required
              autoFocus
            />
          </div>

          {/* Model Architecture */}
          <div className="flex flex-col gap-2">
            <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Architecture</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value as ModelType)} className={selectClass}>
              {Object.keys(modelOptions).map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          {/* Model + Optimizer row */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Model</label>
              <select value={modelName} onChange={(e) => setModelName(e.target.value)} className={selectClass}>
                {modelOptions[modelType].models.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Optimizer</label>
              <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)} className={selectClass}>
                {modelOptions[modelType].optimizers.map((o) => (
                  <option key={o} value={o}>{o}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Pre-train Epochs */}
          <div className="flex flex-col gap-2">
            <label className="text-[13px] font-medium text-[#86868b] uppercase tracking-wider">Pre-train Epochs</label>
            <input
              type="number"
              value={pretrainEpochs}
              onChange={(e) => setPretrainEpochs(Number(e.target.value))}
              min="0"
              className={inputClass}
            />
          </div>

          {/* Buttons */}
          <div className="flex gap-3 mt-2">
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="flex-1 bg-[#2c2c2e] hover:bg-[#3a3a3c] text-[#f5f5f7] py-3 rounded-full text-[15px] font-medium tracking-tight transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading || !name.trim()}
              className="flex-1 bg-[#f5f5f7] text-black hover:bg-white py-3 rounded-full text-[15px] font-medium tracking-tight transition-all duration-200 transform active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Creating...' : 'Create Project'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
