// =============================================================================
// FedLearn Frontend — CreateProjectModal (Instrument design system)
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

export function CreateProjectModalV2({
  isOpen,
  onSubmit,
  onClose,
  isLoading = false,
}: CreateProjectModalProps) {
  const [name, setName] = useState('');
  const [modelType, setModelType] = useState<ModelType>('CNN');
  const [modelName, setModelName] = useState(modelOptions.CNN.models[0]);
  const [optimizer, setOptimizer] = useState(modelOptions.CNN.optimizers[0]);
  const [pretrainEpochs, setPretrainEpochs] = useState(0);

  useEffect(() => {
    setModelName(modelOptions[modelType].models[0]);
    setOptimizer(modelOptions[modelType].optimizers[0]);
  }, [modelType]);

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
        {/* Header */}
        <div
          className="flex items-center justify-between p-5 pb-4"
          style={{ borderBottom: '1px solid var(--border-color)' }}
        >
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-(--accent-primary)" />
            <h2 className="text-[18px] font-display font-medium tracking-tight text-(--text-primary) m-0">
              New Federation
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

        {/* Form */}
        <form
          onSubmit={handleSubmit}
          className="p-6 flex flex-col gap-5"
          style={{ backgroundColor: 'var(--background-primary)' }}
        >
          {/* Project Name */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
              Federation Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. ResNet50 Imaging"
              className={fieldClass}
              style={fieldStyle}
              required
              autoFocus
            />
          </div>

          {/* Model Architecture */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
              Architecture
            </label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as ModelType)}
              className={`${fieldClass} cursor-pointer appearance-none`}
              style={fieldStyle}
            >
              {Object.keys(modelOptions).map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </div>

          {/* Model + Optimizer */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
                Model
              </label>
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className={`${fieldClass} cursor-pointer appearance-none`}
                style={fieldStyle}
              >
                {modelOptions[modelType].models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
                Optimizer
              </label>
              <select
                value={optimizer}
                onChange={(e) => setOptimizer(e.target.value)}
                className={`${fieldClass} cursor-pointer appearance-none`}
                style={fieldStyle}
              >
                {modelOptions[modelType].optimizers.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Pre-train Epochs */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">
              Pre-train Epochs
            </label>
            <input
              type="number"
              value={pretrainEpochs}
              onChange={(e) => setPretrainEpochs(Number(e.target.value))}
              min="0"
              className={fieldClass}
              style={fieldStyle}
            />
          </div>

          {/* Buttons */}
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
              disabled={isLoading || !name.trim()}
              className="flex-1 py-3 rounded-lg text-[14px] font-semibold transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:hover:brightness-100"
              style={{
                backgroundColor:
                  isLoading || !name.trim() ? 'var(--muted)' : 'var(--accent-primary)',
                color:
                  isLoading || !name.trim()
                    ? 'var(--muted-foreground)'
                    : 'var(--primary-foreground)',
              }}
            >
              {isLoading ? 'Creating…' : 'Create Federation'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
