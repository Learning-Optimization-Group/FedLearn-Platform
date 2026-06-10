// =============================================================================
// FedLearn Frontend — CreateProjectModal (Ember design system)
// =============================================================================

import { useState, useEffect } from 'react';
import { Sparkles } from 'lucide-react';
import { Modal, Input, Select, Button } from '../ui';

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

// Plain-language labels for the architecture choice (values stay CNN/Transformer).
const ARCH_LABELS: Record<keyof typeof modelOptions, string> = {
  CNN: 'Image model (CNN)',
  Transformer: 'Text model (Transformer)',
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

const labelClass = 'text-label font-medium text-fg';
const helpClass = 'text-caption text-fg-subtle';

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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ name, modelType, modelName, optimizer, pretrainEpochs });
  };

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <Sparkles strokeWidth={1.5} className="h-5 w-5 text-accent" />
          New project
        </span>
      }
    >
      <p className="-mt-1 mb-5 text-body text-fg-muted">
        A project is one model you'll train together with your devices.
      </p>

      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        {/* Project Name */}
        <div className="flex flex-col gap-1.5">
          <label className={labelClass}>Project name</label>
          <Input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. My first model"
            required
            autoFocus
          />
        </div>

        {/* Model Architecture */}
        <div className="flex flex-col gap-1.5">
          <label className={labelClass}>What kind of model?</label>
          <Select value={modelType} onChange={(e) => setModelType(e.target.value as ModelType)}>
            {(Object.keys(modelOptions) as ModelType[]).map((type) => (
              <option key={type} value={type}>{ARCH_LABELS[type]}</option>
            ))}
          </Select>
        </div>

        {/* Model + Optimizer row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Base model</label>
            <Select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              {modelOptions[modelType].models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </div>
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Optimizer</label>
            <Select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
              {modelOptions[modelType].optimizers.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </Select>
          </div>
        </div>

        {/* Pre-train Epochs */}
        <div className="flex flex-col gap-1.5">
          <label className={labelClass}>Warm-up rounds</label>
          <Input
            type="number"
            value={pretrainEpochs}
            onChange={(e) => setPretrainEpochs(Number(e.target.value))}
            min="0"
          />
          <span className={helpClass}>
            Give the model a head start before devices join. Leave at 0 if unsure.
          </span>
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
            disabled={isLoading || !name.trim()}
            className="flex-1"
          >
            {isLoading ? 'Creating…' : 'Create project'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
