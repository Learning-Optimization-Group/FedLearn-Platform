// =============================================================================
// FedLearn Frontend — CreateProjectModal V2
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

const labelClass = 'text-caption font-semibold text-fg-muted uppercase tracking-wide';

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
          New Project
        </span>
      }
    >
      {/* Form */}
      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        {/* Project Name */}
        <div className="flex flex-col gap-2">
          <label className={labelClass}>Project Name</label>
          <Input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. ResNet50 Imaging"
            required
            autoFocus
          />
        </div>

        {/* Model Architecture */}
        <div className="flex flex-col gap-2">
          <label className={labelClass}>Architecture</label>
          <Select value={modelType} onChange={(e) => setModelType(e.target.value as ModelType)}>
            {Object.keys(modelOptions).map((type) => (
              <option key={type} value={type}>{type}</option>
            ))}
          </Select>
        </div>

        {/* Model + Optimizer row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-2">
            <label className={labelClass}>Model</label>
            <Select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              {modelOptions[modelType].models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </div>
          <div className="flex flex-col gap-2">
            <label className={labelClass}>Optimizer</label>
            <Select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
              {modelOptions[modelType].optimizers.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </Select>
          </div>
        </div>

        {/* Pre-train Epochs */}
        <div className="flex flex-col gap-2">
          <label className={labelClass}>Pre-train Epochs</label>
          <Input
            type="number"
            value={pretrainEpochs}
            onChange={(e) => setPretrainEpochs(Number(e.target.value))}
            min="0"
          />
        </div>

        {/* Buttons */}
        <div className="flex gap-3 mt-4 pt-4 border-t border-hairline">
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
            {isLoading ? 'Creating...' : 'Create Project'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
