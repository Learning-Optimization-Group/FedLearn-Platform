// =============================================================================
// FedLearn Frontend — EditProjectModal (Ember design system)
// =============================================================================

import { useState, useEffect } from 'react';
import { Edit3 } from 'lucide-react';
import type { Project } from '../../services/apiServices';
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

const ARCH_LABELS: Record<keyof typeof modelOptions, string> = {
  CNN: 'Image model (CNN)',
  Transformer: 'Text model (Transformer)',
};

type ModelType = keyof typeof modelOptions;

interface EditProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onSubmit: (id: string, data: Partial<Project>) => void;
  onClose: () => void;
  isLoading?: boolean;
}

const labelClass = 'text-label font-medium text-fg';

export function EditProjectModal({ isOpen, project, onSubmit, onClose, isLoading = false }: EditProjectModalProps) {
  const [name, setName] = useState('');
  const [modelType, setModelType] = useState<ModelType>('CNN');
  const [modelName, setModelName] = useState('');
  const [optimizer, setOptimizer] = useState('');
  const [pretrainEpochs, setPretrainEpochs] = useState(0);

  useEffect(() => {
    if (project && isOpen) {
      setName(project.name);

      const type = (modelOptions as any)[project.modelType] ? (project.modelType as ModelType) : 'CNN';
      setModelType(type);
      setModelName(project.modelName);
      setOptimizer(project.optimizer);
      setPretrainEpochs(0); // pretrainEpochs isn't part of the DTO returning from backend typically, but initializing to 0
    }
  }, [project, isOpen]);

  // Handle cascading dropdowns
  useEffect(() => {
    if (project && isOpen) {
        if (!modelOptions[modelType].models.includes(modelName)) {
            setModelName(modelOptions[modelType].models[0]);
        }
        if (!modelOptions[modelType].optimizers.includes(optimizer)) {
            setOptimizer(modelOptions[modelType].optimizers[0]);
        }
    }
  }, [modelType, project, isOpen, modelName, optimizer]);

  if (!project) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(project.id, { name, modelType, modelName, optimizer, pretrainEpochs } as any);
  };

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <Edit3 strokeWidth={1.5} className="h-5 w-5 text-accent" />
          Edit project
        </span>
      }
    >
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
              {modelOptions[modelType]?.models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </div>
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Optimizer</label>
            <Select value={optimizer} onChange={(e) => setOptimizer(e.target.value)}>
              {modelOptions[modelType]?.optimizers.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </Select>
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
            disabled={isLoading || !name.trim()}
            className="flex-1"
          >
            {isLoading ? 'Saving…' : 'Save changes'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
