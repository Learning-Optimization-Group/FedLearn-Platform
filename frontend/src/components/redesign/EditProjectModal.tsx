// =============================================================================
// FedLearn Frontend — EditProjectModal
// =============================================================================

import { useState, useEffect } from 'react';
import { X, Edit3 } from 'lucide-react';
import type { Project } from '../../services/apiServices';

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

interface EditProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onSubmit: (id: string, data: Partial<Project>) => void;
  onClose: () => void;
  isLoading?: boolean;
}

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


  if (!isOpen || !project) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(project.id, { name, modelType, modelName, optimizer, pretrainEpochs } as any);
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
            <Edit3 className="w-5 h-5 text-cyan-400" />
            <h2 className="text-[18px] font-semibold tracking-tight text-slate-100">Edit Project</h2>
          </div>
          <button onClick={onClose} className="w-7 h-7 flex items-center justify-center text-slate-400 bg-slate-800 hover:bg-slate-700 rounded-sm transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 flex flex-col gap-5 bg-slate-800/20">
          {/* Project Name */}
          <div className="flex flex-col gap-2">
            <label className="text-[12px] font-semibold text-slate-400 uppercase tracking-widest">Project Name</label>
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
            <label className="text-[12px] font-semibold text-slate-400 uppercase tracking-widest">Architecture</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value as ModelType)} className={selectClass}>
              {Object.keys(modelOptions).map((type) => (
                <option key={type} value={type}>{type}</option>
              ))}
            </select>
          </div>

          {/* Model + Optimizer row */}
          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-2">
              <label className="text-[12px] font-semibold text-slate-400 uppercase tracking-widest">Model</label>
              <select value={modelName} onChange={(e) => setModelName(e.target.value)} className={selectClass}>
                {modelOptions[modelType]?.models.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-[12px] font-semibold text-slate-400 uppercase tracking-widest">Optimizer</label>
              <select value={optimizer} onChange={(e) => setOptimizer(e.target.value)} className={selectClass}>
                {modelOptions[modelType]?.optimizers.map((o) => (
                  <option key={o} value={o}>{o}</option>
                ))}
              </select>
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
              disabled={isLoading || !name.trim()}
              className="flex-1 bg-cyan-500 hover:bg-cyan-400 text-slate-950 py-2.5 rounded-md text-[14px] font-semibold tracking-tight transition-all duration-200 shadow-[0_0_15px_rgba(6,182,212,0.3)] disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
