// =============================================================================
// FedLearn Frontend — CreateProjectModal (Ember design system)
// =============================================================================

import { useState, useEffect } from 'react';
import { Sparkles } from 'lucide-react';
import { fetchModelRecipes, type ModelRecipe } from '../../services/apiServices';
import { Modal, Input, Select, Button } from '../ui';

// Last-resort fallback if the catalog can't be fetched (e.g. offline). The
// primary source is GET /api/model-recipes — this only keeps the modal usable.
const FALLBACK_RECIPES: ModelRecipe[] = [
  {
    key: 'CNN',
    displayName: 'Image model (CNN)',
    inputKind: 'image',
    classes: [],
    baseModels: ['net', 'ResNet', 'VGGNet', 'AlexNet'],
    optimizers: ['Adam', 'SGD', 'RMSprop', 'AdamW'],
  },
  {
    key: 'TRANSFORMER',
    displayName: 'Text model (Transformer)',
    inputKind: 'text',
    classes: [],
    baseModels: ['opt-125m', 'bert-tiny'],
    optimizers: ['AdamW', 'Adam'],
  },
];

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
  const [recipes, setRecipes] = useState<ModelRecipe[]>([]);
  const [recipesLoading, setRecipesLoading] = useState(false);
  const [modelType, setModelType] = useState('');
  const [modelName, setModelName] = useState('');
  const [optimizer, setOptimizer] = useState('');
  const [pretrainEpochs, setPretrainEpochs] = useState(0);

  // Fetch the model catalog when the modal opens.
  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    setRecipesLoading(true);
    fetchModelRecipes()
      .then((res) => {
        if (cancelled) return;
        const data = res.data?.length ? res.data : FALLBACK_RECIPES;
        setRecipes(data);
        setModelType(data[0].key);
      })
      .catch(() => {
        if (cancelled) return;
        setRecipes(FALLBACK_RECIPES);
        setModelType(FALLBACK_RECIPES[0].key);
      })
      .finally(() => {
        if (!cancelled) setRecipesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  const selectedRecipe = recipes.find((r) => r.key === modelType);

  // When the selected type changes, reset base model + optimizer to its first.
  useEffect(() => {
    if (!selectedRecipe) return;
    setModelName(selectedRecipe.baseModels[0] ?? '');
    setOptimizer(selectedRecipe.optimizers[0] ?? '');
  }, [modelType, selectedRecipe]);

  // Reset on close
  useEffect(() => {
    if (!isOpen) {
      setName('');
      setModelType('');
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
          <Select
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
            disabled={recipesLoading || recipes.length === 0}
          >
            {recipesLoading && <option value="">Loading models…</option>}
            {recipes.map((r) => (
              <option key={r.key} value={r.key}>{r.displayName}</option>
            ))}
          </Select>
        </div>

        {/* Model + Optimizer row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Base model</label>
            <Select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={recipesLoading || !selectedRecipe}
            >
              {selectedRecipe?.baseModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </div>
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Optimizer</label>
            <Select
              value={optimizer}
              onChange={(e) => setOptimizer(e.target.value)}
              disabled={recipesLoading || !selectedRecipe}
            >
              {selectedRecipe?.optimizers.map((o) => (
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
            disabled={isLoading || !name.trim() || !selectedRecipe}
            className="flex-1"
          >
            {isLoading ? 'Creating…' : 'Create project'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
