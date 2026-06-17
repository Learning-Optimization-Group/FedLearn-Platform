// =============================================================================
// FedLearn Frontend — EditProjectModal (Ember design system)
// =============================================================================

import { useState, useEffect } from 'react';
import { Edit3 } from 'lucide-react';
import { fetchModelRecipes, type ModelRecipe, type Project } from '../../services/apiServices';
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

interface EditProjectModalProps {
  isOpen: boolean;
  project: Project | null;
  onSubmit: (id: string, data: Partial<Project>) => void;
  onClose: () => void;
  isLoading?: boolean;
}

const labelClass = 'text-label font-medium text-fg';

// Case-insensitively resolve a project's modelType against the catalog,
// falling back to the first recipe if there's no match.
function resolveRecipe(recipes: ModelRecipe[], modelType: string): ModelRecipe | undefined {
  const target = modelType?.toLowerCase();
  return recipes.find((r) => r.key.toLowerCase() === target) ?? recipes[0];
}

export function EditProjectModal({ isOpen, project, onSubmit, onClose, isLoading = false }: EditProjectModalProps) {
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
        setRecipes(res.data?.length ? res.data : FALLBACK_RECIPES);
      })
      .catch(() => {
        if (!cancelled) setRecipes(FALLBACK_RECIPES);
      })
      .finally(() => {
        if (!cancelled) setRecipesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  // Once recipes are loaded (or the project changes), preselect the project's
  // type/model/optimizer, matching the type case-insensitively.
  useEffect(() => {
    if (!project || !isOpen || recipes.length === 0) return;
    const recipe = resolveRecipe(recipes, project.modelType);
    setName(project.name);
    setModelType(recipe?.key ?? '');
    setModelName(project.modelName);
    setOptimizer(project.optimizer);
    setPretrainEpochs(0); // pretrainEpochs isn't part of the DTO returned by the backend; initialize to 0
  }, [project, isOpen, recipes]);

  const selectedRecipe = recipes.find((r) => r.key === modelType);

  // Handle cascading dropdowns: keep the existing base model/optimizer if it's
  // valid for the selected type, otherwise reset to the type's first option.
  useEffect(() => {
    if (!project || !isOpen || !selectedRecipe) return;
    if (!selectedRecipe.baseModels.includes(modelName)) {
      setModelName(selectedRecipe.baseModels[0] ?? '');
    }
    if (!selectedRecipe.optimizers.includes(optimizer)) {
      setOptimizer(selectedRecipe.optimizers[0] ?? '');
    }
  }, [modelType, project, isOpen, modelName, optimizer, selectedRecipe]);

  if (!project) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(project.id, { name, modelType, modelName, optimizer, pretrainEpochs } as Partial<Project>);
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
            {isLoading ? 'Saving…' : 'Save changes'}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
