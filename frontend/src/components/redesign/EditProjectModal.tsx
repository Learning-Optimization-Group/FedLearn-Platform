// =============================================================================
// FedLearn Frontend — EditProjectModal (Ember design system)
// =============================================================================

import { useState, useEffect } from 'react';
import { AlertCircle } from 'lucide-react';
import { fetchModelRecipes, errorMessage, type ModelRecipe, type Project } from '../../services/apiServices';
import { Modal, Input, Select, Button, FormField } from '../ui';

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
  onSubmit: (id: string, data: Partial<Project>) => Promise<void>;
  onClose: () => void;
  isLoading?: boolean;
}

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
  const [error, setError] = useState('');

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
    setError('');
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      await onSubmit(project.id, { name, modelType, modelName, optimizer, pretrainEpochs } as Partial<Project>);
    } catch (err) {
      // Keep the modal open and surface the backend detail inline, rather than
      // letting the failure render on the route hidden behind the modal.
      setError(errorMessage(err, 'Could not save changes. Please try again.'));
    }
  };

  return (
    <Modal
      open={isOpen}
      onClose={onClose}
      title="Edit project"
      footer={
        <>
          <Button type="button" variant="secondary" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button
            type="submit"
            form="edit-project-form"
            disabled={isLoading || !name.trim() || !selectedRecipe}
          >
            {isLoading ? 'Saving…' : 'Save changes'}
          </Button>
        </>
      }
    >
      <form id="edit-project-form" onSubmit={handleSubmit} className="flex flex-col gap-5">
        {error && (
          <p className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2.5 text-label text-danger">
            <AlertCircle className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
            {error}
          </p>
        )}

        <FormField label="Project name">
          <Input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. My first model"
            required
            autoFocus
          />
        </FormField>

        <FormField label="What kind of model?">
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
        </FormField>

        {/* Model + Optimizer row */}
        <div className="grid grid-cols-2 gap-4">
          <FormField label="Base model">
            <Select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={recipesLoading || !selectedRecipe}
            >
              {selectedRecipe?.baseModels.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </Select>
          </FormField>
          <FormField label="Optimizer">
            <Select
              value={optimizer}
              onChange={(e) => setOptimizer(e.target.value)}
              disabled={recipesLoading || !selectedRecipe}
            >
              {selectedRecipe?.optimizers.map((o) => (
                <option key={o} value={o}>{o}</option>
              ))}
            </Select>
          </FormField>
        </div>
      </form>
    </Modal>
  );
}
