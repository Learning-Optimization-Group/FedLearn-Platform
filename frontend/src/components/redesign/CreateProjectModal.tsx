// =============================================================================
// FedLearn Frontend — CreateProjectModal (Ember design system)
// =============================================================================

import { useState, useEffect, useRef } from 'react';
import { Sparkles, AlertCircle, Loader2 } from 'lucide-react';
import { fetchModelRecipes, fetchProject, errorMessage, type ModelRecipe, type Project } from '../../services/apiServices';
import { Modal, Input, Select, Button } from '../ui';

// BA-1: a freshly created project comes back INITIALIZING while the backend prepares its weights on an
// async worker. The modal polls the project until it resolves before closing.
const PREPARE_POLL_MS = 1500;
const PREPARE_TIMEOUT_MS = 120_000;
const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

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
  /** Creates the project and resolves with it. The returned project may be INITIALIZING (BA-1). */
  onSubmit: (data: {
    name: string;
    modelType: string;
    modelName: string;
    optimizer: string;
    pretrainEpochs: number;
    taskType?: string;
  }) => Promise<Project>;
  /** Called once the created project is persisted (ready or failed) so the parent can refresh its list. */
  onCreated: () => void;
  onClose: () => void;
  isLoading?: boolean;
}

const labelClass = 'text-label font-medium text-fg';
const helpClass = 'text-caption text-fg-subtle';

export function CreateProjectModalV2({ isOpen, onSubmit, onCreated, onClose, isLoading = false }: CreateProjectModalProps) {
  const [name, setName] = useState('');
  const [recipes, setRecipes] = useState<ModelRecipe[]>([]);
  const [recipesLoading, setRecipesLoading] = useState(false);
  const [modelType, setModelType] = useState('');
  const [modelName, setModelName] = useState('');
  const [optimizer, setOptimizer] = useState('');
  const [pretrainEpochs, setPretrainEpochs] = useState(0);
  const [taskType, setTaskType] = useState('SEQ_CLASSIFICATION');
  const [error, setError] = useState('');
  // 'form' shows the inputs; 'preparing' shows the spinner while init is polled (BA-1).
  const [phase, setPhase] = useState<'form' | 'preparing'>('form');
  // Flipped true when the modal closes so an in-flight poll stops touching state.
  const cancelledRef = useRef(false);

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
    setTaskType('SEQ_CLASSIFICATION');
  }, [modelType, selectedRecipe]);

  // Reset on close; also cancel any in-flight init poll so it stops updating state.
  useEffect(() => {
    if (isOpen) {
      cancelledRef.current = false;
    } else {
      cancelledRef.current = true;
      setName('');
      setModelType('');
      setPretrainEpochs(0);
      setError('');
      setPhase('form');
    }
  }, [isOpen]);

  // Poll a just-created project until its async model init finishes (BA-1).
  const pollUntilReady = async (id: string): Promise<'READY' | 'FAILED' | 'TIMEOUT'> => {
    const deadline = Date.now() + PREPARE_TIMEOUT_MS;
    while (Date.now() < deadline) {
      await delay(PREPARE_POLL_MS);
      if (cancelledRef.current) return 'TIMEOUT';
      try {
        const { data } = await fetchProject(id);
        if (data.status === 'FAILED') return 'FAILED';
        if (data.status !== 'INITIALIZING') return 'READY';
      } catch {
        // Transient error (e.g. a blip) — keep polling until the deadline.
      }
    }
    return 'TIMEOUT';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    try {
      const created = await onSubmit({ name, modelType, modelName, optimizer, pretrainEpochs,
                       ...(modelType === 'LLM_LORA' ? { taskType } : {}) });

      // BA-1: model init runs on an async worker, so the project comes back INITIALIZING. Show a
      // "Preparing" state and poll until it's ready (CREATED) or failed before we close — don't drop
      // the user onto a not-yet-trainable project.
      if (created.status === 'INITIALIZING') {
        setPhase('preparing');
        const outcome = await pollUntilReady(created.id);
        if (cancelledRef.current) return;   // modal was closed mid-poll
        if (outcome === 'FAILED') {
          setPhase('form');
          setError('Model preparation failed. You can delete this project and try again.');
          onCreated();                      // still refresh so the failed project shows in the list
          return;
        }
        // READY or TIMEOUT both close: on TIMEOUT the project is still preparing, and the list's
        // "Preparing" pill (plus the on-focus refresh) will catch up shortly.
      }

      onCreated();
      onClose();
    } catch (err) {
      // Keep the modal open and surface the backend detail inline, rather than
      // letting the failure render on the route hidden behind the modal.
      setPhase('form');
      setError(errorMessage(err, 'Could not create project. Please try again.'));
    }
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
      {phase === 'preparing' ? (
        <div className="flex flex-col items-center gap-4 py-10 text-center">
          <Loader2 className="h-8 w-8 animate-spin text-accent" strokeWidth={1.5} />
          <div className="flex flex-col gap-1">
            <p className="text-body font-medium text-fg">Preparing your model…</p>
            <p className="text-caption text-fg-subtle">
              Setting up the initial weights. This can take a moment.
            </p>
          </div>
        </div>
      ) : (
        <>
      <p className="-mt-1 mb-5 text-body text-fg-muted">
        A project is one model you'll train together with your devices.
      </p>

      <form onSubmit={handleSubmit} className="flex flex-col gap-5">
        {error && (
          <p className="flex items-center gap-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2.5 text-label text-danger">
            <AlertCircle className="h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
            {error}
          </p>
        )}

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

        {/* Task type — LLM_LORA only */}
        {modelType === 'LLM_LORA' && (
          <div className="flex flex-col gap-1.5">
            <label className={labelClass}>Task</label>
            <Select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
              <option value="SEQ_CLASSIFICATION">Classification (text → label)</option>
              <option value="CAUSAL_LM">Generation (instruction fine-tune)</option>
            </Select>
          </div>
        )}

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
        </>
      )}
    </Modal>
  );
}
