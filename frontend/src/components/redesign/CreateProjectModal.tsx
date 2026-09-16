// =============================================================================
// FedLearn Frontend — CreateProjectModal (Ember design system)
// =============================================================================

import { useState, useEffect, useRef } from 'react';
import { AlertCircle, Loader2 } from 'lucide-react';
import { fetchModelRecipes, fetchProject, errorMessage, type ModelRecipe, type Project } from '../../services/apiServices';
import { Modal, Input, Select, Button, FormField } from '../ui';

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
    trainingArm?: string;
  }) => Promise<Project>;
  /** Called once the created project is persisted (ready or failed) so the parent can refresh its list. */
  onCreated: () => void;
  onClose: () => void;
  isLoading?: boolean;
}

export function CreateProjectModalV2({ isOpen, onSubmit, onCreated, onClose, isLoading = false }: CreateProjectModalProps) {
  const [name, setName] = useState('');
  const [recipes, setRecipes] = useState<ModelRecipe[]>([]);
  const [recipesLoading, setRecipesLoading] = useState(false);
  const [modelType, setModelType] = useState('');
  const [modelName, setModelName] = useState('');
  const [optimizer, setOptimizer] = useState('');
  const [pretrainEpochs, setPretrainEpochs] = useState(0);
  const [taskType, setTaskType] = useState('SEQ_CLASSIFICATION');
  // Frozen-head vs full fine-tune. FULL is the default so a user who does not engage with the
  // choice gets exactly the pre-P1 behaviour.
  const [trainingArm, setTrainingArm] = useState('FULL');
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
  // A choice exists only when the recipe declares more than one arm. A single-arm recipe must not
  // render a trade-off — that would imply the un-offered arm had been evaluated for it.
  const offersArmChoice = (selectedRecipe?.supportedArms?.length ?? 0) > 1;
  const tradeoff = offersArmChoice ? selectedRecipe?.armTradeoff : undefined;
  const armFacts = tradeoff?.arms?.[trainingArm];

  // When the selected type changes, reset base model + optimizer to its first.
  useEffect(() => {
    if (!selectedRecipe) return;
    setModelName(selectedRecipe.baseModels[0] ?? '');
    setOptimizer(selectedRecipe.optimizers[0] ?? '');
    // Recipes differ in which arms they support; carrying a stale selection across a model change
    // could submit an arm this recipe never declared.
    setTrainingArm('FULL');
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
                       ...(modelType === 'LLM_LORA' ? { taskType } : {}),
                       // Sent only when the recipe actually offers a choice — otherwise the field
                       // would assert an arm the recipe never declared.
                       ...(offersArmChoice ? { trainingArm } : {}) });

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
      title="New project"
      footer={
        phase === 'form' ? (
          <>
            <Button type="button" variant="secondary" onClick={onClose} disabled={isLoading}>
              Cancel
            </Button>
            <Button
              type="submit"
              form="create-project-form"
              disabled={isLoading || !name.trim() || !selectedRecipe}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" strokeWidth={2} />
                  Creating…
                </>
              ) : (
                'Create project'
              )}
            </Button>
          </>
        ) : undefined
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

      <form id="create-project-form" onSubmit={handleSubmit} className="flex flex-col gap-5">
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

        {/* Training arm — only when the recipe offers a choice, with the MEASURED trade-off.
            The numbers come from the catalog (generated from the research record), never from
            this component, so the UI cannot drift from the measurement. */}
        {offersArmChoice && (
          <div className="flex flex-col gap-3">
            {/* FormField wires its generated id onto a SINGLE control child, so the trade-off
                panel is a sibling rather than a second child — otherwise the label stops being
                programmatically associated with the select. */}
            <FormField
              label="Training arm"
              help="Which parameters each device trains and sends back."
            >
              <Select value={trainingArm} onChange={(e) => setTrainingArm(e.target.value)}>
                {selectedRecipe?.supportedArms?.map((a) => (
                  <option key={a} value={a}>
                    {a === 'FROZEN_HEAD' ? 'Frozen backbone — train the head only' : 'Full fine-tune'}
                  </option>
                ))}
              </Select>
            </FormField>

            {/* The MEASURED trade-off. Every number here comes from the catalog, which generates
                it from the research record — this component never states a figure of its own. */}
            {tradeoff && (
              <div className="rounded-lg border border-border bg-surface-muted p-3 text-sm">
                <p className="font-medium text-ink">{tradeoff.headline}</p>

                {armFacts?.summary && <p className="mt-2 text-muted">{armFacts.summary}</p>}
                {/* Only the NEGATIVE case gets its own line. The record's summary already states
                    on-device capability for the arm that has it, and repeating it would be
                    duplicate text; infeasibility is the actionable half. */}
                {armFacts?.ondeviceFeasible === false && (
                  <p className="mt-1 text-muted">
                    Not feasible on-device — this arm needs a datacenter GPU.
                  </p>
                )}

                {tradeoff.measuredOn && (
                  <p className="mt-2 text-xs text-muted">
                    Measured on {tradeoff.measuredOn.task}
                    {tradeoff.measuredOn.protocol ? ` — ${tradeoff.measuredOn.protocol}` : ''}.
                  </p>
                )}

                {/* Caveats are not garnish: the communication ratio is round-budget dependent and
                    the accuracy and latency figures come from different hardware. Showing the
                    numbers without them would state more than was measured. */}
                {tradeoff.caveats && tradeoff.caveats.length > 0 && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs text-muted">
                      What this measurement does not establish
                    </summary>
                    <ul className="mt-1 list-disc pl-4 text-xs text-muted">
                      {tradeoff.caveats.map((c) => <li key={c}>{c}</li>)}
                    </ul>
                  </details>
                )}
              </div>
            )}
          </div>
        )}

        {/* Task type — LLM_LORA only */}
        {modelType === 'LLM_LORA' && (
          <FormField label="Task">
            <Select value={taskType} onChange={(e) => setTaskType(e.target.value)}>
              <option value="SEQ_CLASSIFICATION">Classification (text → label)</option>
              <option value="CAUSAL_LM">Generation (instruction fine-tune)</option>
            </Select>
          </FormField>
        )}

        <FormField
          label="Warm-up rounds"
          help="Give the model a head start before devices join. Leave at 0 if unsure."
        >
          <Input
            type="number"
            value={pretrainEpochs}
            onChange={(e) => setPretrainEpochs(Number(e.target.value))}
            min="0"
          />
        </FormField>
      </form>
        </>
      )}
    </Modal>
  );
}
