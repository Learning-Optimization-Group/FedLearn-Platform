# Generated projection study: reproduction

Scope: [approved design](../../superpowers/specs/2026-10-27-projection-mechanism-design.md). This extension generates all data locally. It accepts no medical dataset paths and makes no network calls. Older reports and experiments remain frozen. The study does not claim independent medical validation or a novel optimizer.

## Environment

Use the existing [clean CPU profile](clean-reproduction.md), Python 3.12, from a tracked source checkout. No new dependencies are required for fitting or numerical auditing. Set `PYTHONPATH=framework/src:docs/artifacts/capstone/repro`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `PYTHONDONTWRITEBYTECODE=1`. All runner commands set PyTorch CPU threads to one.

In the commands below, replace `python` with the selected clean interpreter. Use new output directories on every invocation. An existing output is refused. Do not rerun failed scientific fits under the same study identity.

## Qualification before main measurement

```sh
python docs/artifacts/capstone/repro/run_projection_study.py preflight \
  --stage qualification --output .temp/projection-study/qualification-plan
python docs/artifacts/capstone/repro/run_projection_study.py execute \
  --plan .temp/projection-study/qualification-plan \
  --output .temp/projection-study/qualification \
  --checkpoint SOURCE_COMMIT
python docs/artifacts/capstone/repro/analyze_projection_study.py \
  --runs .temp/projection-study/qualification \
  --output .temp/projection-study/qualification-audit.json
```

`SOURCE_COMMIT` must contain every source binding in the manifest. Qualification has at most twelve cells and a 30-minute execution budget. It includes replay, zero padding, private/no-noise controls and attack budgets. Qualification labels and seeds are separate from main training seeds. Qualification scores do not authorize retuning the fixed recipe.

## Frozen main measurement

Only after successful independent qualification and a runtime projection below twelve hours:

```sh
python docs/artifacts/capstone/repro/run_projection_study.py preflight \
  --stage main --qualification .temp/projection-study/qualification \
  --output .temp/projection-study/main-plan
python docs/artifacts/capstone/repro/run_projection_study.py execute \
  --plan .temp/projection-study/main-plan --output .temp/projection-study/main \
  --checkpoint SOURCE_COMMIT
python docs/artifacts/capstone/repro/analyze_projection_study.py \
  --runs .temp/projection-study/main --output .temp/projection-study/main-audit.json
```

The main manifest contains exactly136 cells,2720 rounds and54400 client fits. The fitting ceiling is twelve hours. Failure retains partial journals and marks the batch failed. Never add seeds, choose a favorable basis or repeat cells to obtain a preferred result.

## Plotting and interpretation

Numerical analysis runs in the clean CPU profile. Render its aggregate JSON using the existing separate Matplotlib environment:

```sh
python docs/artifacts/capstone/repro/plot_projection_study.py \
  --audit .temp/projection-study/main-audit.json \
  --output .temp/projection-study/figures
```

That plotting interpreter is distinct from the fitting interpreter; the figure directory records its Matplotlib version. Figures show individual fixed-grid points and means, not discovery confidence intervals. Each attack effort budget has its own damaging-control decision. The teacher-aligned diagnostic subspace is an oracle control, not an implementable data-selected defense. Zero padding tests predictive invariance, not a new privacy guarantee.

The [gate review](projection-gate-review-2026-10-27.md) records preparation status before implementation. Qualification and measurement completion require their own actual receipts; this guide alone is not completion evidence.
