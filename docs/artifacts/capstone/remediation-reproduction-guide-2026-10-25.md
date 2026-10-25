# Reproducing the remediation evidence

25 October 2026. Use the capstone worktree and its frozen checkpoints. Nothing here authorizes reopening the Effusion holdout or the official test set.

## Public versus private evidence

Public protocols, manifests, aggregate tables and receipts describe the experiment and its outcome accounting. The source manifests bind the exact implementation and optional ByzFL 0.0.11 distribution. Patient-level manifests, feature arrays, saved heads, predictions and journals remain in the private research area. A public aggregate archive alone cannot reproduce training without access to the qualified NIH preparation and its matching feature bundle.

Provision the original pinned train/selection bundle using the documented preparation process and applicable data access terms. Do not substitute a similar dataset, regenerate a different standardizer or bypass report-hash validation. There is no new raw-image preprocessing or holdout scoring step in the remediation runners.

## Runtime and optional dependency isolation

Use Python 3.12 with the project's qualified Torch/NumPy/SciPy/scikit-learn versions. The public preflight manifests record the actual versions. ByzFL is an optional adapter dependency, isolated from the shared environment; upgrading its transitive dependencies is not part of reproduction.

The commands below assume the original qualified environment and optional distribution under `.temp/capstone-byzfl-20261023/deps`. Set `PYTHONPATH=framework/src:framework:.temp/capstone-byzfl-20261023/deps`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The drivers additionally require one Torch CPU thread. Use fresh output directories: overwriting or resuming an old directory is deliberately refused.

## Source checkpoints and ordered stages

| Stage | Local checkpoint | Primary receipt |
| --- | --- | --- |
| Four-cell qualification | 016677f | phase-2-qualification-verification-2026-10-25.json |
| Eighty-cell screen | 8ea4b7d | phase-3-screen-verification-2026-10-25.json |
| Forty-six-cell buckets | 5940d13 | phase-3-verification-2026-10-25.json |
| 196-cell finalists | 1a9f0ef | phase-4-preflight-verification-2026-10-25.json |

Run only one frozen stage from its matching source/runtime snapshot. Current later source is intentionally different from an earlier stage. The preserved `sources/` archives and prospective manifests identify the correct bytes; a later runner refusing an old manifest is an integrity check, not a reason to weaken the check. External ByzFL source entries are distribution-relative and must be resolved through the runner's source inventory.

Each real stage uses `preflight` to validate inputs, resolve configurations, create an exact source archive and write private/public plans with zero fitting. A local source/protocol/public-manifest checkpoint precedes `execute`. The execute command requires the private-plan SHA256, checkpoint commit and committed public-manifest path. The runner rechecks source, runtime, bundle, configurations and manifest before fitting. Failed native/ByzFL anchors stop later fitting.

For finalist reproduction, use `python docs/artifacts/capstone/repro/run_effusion_finalists.py preflight --help` and `python docs/artifacts/capstone/repro/run_effusion_finalists.py execute --help`. Supply the complete original screen/bucket summaries and their exact hashes. The driver reconstructs the declared policy choices and verifies complete parent accounting; it does not accept an arbitrary replacement pilot. The final private-plan SHA256 is `c8b261d059e02891ebecc3e4658217cce4d8e40aa3c1b43cda4afd09d123ddc9`; public-manifest SHA256 is `000c0180849716a7d95b37dd649d4d257313722ec5414b8b7c76bcf7b9f72a37`.

## Analysis and reporting without retraining

`repro/analyze_effusion_pilots.py` reconciles the complete screen and optional bucket stage. `python docs/artifacts/capstone/repro/analyze_effusion_finalists.py --help` exposes the final saved-head analysis. It accepts the run directory, expected summary hash and a fresh aggregate output directory; it performs no fitting. Its only data loader is the qualified training/selection bundle loader. The frozen final analysis requires exactly 2,000 whole-patient bootstrap draws at seed 20261025 and all 196 outcomes before inferential results are produced.

The analyzer checks raw configuration and seed identity, deterministic client/attack assignments, workload and projection hashes, complete twenty-round eighty-client cohorts, calibration, final heads and five rescored metrics. It independently computes rank-based AUROC. Outputs are aggregate `summary.json`, `arms.csv`, `contrasts.csv`, `report.md` and `scoring-audit.json`; patient identities, margins and model arrays are not exported.

`repro/plot_remediation_results.py` consumes only complete public pilot/finalist summary JSON. It writes PNG/SVG figures, exact plot data and hashes. `repro/render_integrated_report.py` renders the source manuscript into fresh PDF/standalone HTML and page PNGs. The renderer validates caller-specified result tokens and records source, dependency, font, figure and output hashes. Final page inspection is required in addition to text extraction.

## Meaningful checks

The focused reproduction tests cover frozen identities and budgets, finite-iteration ERM bounds, owned attack information, baseline failures, consistently rehashed record swaps, statistical signs/targets, incomplete-matrix suppression and aggregate export boundaries. Run those tests from their matching checkpoint before real fitting. The framework gate uses `PYTHONPATH=src python -m pytest -q` in `framework/`; some gRPC tests require localhost socket permissions. The retained permitted run passed 2,378 tests with fourteen deselected. A sandbox-denied network test is not evidence of a numerical training defect.

## Interpretation after replay

Exact replay supports the integrity of the reported experiment. It does not remove development-set selection, show external clinical performance or establish joint privacy for all research outputs. Private clean cells use the stated simulated-client adjacency; attacked cells protect an honest client with fixed coalition identities/data conditional on released history. Shared/public experiment randomness and research diagnostics must not be deployed as a protected release design.
