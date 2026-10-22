"""Effusion protocol v1 amendment A7: bounded federated recipe study on R0, then confirmation.

Grid of client LR x update clip, chosen on two seeds by federated selection AUROC; the
chosen recipe is then run on all five seeds as federated, central-SGD and central DP
(eps 8/4/1). Selection half only; the holdout and official test are not touched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_effusion_federated as federated  # noqa: E402  (also pins this worktree's fedlearn)

Refusal = federated.Refusal
SEEDS = federated.SEEDS
GRID = [(lr, clip) for lr in (0.003, 0.01, 0.03, 0.1) for clip in (0.05, 0.25, 1.0)]
SELECTION_SEEDS = (20261101, 20261102)
EPSILONS = (1.0, 4.0, 8.0)
DELTA = 1e-5
CONTRAST_SEED = 20261026


def choose(results):
    """Index of the eligible configuration with the highest mean AUROC; ties -> smaller clip, then smaller LR."""
    eligible = [i for i, r in enumerate(results) if r['eligible']]
    if not eligible:
        raise Refusal('no_eligible_recipe')
    return min(eligible, key=lambda i: (-results[i]['meanAuroc'], results[i]['clip'], results[i]['lr']))


def confirmation_cells():
    cells = []
    for seed in SEEDS:
        cells.append({'kind': 'fed', 'seed': seed, 'epsilon': None, 'name': f'fed_{seed}'})
        cells.append({'kind': 'central', 'seed': seed, 'epsilon': None, 'name': f'central_{seed}'})
        cells += [{'kind': 'dp', 'seed': seed, 'epsilon': eps, 'name': f'dp_eps{eps:g}_{seed}'} for eps in EPSILONS]
    return cells


def linear_contrasts(y, patients, scores, contrasts, *, reps=2000, seed=CONTRAST_SEED):
    """Seed-averaged linear combinations of AUROCs with patient-cluster bootstrap intervals.

    `contrasts` maps a name to a list (one per seed) of {score_name: coefficient}.
    """
    from sklearn.metrics import roc_auc_score
    from fit_effusion_gate import cluster_bootstrap

    needed = sorted({n for terms in contrasts.values() for seed_terms in terms for n in seed_terms})

    def evaluate(index):
        auc = {n: roc_auc_score(y[index], scores[n][index]) for n in needed}  # once per replicate
        return {name: float(np.mean([sum(c * auc[n] for n, c in seed_terms.items()) for seed_terms in terms]))
                for name, terms in contrasts.items()}

    point = evaluate(np.arange(len(y)))
    draws = {name: [] for name in contrasts}
    for index in cluster_bootstrap(patients, reps=reps, seed=seed):
        if y[index].min() == y[index].max():
            continue
        for name, value in evaluate(index).items():
            draws[name].append(value)
    return {name: {'point': point[name],
                   'interval95': [float(np.percentile(draws[name], 2.5)), float(np.percentile(draws[name], 97.5))]}
            for name in contrasts}


def main(args):
    import effusion_labels
    import fit_effusion_gate
    from sklearn.metrics import roc_auc_score
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    raw_gate = (Path(args.gate_dir) / 'gate.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.gate_sha256:
        raise Refusal('gate_hash_mismatch')
    authority = federated.check_gate(json.loads(raw_gate), args.amendment)
    raw_dp = (Path(args.dp_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_dp).hexdigest() != args.dp_sha256:
        raise Refusal('dp_summary_hash_mismatch')
    previous = np.load(Path(args.dp_dir) / 'selection-margins.npz')
    convex = np.load(Path(args.gate_dir) / 'selection-scores.npz')['R0']
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    train_x, selection_x = features['R0']
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    mean, std = fit_effusion_gate.standardiser(train_x)
    if not (np.array_equal(mean, heads['R0_mean']) and np.array_equal(std, heads['R0_std'])):
        raise Refusal('standardiser_mismatch')
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    train_patients = [r['patient_id'] for r in labels['train']['rows']]
    selection_patients = np.array([r['patient_id'] for r in labels['selection']['rows']])
    common = dict(features_path=args.historical_cache, report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.recipe-', dir=output.parent))
    started = time.perf_counter()

    def execute(name, seed, clients, recipe, privacy=None):
        record, margin = federated.run_one(name, seed, train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                                           clients=clients, info={'representation': 'R0', 'amendment': 'A7'},
                                           privacy=privacy, recipe=recipe, **common)
        (staging / f'{name}.json').write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
        last = record['rounds'][-1]
        finite = bool(np.isfinite(margin).all())
        run = {'name': name, 'seed': seed, 'clients': clients, **recipe, 'epsilon': None if privacy is None else privacy['epsilon'],
               'finite': finite, 'refusals': len(record['meta']['refusals']),
               'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
               'clipRateLastRound': last['privacy']['clip_rate'], 'noiseToSignalLastRound': last['privacy']['noise_to_signal']}
        print(json.dumps({k: run[k] for k in ('name', 'finite', 'refusals')}), flush=True)
        return run, margin

    try:
        grid = []
        for lr, clip in GRID:
            recipe = {'lr': lr, 'clip_norm': clip}
            runs = [execute(f'grid_lr{lr:g}_clip{clip:g}_{s}', s, federated.CLIENTS, recipe)[0] for s in SELECTION_SEEDS]
            eligible = all(r['finite'] and r['refusals'] == 0 for r in runs)
            grid.append({'lr': lr, 'clip': clip, 'eligible': eligible, 'runs': runs,
                         'meanAuroc': float(np.mean([r['auroc'] for r in runs])) if eligible else None,
                         'meanClipRateLastRound': float(np.mean([r['clipRateLastRound'] for r in runs]))})
        chosen = grid[choose(grid)]
        recipe = {'lr': chosen['lr'], 'clip_norm': chosen['clip']}
        margins, confirmation = {}, []
        for cell in confirmation_cells():
            privacy = None if cell['kind'] != 'dp' else {'mode': 'central', 'epsilon': cell['epsilon'], 'delta': DELTA}
            clients = 1 if cell['kind'] == 'central' else federated.CLIENTS
            run, margin = execute(cell['name'], cell['seed'], clients, recipe, privacy)
            confirmation.append(run)
            margins[cell['name']] = margin
        scores = {**margins, 'convex': convex, **{f'old_{k}': previous[k] for k in previous.files if k.startswith('R0_')}}
        contrasts = {
            'recipeGain(fed_new-fed_old)': [{f'fed_{s}': 1, f'old_R0_clean_{s}': -1} for s in SEEDS],
            'fed-central': [{f'fed_{s}': 1, f'central_{s}': -1} for s in SEEDS],
            'fed-convex': [{f'fed_{s}': 1, 'convex': -1} for s in SEEDS],
            'central-convex': [{f'central_{s}': 1, 'convex': -1} for s in SEEDS],
        }
        for eps in EPSILONS:
            contrasts[f'dpCost_eps{eps:g}'] = [{f'fed_{s}': 1, f'dp_eps{eps:g}_{s}': -1} for s in SEEDS]
            contrasts[f'dpCostChange_eps{eps:g}(new-old)'] = [
                {f'fed_{s}': 1, f'dp_eps{eps:g}_{s}': -1, f'old_R0_clean_{s}': -1, f'old_R0_eps{eps:g}_{s}': 1} for s in SEEDS]
            contrasts[f'dpUtilityGain_eps{eps:g}(new-old)'] = [{f'dp_eps{eps:g}_{s}': 1, f'old_R0_eps{eps:g}_{s}': -1} for s in SEEDS]
        summary = {
            'schema': 'capstone.effusion.recipe_study.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A7',
            'gateAuthority': authority, 'representation': 'R0 (A6)',
            'scope': 'Exploratory optimiser check on the selection half (A3/A7); holdout untouched.',
            'grid': grid, 'selectionSeeds': list(SELECTION_SEEDS), 'chosenRecipe': recipe,
            'confirmation': confirmation,
            'seedMeans': {kind: float(np.mean([r['auroc'] for r in confirmation if r['name'].startswith(kind)]))
                          for kind in ['fed_', 'central_'] + [f'dp_eps{e:g}_' for e in EPSILONS]},
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrasts),
            'convexR0Auroc': float(roc_auc_score(y_selection, convex)),
            'inputs': {'labelInputs': provenance, 'dpSummarySha256': args.dp_sha256, 'gateSha256': args.gate_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in ('run_effusion_recipe.py', 'run_effusion_federated.py')},
            'engineSourceSha256': federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / 'simulation' / 'engine.py'),
            'elapsedSeconds': round(time.perf_counter() - started, 1),
        }
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        np.savez(staging / 'selection-margins.npz', **margins)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--dp-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--dp-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'chosen': summary['chosenRecipe'], 'elapsedSeconds': summary['elapsedSeconds']}))
