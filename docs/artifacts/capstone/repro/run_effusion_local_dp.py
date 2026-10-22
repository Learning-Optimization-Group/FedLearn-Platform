"""Effusion protocol v1 amendment A11: local DP on R0 compared with central DP at equal epsilon.

25 runs: local DP at eps 1/4/8/32/128 x 5 seeds, A8 recipe. Contrasts against the A8 central-DP and
clean confirmation runs. Selection half only; the holdout was opened once (A10) and is not reused.
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
from run_effusion_recipe import linear_contrasts  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
EPSILONS = (1.0, 4.0, 8.0, 32.0, 128.0)
CENTRAL_EPSILONS = (1.0, 4.0, 8.0)
RECIPE = {'lr': 0.03, 'clip_norm': 0.05}
DELTA = 1e-5
CONTRAST_SEED = 20261030


def cells():
    return [{'epsilon': eps, 'seed': seed, 'name': f'local_eps{eps:g}_{seed}'} for eps in EPSILONS for seed in SEEDS]


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
    raw_a8 = (Path(args.dp_recipe_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a8).hexdigest() != args.dp_recipe_sha256:
        raise Refusal('a8_summary_hash_mismatch')
    a8 = np.load(Path(args.dp_recipe_dir) / 'selection-margins.npz')
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
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.local-dp-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells():
            record, margin = federated.run_one(
                cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                clients=federated.CLIENTS, features_path=args.historical_cache,
                report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256, info={'representation': 'R0', 'amendment': 'A11'},
                privacy={'mode': 'local', 'epsilon': cell['epsilon'], 'delta': DELTA}, recipe=RECIPE)
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            last = record['rounds'][-1]
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         'noiseMultiplier': record['meta']['noise_multiplier'], 'accountedEpsilon': record['meta']['accounted_epsilon'],
                         'clientNoiseStd': last['privacy']['client_noise_std'], 'serverNoiseStd': last['privacy']['server_noise_std'],
                         'serverClipNorm': record['meta']['server_clip_norm']})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        scores = {**margins, **{f'a8_{k}': a8[k] for k in a8.files}}
        contrasts = {}
        for eps in EPSILONS:
            e = f'{eps:g}'
            if not all(f'local_eps{e}_{s}' in usable for s in SEEDS):
                continue
            contrasts[f'auroc|local|eps{e}'] = [{f'local_eps{e}_{s}': 1} for s in SEEDS]
            contrasts[f'localCost|eps{e}'] = [{f'a8_clean_{s}': 1, f'local_eps{e}_{s}': -1} for s in SEEDS]
            if eps in CENTRAL_EPSILONS:
                contrasts[f'local-central|eps{e}'] = [{f'local_eps{e}_{s}': 1, f'a8_dp_eps{e}_{s}': -1} for s in SEEDS]
        summary = {
            'schema': 'capstone.effusion.local_dp.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A11',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE, 'delta': DELTA,
            'trustModel': 'local: no trusted aggregator; central: trusted aggregator adds noise. Equal epsilon is not an equivalent claim.',
            'privacyUnit': 'one simulated client (~280 patients), replace-one', 'runs': runs,
            'unusableCells': sorted({r['name'] for r in runs} - usable),
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'a8SummarySha256': args.dp_recipe_sha256, 'gateSha256': args.gate_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in ('run_effusion_local_dp.py', 'run_effusion_federated.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--dp-recipe-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--dp-recipe-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'unusable': summary['unusableCells'], 'elapsedSeconds': summary['elapsedSeconds']}))
