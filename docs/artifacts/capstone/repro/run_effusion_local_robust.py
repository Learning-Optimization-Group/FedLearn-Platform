"""Effusion protocol v1 amendment A12: robust aggregation under local DP, with and without attackers.

135 runs: local DP eps {8, 32, 128} x {mean, trimmed_mean, median} x {clean, sign flip, label flip} x 5
seeds, A8 recipe. The eps-8 clean mean cells must reproduce A11 exactly. Selection half only.
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
from run_effusion_attacks import BEHAVIORS as ALL_BEHAVIORS  # noqa: E402
from run_effusion_recipe import linear_contrasts  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
EPSILONS = (8.0, 32.0, 128.0)
RULES = ('mean', 'trimmed_mean', 'median')
BEHAVIORS = {k: ALL_BEHAVIORS[k] for k in ('clean', 'sign_flip', 'label_flip')}
RECIPE = {'lr': 0.03, 'clip_norm': 0.05}
DELTA = 1e-5
CONTRAST_SEED = 20261031


def cells():
    return [{'epsilon': eps, 'rule': rule, 'condition': condition, 'seed': seed,
             'name': f'local_eps{eps:g}_{rule}_{condition}_{seed}'}
            for eps in EPSILONS for rule in RULES for condition in BEHAVIORS for seed in SEEDS]


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
    raw_a11 = (Path(args.local_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a11).hexdigest() != args.local_sha256:
        raise Refusal('a11_summary_hash_mismatch')
    a11 = np.load(Path(args.local_dir) / 'selection-margins.npz')
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
    staging = Path(tempfile.mkdtemp(prefix='.local-robust-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells():
            record, margin = federated.run_one(
                cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                clients=federated.CLIENTS, features_path=args.historical_cache,
                report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256, info={'representation': 'R0', 'amendment': 'A12'},
                privacy={'mode': 'local', 'epsilon': cell['epsilon'], 'delta': DELTA}, recipe=RECIPE,
                rule=cell['rule'], behaviors=BEHAVIORS[cell['condition']])
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            last = record['rounds'][-1]
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         'attackers': sum(1 for kinds in record['meta']['behavior_assignment'].values()
                                          if set(kinds) & {'sign_flip_scale', 'label_flip'}),
                         'clientNoiseStd': last['privacy']['client_noise_std'], 'accountedEpsilon': record['meta']['accounted_epsilon']})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        replay = {s: bool(f'local_eps8_mean_clean_{s}' in margins and
                          np.array_equal(margins[f'local_eps8_mean_clean_{s}'], a11[f'local_eps8_{s}'])) for s in SEEDS}
        contrasts = {}
        for eps in EPSILONS:
            e = f'{eps:g}'
            for rule in RULES:
                for condition in BEHAVIORS:
                    names = [f'local_eps{e}_{rule}_{condition}_{s}' for s in SEEDS]
                    if not all(n in usable for n in names):
                        continue
                    contrasts[f'auroc|eps{e}|{rule}|{condition}'] = [{n: 1} for n in names]
                    if condition != 'clean' and all(f'local_eps{e}_{rule}_clean_{s}' in usable for s in SEEDS):
                        contrasts[f'drop|eps{e}|{rule}|{condition}'] = [
                            {f'local_eps{e}_{rule}_clean_{s}': 1, f'local_eps{e}_{rule}_{condition}_{s}': -1} for s in SEEDS]
                    if rule != 'mean' and all(f'local_eps{e}_mean_{condition}_{s}' in usable for s in SEEDS):
                        contrasts[f'ruleVsMean|eps{e}|{rule}|{condition}'] = [
                            {f'local_eps{e}_{rule}_{condition}_{s}': 1, f'local_eps{e}_mean_{condition}_{s}': -1} for s in SEEDS]
        summary = {
            'schema': 'capstone.effusion.local_robust.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A12',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE, 'delta': DELTA,
            'attackerNoise': 'attackers do not privatise; honest clients add N(0,(2zC)^2)',
            'runs': runs, 'unusableCells': sorted({r['name'] for r in runs} - usable), 'replayOfA11': replay,
            'contrasts': linear_contrasts(y_selection, selection_patients, margins, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'a11SummarySha256': args.local_sha256, 'gateSha256': args.gate_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_local_robust.py', 'run_effusion_attacks.py', 'run_effusion_federated.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--local-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--local-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'replay': summary['replayOfA11'], 'unusable': summary['unusableCells'],
                      'elapsedSeconds': summary['elapsedSeconds']}))
