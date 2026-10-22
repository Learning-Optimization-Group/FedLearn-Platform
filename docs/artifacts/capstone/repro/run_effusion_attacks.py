"""Effusion protocol v1 amendment A9: bounded attack study on R0 with and without central DP.

100 cells: no-DP {mean, trimmed_mean, median} and DP eps 4 {mean, trimmed_mean} x {clean, sign
flip, label flip, ALIE} x 5 seeds, at 20% attacker slots. Selection half only; holdout untouched.
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
RECIPES = {False: {'lr': 0.1, 'clip_norm': 1.0}, True: {'lr': 0.03, 'clip_norm': 0.05}}
DP = {'mode': 'central', 'epsilon': 4.0, 'delta': 1e-5}
BEHAVIORS = {
    'clean': [],
    'sign_flip': [{'type': 'sign_flip_scale', 'fraction': 0.2, 'scale': -4.0}],
    'label_flip': [{'type': 'label_flip', 'fraction': 0.2}],
    'alie': [{'type': 'alie', 'fraction': 0.2, 'z': 0.4888}],
}
CONTRAST_SEED = 20261028


def cells():
    out = []
    for private, rules in ((False, ('mean', 'trimmed_mean', 'median')), (True, ('mean', 'trimmed_mean'))):
        for rule in rules:
            for condition in BEHAVIORS:
                for seed in SEEDS:
                    tag = 'dp4' if private else 'nodp'
                    out.append({'epsilon': DP['epsilon'] if private else None, 'rule': rule, 'condition': condition,
                                'seed': seed, 'name': f'{tag}_{rule}_{condition}_{seed}'})
    return out


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
    staging = Path(tempfile.mkdtemp(prefix='.attacks-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells():
            private = cell['epsilon'] is not None
            record, margin = federated.run_one(
                cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                clients=federated.CLIENTS, features_path=args.historical_cache,
                report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256,
                info={'representation': 'R0', 'amendment': 'A9'}, privacy=DP if private else None,
                recipe=RECIPES[private], rule=cell['rule'], behaviors=BEHAVIORS[cell['condition']])
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            last = record['rounds'][-1]
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         'attackers': sum(1 for kinds in record['meta']['behavior_assignment'].values()
                                          if set(kinds) & {'sign_flip_scale', 'label_flip', 'alie'}),
                         'clipRateLastRound': last['privacy']['clip_rate'],
                         'deviationRatioLastRound': (last.get('robustness') or {}).get('deviation_ratio'),
                         'accountedEpsilon': record['meta']['accounted_epsilon']})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        contrasts, groups = {}, sorted({(r['epsilon'] is not None, r['rule']) for r in runs})
        for private, rule in groups:
            tag = 'dp4' if private else 'nodp'
            for condition in BEHAVIORS:
                names = [f'{tag}_{rule}_{condition}_{s}' for s in SEEDS]
                if all(n in usable for n in names):
                    contrasts[f'auroc|{tag}|{rule}|{condition}'] = [{n: 1} for n in names]
                    if condition != 'clean':
                        contrasts[f'drop|{tag}|{rule}|{condition}'] = [
                            {f'{tag}_{rule}_clean_{s}': 1, f'{tag}_{rule}_{condition}_{s}': -1} for s in SEEDS]
            if rule != 'mean':
                for condition in BEHAVIORS:
                    pair = [(f'{tag}_{rule}_{condition}_{s}', f'{tag}_mean_{condition}_{s}') for s in SEEDS]
                    if all(a in usable and b in usable for a, b in pair):
                        contrasts[f'ruleVsMean|{tag}|{rule}|{condition}'] = [{a: 1, b: -1} for a, b in pair]
        summary = {
            'schema': 'capstone.effusion.attack_study.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A9',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipes': {str(k): v for k, v in RECIPES.items()},
            'dp': DP, 'behaviors': BEHAVIORS, 'runs': runs,
            'unusableCells': sorted({r['name'] for r in runs} - usable),
            'contrasts': linear_contrasts(y_selection, selection_patients, margins, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in ('run_effusion_attacks.py', 'run_effusion_federated.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'unusable': summary['unusableCells'], 'elapsedSeconds': summary['elapsedSeconds']}))
