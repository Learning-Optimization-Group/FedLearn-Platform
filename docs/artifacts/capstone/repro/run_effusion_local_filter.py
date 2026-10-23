"""Effusion protocol v1 amendment A20: noise-aware norm filtering under local DP on R0.

Local DP eps {8, 32, 128}, rule mean, 16/80 attackers. Filter on: clean, non-compliant and noise-matching
sign flip / label flip. Filter off: noise-matching only (the non-compliant and clean filter-off cells are
A12's, compared from its pinned output). 5 seeds, 105 runs. Selection half only; holdout untouched.
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
from run_effusion_attacks import BEHAVIORS as A9_BEHAVIORS  # noqa: E402
from run_effusion_recipe import linear_contrasts  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
EPSILONS = (8.0, 32.0, 128.0)
RECIPE = {'lr': 0.03, 'clip_norm': 0.05}
DELTA = 1e-5
NORM_FILTER = {'k': 4.0}
CONDITIONS = {
    'clean': [],
    'sign_flip': A9_BEHAVIORS['sign_flip'],
    'label_flip': A9_BEHAVIORS['label_flip'],
    'sign_flip_noisy': [{**b, 'privatize': True} for b in A9_BEHAVIORS['sign_flip']],
    'label_flip_noisy': [{**b, 'privatize': True} for b in A9_BEHAVIORS['label_flip']],
}
FILTER_OFF_CONDITIONS = ('sign_flip_noisy', 'label_flip_noisy')
ATTACK_TYPES = {'sign_flip_scale', 'label_flip'}
CONTRAST_SEED = 20261040


def cells():
    out = []
    for eps in EPSILONS:
        for filtered in (True, False):
            for condition in (CONDITIONS if filtered else FILTER_OFF_CONDITIONS):
                for seed in SEEDS:
                    tag = 'filter' if filtered else 'nofilter'
                    out.append({'epsilon': eps, 'filter': filtered, 'condition': condition, 'seed': seed,
                                'name': f'{tag}_eps{eps:g}_mean_{condition}_{seed}'})
    return out


def off_name(eps, condition, seed):
    """The filter-off margin for a cell: A12's for clean and non-compliant attacks, this study's otherwise."""
    if condition in FILTER_OFF_CONDITIONS:
        return f'nofilter_eps{eps:g}_mean_{condition}_{seed}'
    return f'a12_local_eps{eps:g}_mean_{condition}_{seed}'


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
    raw_a12 = (Path(args.a12_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a12).hexdigest() != args.a12_sha256:
        raise Refusal('a12_hash_mismatch')
    a12 = np.load(Path(args.a12_dir) / 'selection-margins.npz')
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
    staging = Path(tempfile.mkdtemp(prefix='.local-filter-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells():
            record, margin = federated.run_one(
                cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                clients=federated.CLIENTS, features_path=args.historical_cache,
                report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256,
                info={'representation': 'R0', 'amendment': 'A20'},
                privacy={'mode': 'local', 'epsilon': cell['epsilon'], 'delta': DELTA}, recipe=RECIPE,
                behaviors=CONDITIONS[cell['condition']], norm_filter=NORM_FILTER if cell['filter'] else None)
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            drops = [line['participants'].get('filtered') for line in record['rounds']]
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         'attackers': sum(1 for kinds in record['meta']['behavior_assignment'].values()
                                          if set(kinds) & ATTACK_TYPES),
                         'filteredHonest': None if not cell['filter'] else sum(d['honest'] for d in drops),
                         'filteredAttacker': None if not cell['filter'] else sum(d['attacker'] for d in drops),
                         'uploadsSeen': sum(line['participants']['admitted'] for line in record['rounds']),
                         'clientNoiseStd': record['rounds'][-1]['privacy']['client_noise_std']})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'auroc', 'filteredHonest',
                                                       'filteredAttacker')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        scores = {**margins, **{f'a12_{k}': a12[k] for k in a12.files}}
        contrasts = {}
        for eps in EPSILONS:
            e = f'{eps:g}'
            for condition in CONDITIONS:
                on = [f'filter_eps{e}_mean_{condition}_{s}' for s in SEEDS]
                if not all(n in usable for n in on):
                    continue
                contrasts[f'auroc|filter|eps{e}|{condition}'] = [{n: 1} for n in on]
                contrasts[f'filterOnMinusOff|eps{e}|{condition}'] = [
                    {f'filter_eps{e}_mean_{condition}_{s}': 1, off_name(eps, condition, s): -1} for s in SEEDS]
                if condition != 'clean':
                    contrasts[f'drop|filter|eps{e}|{condition}'] = [
                        {f'filter_eps{e}_mean_clean_{s}': 1, f'filter_eps{e}_mean_{condition}_{s}': -1} for s in SEEDS]
            for condition in FILTER_OFF_CONDITIONS:
                off = [f'nofilter_eps{e}_mean_{condition}_{s}' for s in SEEDS]
                if all(n in usable for n in off):
                    contrasts[f'auroc|nofilter|eps{e}|{condition}'] = [{n: 1} for n in off]
        summary = {
            'schema': 'capstone.effusion.local_filter.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A20',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE, 'normFilter': NORM_FILTER,
            'conditions': CONDITIONS, 'runs': runs, 'unusableCells': sorted({r['name'] for r in runs} - usable),
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256, 'a12SummarySha256': args.a12_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_local_filter.py', 'run_effusion_federated.py', 'run_effusion_attacks.py',
                              'run_effusion_recipe.py')},
            'frameworkSourceSha256': {name: federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / name) for name in
                                      ('simulation/engine.py', 'simulation/config.py', 'simulation/behaviors.py',
                                       'server/composed_strategy.py', 'privacy/local_dp.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--a12-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--a12-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'unusable': summary['unusableCells'], 'elapsedSeconds': summary['elapsedSeconds']}))
