"""Effusion protocol v1 amendment A14: attacks built for robust rules, attacker-fraction sweep and
small-k trimming on R0, with and without central DP.

No DP: {mean, trimmed k=16, median}; DP eps 4: {mean, trimmed k=1, 4, 16}. Conditions: clean, and
{min-max, Fang trimmed-mean, IPM 0.1} at 4, 8 and 16 of 80 attacker slots; 5 seeds; 350 runs. The
clean cells A9 also ran must reproduce A9's margins exactly. Selection half only; holdout untouched.
"""
import argparse
import hashlib
import json
import math
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
CLIENTS = federated.CLIENTS
RECIPES = {False: {'lr': 0.1, 'clip_norm': 1.0}, True: {'lr': 0.03, 'clip_norm': 0.05}}  # A7 / A8, as in A9
DP = {'mode': 'central', 'epsilon': 4.0, 'delta': 1e-5}
# Rule label -> (engine rule, trimmed values per side or None).
RULES = {False: {'mean': ('mean', None), 'trim16': ('trimmed_mean', 16), 'median': ('median', None)},
         True: {'mean': ('mean', None), 'trim1': ('trimmed_mean', 1), 'trim4': ('trimmed_mean', 4),
                'trim16': ('trimmed_mean', 16)}}
ATTACKS = {'min_max': {'type': 'min_max', 'perturbation': 'std'},
           'fang_trim': {'type': 'fang_trim', 'b': 2.0},
           'ipm01': {'type': 'ipm', 'scale': 0.1}}
ATTACKERS = (4, 8, 16)
ATTACK_TYPES = {spec['type'] for spec in ATTACKS.values()}
# A9 cells this study repeats exactly: A14 name -> A9 name (per seed suffix).
A9_REPLAY = {'nodp_mean_clean': 'nodp_mean_clean', 'nodp_trim16_clean': 'nodp_trimmed_mean_clean',
             'nodp_median_clean': 'nodp_median_clean', 'dp4_mean_clean': 'dp4_mean_clean',
             'dp4_trim16_clean': 'dp4_trimmed_mean_clean'}
CONTRAST_SEED = 20261033


def trim_ratio(k, clients=CLIENTS):
    """The engine's trim ratio for k values per side; the engine trims floor(ratio * n)."""
    ratio = k / clients
    if math.floor(ratio * clients) != k:
        raise Refusal(f'trim_ratio_not_exact:{k}')
    return ratio


def conditions():
    out = {'clean': []}
    for label, spec in ATTACKS.items():
        for m in ATTACKERS:
            out[f'{label}_m{m}'] = [{**spec, 'fraction': m / CLIENTS}]
    return out


def cells():
    out = []
    for private in (False, True):
        tag = 'dp4' if private else 'nodp'
        for label, (rule, k) in RULES[private].items():
            for condition, behaviors in conditions().items():
                for seed in SEEDS:
                    out.append({'private': private, 'rule': rule, 'ruleLabel': label, 'k': k, 'condition': condition,
                                'behaviors': behaviors, 'seed': seed, 'name': f'{tag}_{label}_{condition}_{seed}'})
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
    raw_a9 = (Path(args.a9_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a9).hexdigest() != args.a9_sha256:
        raise Refusal('a9_hash_mismatch')
    a9 = np.load(Path(args.a9_dir) / 'selection-margins.npz')
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
    staging = Path(tempfile.mkdtemp(prefix='.strong-attacks-', dir=output.parent))
    started = time.perf_counter()
    margins, runs, replay = {}, [], {}
    try:
        for cell in cells():
            private = cell['private']
            rule_params = None if cell['k'] is None else {'trim_ratio': trim_ratio(cell['k'])}
            record, margin = federated.run_one(
                cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                clients=CLIENTS, features_path=args.historical_cache,
                report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256,
                info={'representation': 'R0', 'amendment': 'A14'}, privacy=DP if private else None,
                recipe=RECIPES[private], rule=cell['rule'], rule_params=rule_params, behaviors=cell['behaviors'])
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            last = record['rounds'][-1]
            notes = sorted({str(line['attack_note']) for line in record['rounds'] if line.get('attack_note')})
            runs.append({k: v for k, v in cell.items() if k != 'behaviors'} | {
                'finite': finite, 'refusals': len(record['meta']['refusals']),
                'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                'attackers': sum(1 for kinds in record['meta']['behavior_assignment'].values()
                                 if set(kinds) & ATTACK_TYPES),
                'clipRateLastRound': last['privacy']['clip_rate'],
                'deviationRatioLastRound': (last.get('robustness') or {}).get('deviation_ratio'),
                'sensitivityLastRound': last['privacy'].get('sensitivity'),
                'noiseStdLastRound': last['privacy'].get('noise_std'),
                'accountedEpsilon': record['meta']['accounted_epsilon'], 'attackNotes': notes})
            if finite:
                margins[cell['name']] = margin
            stem = cell['name'].rsplit('_', 1)[0]
            if stem in A9_REPLAY:
                replay[cell['name']] = bool(np.array_equal(margin, a9[f"{A9_REPLAY[stem]}_{cell['seed']}"]))
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'attackers', 'auroc')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        contrasts = {}
        for private in (False, True):
            tag = 'dp4' if private else 'nodp'
            for label in RULES[private]:
                for condition in conditions():
                    names = [f'{tag}_{label}_{condition}_{s}' for s in SEEDS]
                    if not all(n in usable for n in names):
                        continue
                    contrasts[f'auroc|{tag}|{label}|{condition}'] = [{n: 1} for n in names]
                    if condition != 'clean':
                        contrasts[f'drop|{tag}|{label}|{condition}'] = [
                            {f'{tag}_{label}_clean_{s}': 1, f'{tag}_{label}_{condition}_{s}': -1} for s in SEEDS]
                    if label != 'mean':
                        pair = [(f'{tag}_{label}_{condition}_{s}', f'{tag}_mean_{condition}_{s}') for s in SEEDS]
                        if all(b in usable for _, b in pair):
                            contrasts[f'ruleVsMean|{tag}|{label}|{condition}'] = [{a: 1, b: -1} for a, b in pair]
        summary = {
            'schema': 'capstone.effusion.strong_attack_study.v1',
            'protocol': 'effusion-baseline-protocol-v1 amendment A14',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipes': {str(k): v for k, v in RECIPES.items()},
            'dp': DP, 'rules': {str(k): v for k, v in RULES.items()}, 'attacks': ATTACKS,
            'attackerSlots': list(ATTACKERS), 'runs': runs,
            'unusableCells': sorted({r['name'] for r in runs} - usable),
            'a9Replay': {'cells': replay, 'allExact': bool(replay) and all(replay.values())},
            'contrasts': linear_contrasts(y_selection, selection_patients, margins, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256, 'a9SummarySha256': args.a9_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_strong_attacks.py', 'run_effusion_federated.py', 'run_effusion_recipe.py')},
            'engineSourceSha256': {name: federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / 'simulation' / name)
                                   for name in ('engine.py', 'behaviors.py', 'config.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--a9-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--a9-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'unusable': summary['unusableCells'], 'a9Replay': summary['a9Replay']['allExact'],
                      'elapsedSeconds': summary['elapsedSeconds']}))
