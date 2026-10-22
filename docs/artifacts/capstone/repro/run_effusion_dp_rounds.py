"""Effusion protocol v1 amendment A15: number of central-DP releases on R0.

T in {1, 2, 5, 10, 20} rounds x {clean, claimed eps 8, claimed eps 4} x 5 seeds, A8 recipe, z
calibrated exactly with GDP for each T (as in A13). The T = 20 cells must reproduce A13 (DP) and
A8 (clean) exactly. Selection half only; holdout untouched.
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
from run_effusion_gdp_recipe import DELTA, RECIPE, calibration, check_calibration  # noqa: E402
from run_effusion_recipe import linear_contrasts  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
ROUNDS = (1, 2, 5, 10, 20)
PRIVACY = (None, 8.0, 4.0)
CONTRAST_SEED = 20261034


def privacy_label(epsilon):
    return 'clean' if epsilon is None else f'eps{epsilon:g}'


def cells():
    return [{'rounds': t, 'epsilon': eps, 'seed': s, 'name': f'T{t}_{privacy_label(eps)}_{s}'}
            for t in ROUNDS for eps in PRIVACY for s in SEEDS]


def replay_source(cell):
    """The earlier margin a T = 20 cell must equal: A13 for DP, A8 for clean; None otherwise."""
    if cell['rounds'] != 20:
        return None
    if cell['epsilon'] is None:
        return 'a8', f"clean_{cell['seed']}"
    return 'a13', f"gdp_eps{cell['epsilon']:g}_{cell['seed']}"


def _pinned(directory, expected, label):
    raw = (Path(directory) / 'summary.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise Refusal(f'{label}_hash_mismatch')
    return np.load(Path(directory) / 'selection-margins.npz')


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
    earlier = {'a8': _pinned(args.a8_dir, args.a8_sha256, 'a8'), 'a13': _pinned(args.a13_dir, args.a13_sha256, 'a13')}
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
    calibrations = {(t, eps): calibration(eps, rounds=t) for t in ROUNDS for eps in PRIVACY if eps is not None}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.dp-rounds-', dir=output.parent))
    started = time.perf_counter()
    margins, runs, replay = {}, [], {}
    try:
        for cell in cells():
            cal = calibrations.get((cell['rounds'], cell['epsilon']))
            privacy = None if cal is None else {'mode': 'central', 'epsilon': cal['rdpEquivalentEpsilon'], 'delta': DELTA}
            record, margin = federated.run_one(
                cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                clients=federated.CLIENTS, features_path=args.historical_cache,
                report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256, rounds=cell['rounds'],
                info={'representation': 'R0', 'amendment': 'A15'}, privacy=privacy, recipe=RECIPE)
            exact = (None if cal is None else
                     check_calibration(cal, float(record['meta']['noise_multiplier']), rounds=cell['rounds']))
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            last = record['rounds'][-1]
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         'noiseMultiplier': record['meta']['noise_multiplier'], 'exactEpsilon': exact,
                         'noiseStdLastRound': last['privacy']['noise_std'],
                         'noiseToSignalLastRound': last['privacy']['noise_to_signal'],
                         'clipRateLastRound': last['privacy']['clip_rate']})
            if finite:
                margins[cell['name']] = margin
            source = replay_source(cell)
            if source is not None:
                replay[cell['name']] = bool(np.array_equal(margin, earlier[source[0]][source[1]]))
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'auroc')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        contrasts = {}
        for t in ROUNDS:
            for eps in PRIVACY:
                p = privacy_label(eps)
                names = [f'T{t}_{p}_{s}' for s in SEEDS]
                if not all(n in usable for n in names):
                    continue
                contrasts[f'auroc|T{t}|{p}'] = [{n: 1} for n in names]
                if t != 20 and all(f'T20_{p}_{s}' in usable for s in SEEDS):
                    contrasts[f'vsT20|T{t}|{p}'] = [{f'T{t}_{p}_{s}': 1, f'T20_{p}_{s}': -1} for s in SEEDS]
                if eps is not None and all(f'T{t}_clean_{s}' in usable for s in SEEDS):
                    contrasts[f'dpCost|T{t}|{p}'] = [{f'T{t}_clean_{s}': 1, f'T{t}_{p}_{s}': -1} for s in SEEDS]
        summary = {
            'schema': 'capstone.effusion.dp_rounds.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A15',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE,
            'calibration': {f'T{t}|eps{eps:g}': c for (t, eps), c in calibrations.items()},
            'runs': runs, 'unusableCells': sorted({r['name'] for r in runs} - usable),
            'replay': {'cells': replay, 'allExact': bool(replay) and all(replay.values())},
            'contrasts': linear_contrasts(y_selection, selection_patients, margins, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256, 'a8SummarySha256': args.a8_sha256,
                       'a13SummarySha256': args.a13_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_dp_rounds.py', 'run_effusion_gdp_recipe.py', 'run_effusion_federated.py',
                              'run_effusion_recipe.py')},
            'engineSourceSha256': federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / 'simulation' / 'engine.py'),
            'accountantSourceSha256': federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / 'privacy' / 'gdp_accountant.py'),
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--a8-dir', '--a13-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--a8-sha256', '--a13-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'unusable': summary['unusableCells'], 'replay': summary['replay']['allExact'],
                      'elapsedSeconds': summary['elapsedSeconds']}))
