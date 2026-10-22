"""Effusion protocol v1 amendment A13: exact Gaussian-DP noise calibration on R0.

Reruns the A8 confirmation DP cells (A8 recipe, five seeds, claimed eps 8/4/1) with z calibrated by
the exact GDP accountant instead of RDP, and compares each with its A8 counterpart. The simulation
engine is unchanged: the harness passes the RDP-equivalent budget whose RDP-calibrated z equals the
GDP z, then checks the z each run recorded. Selection half only; holdout untouched.
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
EPSILONS = (1.0, 4.0, 8.0)
DELTA = 1e-5
RECIPE = {'lr': 0.03, 'clip_norm': 0.05}  # A8
Z_TOLERANCE = 1e-6
CONTRAST_SEED = 20261032


def calibration(epsilon, rounds=federated.ROUNDS, delta=DELTA):
    """The exact-GDP z for a claimed epsilon, and the RDP budget the engine must be given to use it."""
    from fedlearn.privacy.dp_accountant import compute_rdp, get_epsilon
    from fedlearn.privacy.gdp_accountant import gdp_required_noise_multiplier
    z = float(gdp_required_noise_multiplier(epsilon, rounds, delta))
    return {'claimedEpsilon': epsilon, 'gdpNoiseMultiplier': z,
            'rdpEquivalentEpsilon': float(get_epsilon(compute_rdp(1.0, z, rounds), delta)[0])}


def check_calibration(cal, recorded_z, rounds=federated.ROUNDS, delta=DELTA):
    """Refuse a run whose recorded z is not the GDP z, or whose exact epsilon exceeds the claim."""
    from fedlearn.privacy.gdp_accountant import gdp_epsilon
    if abs(recorded_z - cal['gdpNoiseMultiplier']) > Z_TOLERANCE * cal['gdpNoiseMultiplier']:
        raise Refusal(f"noise_multiplier_mismatch:{recorded_z}!={cal['gdpNoiseMultiplier']}")
    exact = float(gdp_epsilon(recorded_z, rounds, delta))
    if exact > cal['claimedEpsilon'] + 1e-6:
        raise Refusal(f"exact_epsilon_exceeds_claim:{exact}>{cal['claimedEpsilon']}")
    return exact


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
    raw_a8 = (Path(args.a8_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a8).hexdigest() != args.a8_sha256:
        raise Refusal('a8_hash_mismatch')
    a8_summary = json.loads(raw_a8)
    if a8_summary['chosenRecipe'] != RECIPE:
        raise Refusal('a8_recipe_mismatch')
    a8 = np.load(Path(args.a8_dir) / 'selection-margins.npz')
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
    calibrations = {eps: calibration(eps) for eps in EPSILONS}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.gdp-recipe-', dir=output.parent))
    started = time.perf_counter()
    try:
        margins, runs = {}, []
        for seed in SEEDS:
            for eps in EPSILONS:
                cal = calibrations[eps]
                name = f'gdp_eps{eps:g}_{seed}'
                privacy = {'mode': 'central', 'epsilon': cal['rdpEquivalentEpsilon'], 'delta': DELTA}
                record, margin = federated.run_one(
                    name, seed, train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                    clients=federated.CLIENTS, info={'representation': 'R0', 'amendment': 'A13'},
                    privacy=privacy, recipe=RECIPE, **common)
                exact = check_calibration(cal, float(record['meta']['noise_multiplier']))
                (staging / f'{name}.json').write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
                finite = bool(np.isfinite(margin).all())
                last = record['rounds'][-1]
                runs.append({'name': name, 'seed': seed, 'claimedEpsilon': eps, 'finite': finite,
                             'refusals': len(record['meta']['refusals']),
                             'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                             'noiseMultiplier': record['meta']['noise_multiplier'], 'exactEpsilon': exact,
                             'rdpAccountedEpsilon': record['meta']['accounted_epsilon'],
                             'noiseStd': last['privacy']['noise_std'],
                             'clipRateLastRound': last['privacy']['clip_rate']})
                margins[name] = margin
                print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'auroc')}), flush=True)
        scores = {**margins, **{f'a8_{k}': a8[k] for k in a8.files}}
        contrasts = {}
        for eps in EPSILONS:
            e = f'{eps:g}'
            contrasts[f'gainVsA8_eps{e}'] = [{f'gdp_eps{e}_{s}': 1, f'a8_dp_eps{e}_{s}': -1} for s in SEEDS]
            contrasts[f'dpCost_eps{e}'] = [{f'a8_clean_{s}': 1, f'gdp_eps{e}_{s}': -1} for s in SEEDS]
        summary = {
            'schema': 'capstone.effusion.gdp_recipe.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A13',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE,
            'scope': 'Exact-GDP noise calibration of the A8 DP cells on the selection half (A13); holdout untouched.',
            'calibration': {f'{eps:g}': calibrations[eps] for eps in EPSILONS},
            'runs': runs,
            'seedMeans': {f'gdp_eps{eps:g}': float(np.mean([r['auroc'] for r in runs if r['claimedEpsilon'] == eps]))
                          for eps in EPSILONS},
            'a8SeedMeans': {k: v for k, v in a8_summary['seedMeans'].items()},
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'a8SummarySha256': args.a8_sha256, 'gateSha256': args.gate_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_gdp_recipe.py', 'run_effusion_recipe.py', 'run_effusion_federated.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--a8-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--a8-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'seedMeans': summary['seedMeans'], 'elapsedSeconds': summary['elapsedSeconds']}))
