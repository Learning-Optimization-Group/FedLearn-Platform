"""Effusion protocol v1 amendment A4: central-DP sweep on R1 (primary) and R0 (width arm).

Same engine, recipe, clients and seeds as the section 6 baseline. Selection half only;
no holdout or official test. The privacy unit is one simulated client, not a patient.
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
EPSILONS = (1.0, 4.0, 8.0)
DELTA = 1e-5
ARMS = ('R1', 'R0')
CONTRAST_SEED = 20261024


def _label(epsilon):
    return 'clean' if epsilon is None else f'eps{epsilon:g}'


def cells():
    return [{'arm': arm, 'epsilon': eps, 'seed': seed, 'name': f'{arm}_{_label(eps)}_{seed}'}
            for arm in ARMS for eps in (None, *EPSILONS) for seed in SEEDS]


def contrasts(y, patients, margins, *, epsilons=EPSILONS, reps=2000, seed=CONTRAST_SEED):
    """DP cost (clean - DP, same arm and seed) and the width contrast cost(R1) - cost(R0).

    Both are averaged over seeds inside every patient-cluster bootstrap replicate, so the
    interval reflects patient resampling of the seed-averaged quantity.
    """
    from sklearn.metrics import roc_auc_score
    from fit_effusion_gate import cluster_bootstrap

    def costs(index):
        yy = y[index]
        out = {}
        for arm in ARMS:
            for eps in epsilons:
                diffs = [roc_auc_score(yy, margins[f'{arm}_clean_{s}'][index]) -
                         roc_auc_score(yy, margins[f'{arm}_{_label(eps)}_{s}'][index]) for s in SEEDS]
                out[(arm, eps)] = diffs
        return out

    full = costs(np.arange(len(y)))
    draws = {key: [] for key in full}
    width = {eps: [] for eps in epsilons}
    for index in cluster_bootstrap(patients, reps=reps, seed=seed):
        if y[index].min() == y[index].max():
            continue
        values = costs(index)
        for key, diffs in values.items():
            draws[key].append(float(np.mean(diffs)))
        for eps in epsilons:
            width[eps].append(float(np.mean(values[('R1', eps)]) - np.mean(values[('R0', eps)])))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {
        'cost': {arm: {str(eps): {'perSeed': [float(d) for d in full[(arm, eps)]], 'seedMean': float(np.mean(full[(arm, eps)])),
                                  'interval95': ci(draws[(arm, eps)])} for eps in epsilons} for arm in ARMS},
        'widthContrast': {str(eps): {'point': float(np.mean(full[('R1', eps)]) - np.mean(full[('R0', eps)])),
                                     'interval95': ci(width[eps])} for eps in epsilons},
        'bootstrap': {'reps': reps, 'seed': seed, 'method': 'patient-cluster percentile; seed-averaged within replicate'},
    }


def main(args):
    import effusion_labels
    import fit_effusion_gate
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    raw_gate = (Path(args.gate_dir) / 'gate.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.gate_sha256:
        raise Refusal('gate_hash_mismatch')
    authority = federated.check_gate(json.loads(raw_gate), args.amendment)
    raw_baseline = (Path(args.baseline_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_baseline).hexdigest() != args.baseline_sha256:
        raise Refusal('baseline_hash_mismatch')
    baseline = np.load(Path(args.baseline_dir) / 'selection-margins.npz')
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    heads = np.load(Path(args.gate_dir) / 'selected-heads.npz')
    standardisers = {}
    for arm in ARMS:
        mean, std = fit_effusion_gate.standardiser(features[arm][0])
        if not (np.array_equal(mean, heads[f'{arm}_mean']) and np.array_equal(std, heads[f'{arm}_std'])):
            raise Refusal(f'standardiser_mismatch:{arm}')
        standardisers[arm] = (mean, std)
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    train_patients = [r['patient_id'] for r in labels['train']['rows']]
    selection_patients = np.array([r['patient_id'] for r in labels['selection']['rows']])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.dp-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells():
            train_x, selection_x = features[cell['arm']]
            mean, std = standardisers[cell['arm']]
            privacy = None if cell['epsilon'] is None else {'mode': 'central', 'epsilon': cell['epsilon'], 'delta': DELTA}
            info = {'representation': cell['arm'], 'amendment': 'A4', 'gateSha256': args.gate_sha256}
            record, margin = federated.run_one(cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x,
                                               y_selection, mean, std, clients=federated.CLIENTS,
                                               features_path=args.xrv_cache if cell['arm'] == 'R1' else args.historical_cache,
                                               report_sha256=args.xrv_report_sha256 if cell['arm'] == 'R1' else fit_effusion_gate.HISTORICAL_REPORT_SHA256,
                                               info=info, privacy=privacy)
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            margins[cell['name']] = margin
            last = record['rounds'][-1]
            runs.append({**cell, 'auroc': last['auc'], 'finite': bool(np.isfinite(margin).all()),
                         'refusals': len(record['meta']['refusals']), 'parameters': record['meta']['workload']['parameters'],
                         'noiseMultiplier': record['meta']['noise_multiplier'], 'accountedEpsilon': record['meta']['accounted_epsilon'],
                         'noiseStd': last['privacy']['noise_std'], 'noiseToSignalLastRound': last['privacy']['noise_to_signal'],
                         'clipRateLastRound': last['privacy']['clip_rate']})
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals')}), flush=True)
        replay = {f'R1_clean_{s}': bool(np.array_equal(margins[f'R1_clean_{s}'], baseline[f'fed_{s}'])) for s in SEEDS}
        summary = {
            'schema': 'capstone.effusion.central_dp_sweep.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A4',
            'gateAuthority': authority, 'scope': 'Central DP (client-level, replace-one, q=1) on R1 and R0; selection half only.',
            'privacyUnit': 'one simulated client (~280 patients); not patient-level DP',
            'epsilons': list(EPSILONS), 'delta': DELTA, 'seeds': list(SEEDS), 'runs': runs,
            'replayOfSection6Baseline': replay,
            'contrasts': contrasts(y_selection, selection_patients, margins),
            'inputs': {'labelInputs': provenance, 'baselineSummarySha256': args.baseline_sha256,
                       'xrvCacheReportSha256': args.xrv_report_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_dp.py', 'run_effusion_federated.py', 'fit_effusion_gate.py', 'effusion_labels.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--baseline-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--baseline-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'replay': summary['replayOfSection6Baseline'], 'elapsedSeconds': summary['elapsedSeconds']}))
