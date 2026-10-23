"""Effusion protocol v1 amendment A22, conditional DP follow-up: central DP on each encoder that passed the gate.

Clean and claimed eps 8/4/1 with the best R0 configuration (A8 recipe, zero-initialised head, exact-GDP z,
T = 20), five seeds, each compared seed by seed with A17's R0 zero_T20 cells. Refuses to run unless the
pinned gate summary records at least one passing representation. Selection half only; holdout untouched.
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
from run_effusion_encoder_gate import WIDTHS, load_cache  # noqa: E402
from run_effusion_gdp_recipe import DELTA, calibration, check_calibration  # noqa: E402
from run_effusion_recipe import linear_contrasts  # noqa: E402
from run_effusion_zero_head import privacy_label, run_zero  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
PRIVACY = (None, 8.0, 4.0, 1.0)
CONTRAST_SEED = 20261043


def passing(gate_summary):
    return sorted(name for name, r in gate_summary['results'].items() if r.get('passed'))


def cells(representations):
    return [{'representation': rep, 'epsilon': eps, 'seed': s, 'name': f'{rep}_T20_{privacy_label(eps)}_{s}',
             'r0': f'zero_T20_{privacy_label(eps)}_{s}'}
            for rep in representations for eps in PRIVACY for s in SEEDS]


def main(args):
    import effusion_labels
    import fit_effusion_gate
    from sklearn.metrics import roc_auc_score
    output = Path(args.output)
    if output.exists():
        raise Refusal('output_exists')
    warnings.filterwarnings('ignore', message=r'.* encountered in matmul', category=RuntimeWarning)
    raw_gate = (Path(args.encoder_gate_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_gate).hexdigest() != args.encoder_gate_sha256:
        raise Refusal('encoder_gate_hash_mismatch')
    gate_summary = json.loads(raw_gate)
    representations = passing(gate_summary)
    if not representations:
        raise Refusal('no_representation_passed')
    raw_a17 = (Path(args.a17_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a17).hexdigest() != args.a17_sha256:
        raise Refusal('a17_hash_mismatch')
    a17 = np.load(Path(args.a17_dir) / 'selection-margins.npz')
    labels, provenance = effusion_labels.load_labels(args.historical_cache, args.lock_dir)
    features, _ = fit_effusion_gate._load_representations(args.historical_cache, args.xrv_cache, args.xrv_report_sha256, labels)
    caches = {'R2': (args.r2_cache, args.r2_report_sha256), 'R3': (args.r3_cache, args.r3_report_sha256)}
    for rep in representations:
        if rep in caches:
            features[rep], _ = load_cache(*caches[rep], labels, WIDTHS[rep])
    y_train, y_selection = labels['train']['y'], labels['selection']['y']
    train_patients = [r['patient_id'] for r in labels['train']['rows']]
    selection_patients = np.array([r['patient_id'] for r in labels['selection']['rows']])
    calibrations = {eps: calibration(eps) for eps in PRIVACY if eps is not None}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.encoder-dp-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells(representations):
            train_x, selection_x = features[cell['representation']]
            mean, std = fit_effusion_gate.standardiser(train_x)
            cal = calibrations.get(cell['epsilon'])
            privacy = None if cal is None else {'mode': 'central', 'epsilon': cal['rdpEquivalentEpsilon'], 'delta': DELTA}
            record, margin = run_zero(cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x, y_selection,
                                      mean, std, features_path=args.historical_cache,
                                      report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256, rounds=20, privacy=privacy)
            exact = None if cal is None else check_calibration(cal, float(record['meta']['noise_multiplier']))
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None, 'exactEpsilon': exact,
                         'width': int(train_x.shape[1]),
                         'noiseToSignalLastRound': record['rounds'][-1]['privacy']['noise_to_signal']})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'auroc')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        scores = {**margins, **{f'a17_{k}': a17[k] for k in a17.files}}
        contrasts = {}
        for rep in representations:
            for eps in PRIVACY:
                group = [c for c in cells([rep]) if c['epsilon'] == eps]
                if not all(c['name'] in usable for c in group):
                    continue
                p = privacy_label(eps)
                contrasts[f'auroc|{rep}|{p}'] = [{c['name']: 1} for c in group]
                contrasts[f'minusR0|{rep}|{p}'] = [{c['name']: 1, f"a17_{c['r0']}": -1} for c in group]
                if eps is not None:
                    contrasts[f'dpCost|{rep}|{p}'] = [{f'{rep}_T20_clean_{c["seed"]}': 1, c['name']: -1} for c in group]
                    contrasts[f'dpCostMinusR0dpCost|{rep}|{p}'] = [
                        {f'{rep}_T20_clean_{c["seed"]}': 1, c['name']: -1, f'a17_zero_T20_clean_{c["seed"]}': -1,
                         f"a17_{c['r0']}": 1} for c in group]
        summary = {
            'schema': 'capstone.effusion.encoder_dp.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A22 (DP follow-up)',
            'representations': representations, 'runs': runs, 'unusableCells': sorted({r['name'] for r in runs} - usable),
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'encoderGateSha256': args.encoder_gate_sha256, 'a17SummarySha256': args.a17_sha256},
            'sourceSha256': {n: federated._sha256_file(HERE / n) for n in
                             ('run_effusion_encoder_dp.py', 'run_effusion_zero_head.py', 'run_effusion_encoder_gate.py',
                              'run_effusion_gdp_recipe.py', 'run_effusion_federated.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--r2-cache', '--r3-cache', '--encoder-gate-dir',
                 '--a17-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--r2-report-sha256', '--r3-report-sha256', '--encoder-gate-sha256', '--a17-sha256'):
        parser.add_argument(flag, required=True)
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'representations': summary['representations'],
                      'elapsedSeconds': summary['elapsedSeconds']}))
