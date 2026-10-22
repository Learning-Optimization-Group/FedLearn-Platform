"""Effusion protocol v1 amendment A17: zero-initialised head under central DP on R0.

T in {5, 20} x {clean, claimed eps 8, 4, 1} x 5 seeds with the A8 recipe and exact-GDP z, identical
to A13/A15 except that the initial global head is all zeros. Compared seed by seed with the
random-init cells of A15 (and A13 for T = 20, eps 1). Selection half only; holdout untouched.
"""
import argparse
from collections import OrderedDict
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
ROUNDS = (5, 20)
PRIVACY = (None, 8.0, 4.0, 1.0)
CONTRAST_SEED = 20261036


class ZeroHeadWorkload(federated.EffusionWorkload):
    """The Effusion workload with the initial global head set to zero (data-independent, no privacy cost).

    Only the seeded build, which the engine uses for the initial global, is zeroed; unseeded builds are
    containers the engine loads parameters into.
    """

    def build_model(self, init_seed=None):
        model = super().build_model(init_seed)
        if init_seed is not None:
            import torch
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
        return model


def privacy_label(epsilon):
    return 'clean' if epsilon is None else f'eps{epsilon:g}'


def cells():
    return [{'rounds': t, 'epsilon': eps, 'seed': s, 'name': f'zero_T{t}_{privacy_label(eps)}_{s}'}
            for t in ROUNDS for eps in PRIVACY for s in SEEDS]


def counterpart(cell):
    """The random-init margin to compare with: (study, name), or None when none was run."""
    p, t, s = privacy_label(cell['epsilon']), cell['rounds'], cell['seed']
    if t == 20 and cell['epsilon'] == 1.0:
        return 'a13', f'gdp_eps1_{s}'
    if cell['epsilon'] == 1.0:
        return None
    return 'a15', f'T{t}_{p}_{s}'


def run_zero(name, seed, train_x, train_y, patients, selection_x, selection_y, mean, std, *, features_path,
             report_sha256, rounds, privacy):
    """federated.run_one with ZeroHeadWorkload; returns (record, float64 selection margins)."""
    import torch
    cfg = federated.config(name, seed, clients=federated.CLIENTS, features_path=features_path,
                           report_sha256=report_sha256, rounds=rounds, privacy=privacy, recipe=RECIPE)
    workload = ZeroHeadWorkload(train_x, train_y, patients, selection_x, selection_y, mean, std,
                                clients=federated.CLIENTS, partition_seed=cfg['seeds']['partition'],
                                info={'representation': 'R0', 'amendment': 'A17', 'headInit': 'zero'})
    final = {}
    record = federated.run(cfg, workload=workload, on_final_parameters=final.update)
    model = workload.build_model()
    federated.apply_trainable_subset(model, OrderedDict(final))
    model.eval()
    with torch.no_grad():
        logits = model(workload.test_x).to(torch.float64)
    record['meta']['finalHead'] = {k: v.tolist() for k, v in final.items()}
    return record, (logits[:, 1] - logits[:, 0]).numpy()


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
    earlier = {'a13': _pinned(args.a13_dir, args.a13_sha256, 'a13'), 'a15': _pinned(args.a15_dir, args.a15_sha256, 'a15')}
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
    staging = Path(tempfile.mkdtemp(prefix='.zero-head-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in cells():
            cal = calibrations.get((cell['rounds'], cell['epsilon']))
            privacy = None if cal is None else {'mode': 'central', 'epsilon': cal['rdpEquivalentEpsilon'], 'delta': DELTA}
            record, margin = run_zero(cell['name'], cell['seed'], train_x, y_train, train_patients, selection_x,
                                      y_selection, mean, std, features_path=args.historical_cache,
                                      report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256,
                                      rounds=cell['rounds'], privacy=privacy)
            exact = (None if cal is None else
                     check_calibration(cal, float(record['meta']['noise_multiplier']), rounds=cell['rounds']))
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            head = {k: np.asarray(v) for k, v in record['meta']['finalHead'].items()}
            runs.append({**cell, 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         'noiseMultiplier': record['meta']['noise_multiplier'], 'exactEpsilon': exact,
                         'finalWeightNorm': float(np.linalg.norm(head['head.weight'])),
                         'finalBiasNorm': float(np.linalg.norm(head['head.bias'])),
                         'clipRateLastRound': record['rounds'][-1]['privacy']['clip_rate']})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1][k] for k in ('name', 'finite', 'refusals', 'auroc')}), flush=True)
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0}
        scores = dict(margins)
        contrasts = {}
        for t in ROUNDS:
            for eps in PRIVACY:
                p = privacy_label(eps)
                group = [c for c in cells() if c['rounds'] == t and c['epsilon'] == eps]
                if not all(c['name'] in usable for c in group):
                    continue
                contrasts[f'auroc|T{t}|{p}'] = [{c['name']: 1} for c in group]
                pairs = [(c['name'], counterpart(c)) for c in group]
                if all(src is not None for _, src in pairs):
                    for name, (study, other) in pairs:
                        scores[f'{study}_{other}'] = earlier[study][other]
                    contrasts[f'zeroMinusRandom|T{t}|{p}'] = [{n: 1, f'{st}_{o}': -1} for n, (st, o) in pairs]
                if eps is not None:
                    contrasts[f'dpCost|T{t}|{p}'] = [{f'zero_T{t}_clean_{s}': 1, f'zero_T{t}_{p}_{s}': -1} for s in SEEDS]
        summary = {
            'schema': 'capstone.effusion.zero_head.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A17',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE, 'headInit': 'zero',
            'calibration': {f'T{t}|eps{eps:g}': c for (t, eps), c in calibrations.items()},
            'runs': runs, 'unusableCells': sorted({r['name'] for r in runs} - usable),
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrasts, seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256, 'a13SummarySha256': args.a13_sha256,
                       'a15SummarySha256': args.a15_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_zero_head.py', 'run_effusion_gdp_recipe.py', 'run_effusion_federated.py',
                              'run_effusion_recipe.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--a13-dir', '--a15-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--a13-sha256', '--a15-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'unusable': summary['unusableCells'], 'elapsedSeconds': summary['elapsedSeconds']}))
