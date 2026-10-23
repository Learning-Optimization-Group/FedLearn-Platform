"""Effusion protocol v1 amendment A23: distributed DP (Skellam over coded LightSecAgg) on R0.

A17's T = 20 zero-head configuration with only the privacy mechanism changed:
- distributed Skellam at b in {12, 14, 16, 20} bits x claimed eps 8/4/1 (b 12 at eps 1 is infeasible);
- central Gaussian with RDP-calibrated z, the matched-accountant control;
- clip-skipping sign-flip attackers (4 and 16 of 80) at eps 4, distributed b 16 vs central RDP.
Five seeds each; compared seed by seed with each other and with A17's exact-GDP cells. Selection
half only; holdout untouched.
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
from run_effusion_gdp_recipe import DELTA, RECIPE  # noqa: E402
from run_effusion_recipe import linear_contrasts  # noqa: E402
from run_effusion_zero_head import ZeroHeadWorkload, privacy_label  # noqa: E402

Refusal = federated.Refusal
SEEDS = federated.SEEDS
ROUNDS = 20
BITS = (12, 14, 16, 20)
PRIVACY = (8.0, 4.0, 1.0)
INFEASIBLE = {(12, 1.0)}
ATTACK_BITS, ATTACK_EPSILON, ATTACKERS = 16, 4.0, (4, 16)
ATTACK = {'type': 'sign_flip_scale', 'scale': -10.0}
CONTRAST_SEED = 20261044


def distributed_privacy(epsilon, bits):
    return {'mode': 'distributed', 'epsilon': epsilon, 'delta': DELTA, 'distributed': {'bits': bits}}


def central_privacy(epsilon):
    return {'mode': 'central', 'epsilon': epsilon, 'delta': DELTA}


def cells():
    """Every pre-registered cell, and the cells recorded as not run (with the reason)."""
    out, not_run = [], []
    for b in BITS:
        for eps in PRIVACY:
            for s in SEEDS:
                name = f'dist_b{b}_{privacy_label(eps)}_{s}'
                if (b, eps) in INFEASIBLE:
                    not_run.append({'name': name, 'reason': f'{b} bits cannot hold the eps {eps:g} aggregate at any scale'})
                else:
                    out.append({'name': name, 'seed': s, 'privacy': distributed_privacy(eps, b), 'behaviors': []})
    for eps in PRIVACY:
        for s in SEEDS:
            out.append({'name': f'rdp_{privacy_label(eps)}_{s}', 'seed': s, 'privacy': central_privacy(eps), 'behaviors': []})
    for m in ATTACKERS:
        for s in SEEDS:
            attack = [{**ATTACK, 'fraction': m / federated.CLIENTS}]
            p = privacy_label(ATTACK_EPSILON)
            out.append({'name': f'dist_b{ATTACK_BITS}_{p}_signflip{m}_{s}', 'seed': s,
                        'privacy': distributed_privacy(ATTACK_EPSILON, ATTACK_BITS), 'behaviors': attack})
            out.append({'name': f'rdp_{p}_signflip{m}_{s}', 'seed': s, 'privacy': central_privacy(ATTACK_EPSILON),
                        'behaviors': attack})
    return out, not_run


def contrast_spec(usable):
    """Pre-registered contrasts over the usable cells (a group with any unusable cell is skipped)."""
    spec = {}

    def add(name, terms):
        if all(n in usable or n.startswith('a17_') for t in terms for n in t):
            spec[name] = terms

    for eps in PRIVACY:
        p = privacy_label(eps)
        add(f'auroc|rdp|{p}', [{f'rdp_{p}_{s}': 1} for s in SEEDS])
        add(f'rdpMinusGdp|{p}', [{f'rdp_{p}_{s}': 1, f'a17_zero_T20_{p}_{s}': -1} for s in SEEDS])
        for b in BITS:
            if (b, eps) in INFEASIBLE:
                continue
            add(f'auroc|b{b}|{p}', [{f'dist_b{b}_{p}_{s}': 1} for s in SEEDS])
            add(f'distMinusRdp|b{b}|{p}', [{f'dist_b{b}_{p}_{s}': 1, f'rdp_{p}_{s}': -1} for s in SEEDS])
            add(f'distMinusGdp|b{b}|{p}', [{f'dist_b{b}_{p}_{s}': 1, f'a17_zero_T20_{p}_{s}': -1} for s in SEEDS])
    p, b = privacy_label(ATTACK_EPSILON), ATTACK_BITS
    for m in ATTACKERS:
        add(f'auroc|b{b}|{p}|signflip{m}', [{f'dist_b{b}_{p}_signflip{m}_{s}': 1} for s in SEEDS])
        add(f'auroc|rdp|{p}|signflip{m}', [{f'rdp_{p}_signflip{m}_{s}': 1} for s in SEEDS])
        add(f'attackDamage|b{b}|signflip{m}',
            [{f'dist_b{b}_{p}_{s}': 1, f'dist_b{b}_{p}_signflip{m}_{s}': -1} for s in SEEDS])
        add(f'attackDamage|rdp|signflip{m}', [{f'rdp_{p}_{s}': 1, f'rdp_{p}_signflip{m}_{s}': -1} for s in SEEDS])
        add(f'damageDistMinusRdp|signflip{m}',
            [{f'dist_b{b}_{p}_{s}': 1, f'dist_b{b}_{p}_signflip{m}_{s}': -1, f'rdp_{p}_{s}': -1,
              f'rdp_{p}_signflip{m}_{s}': 1} for s in SEEDS])
    return spec


def run_cell(cell, train_x, train_y, patients, selection_x, selection_y, mean, std, *, features_path, report_sha256):
    """One engine run with the zero-initialised head; returns (record, float64 selection margins)."""
    import torch
    cfg = federated.config(cell['name'], cell['seed'], clients=federated.CLIENTS, features_path=features_path,
                           report_sha256=report_sha256, rounds=ROUNDS, privacy=cell['privacy'], recipe=RECIPE,
                           behaviors=cell['behaviors'])
    workload = ZeroHeadWorkload(train_x, train_y, patients, selection_x, selection_y, mean, std,
                                clients=federated.CLIENTS, partition_seed=cfg['seeds']['partition'],
                                info={'representation': 'R0', 'amendment': 'A23', 'headInit': 'zero'})
    final = {}
    record = federated.run(cfg, workload=workload, on_final_parameters=final.update)
    model = workload.build_model()
    federated.apply_trainable_subset(model, OrderedDict(final))
    model.eval()
    with torch.no_grad():
        logits = model(workload.test_x).to(torch.float64)
    record['meta']['finalHead'] = {k: v.tolist() for k, v in final.items()}
    return record, (logits[:, 1] - logits[:, 0]).numpy()


def integrity(record, cell):
    """What a run must satisfy: exact secure sums and an accounted epsilon within the claim."""
    from fedlearn.privacy.dp_accountant import required_noise_multiplier
    meta, claim = record['meta'], float(cell['privacy']['epsilon'])
    out = {'accountedEpsilon': meta['accounted_epsilon'], 'withinClaim': meta['accounted_epsilon'] <= claim + 1e-6}
    if cell['privacy']['mode'] == 'distributed':
        lines = record['rounds']
        out.update(secureSumExact=all(l['secagg']['exact'] for l in lines),
                   wrappedCoordinates=[l['secagg']['wrapped_coordinates'] for l in lines],
                   maxRoundingTries=max(l['secagg']['max_rounding_tries'] for l in lines),
                   plan={k: meta['distributed_dp'][k] for k in ('bits', 'prime', 'scale', 'mu', 'delta2', 'order',
                                                                'sum_noise_std', 'd', 'd_pad')},
                   effectiveNoiseMultiplier=meta['distributed_dp']['sum_noise_std'] / (2 * RECIPE['clip_norm']),
                   bytesPerClient=lines[0]['secagg']['bytes_per_client'],
                   bytesPerClientUncoded=lines[0]['secagg']['bytes_per_client_uncoded'])
    else:
        expected = required_noise_multiplier(claim, 1.0, ROUNDS, DELTA)
        out.update(noiseMultiplier=meta['noise_multiplier'],
                   noiseMultiplierMatches=abs(meta['noise_multiplier'] - expected) <= 1e-9 * expected)
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
    raw_a17 = (Path(args.a17_dir) / 'summary.json').read_bytes()
    if hashlib.sha256(raw_a17).hexdigest() != args.a17_sha256:
        raise Refusal('a17_hash_mismatch')
    a17 = np.load(Path(args.a17_dir) / 'selection-margins.npz')
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
    todo, not_run = cells()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.distributed-dp-', dir=output.parent))
    started = time.perf_counter()
    margins, runs = {}, []
    try:
        for cell in todo:
            record, margin = run_cell(cell, train_x, y_train, train_patients, selection_x, y_selection, mean, std,
                                      features_path=args.historical_cache,
                                      report_sha256=fit_effusion_gate.HISTORICAL_REPORT_SHA256)
            (staging / f"{cell['name']}.json").write_text(json.dumps(record, sort_keys=True, default=str) + '\n')
            finite = bool(np.isfinite(margin).all())
            runs.append({'name': cell['name'], 'seed': cell['seed'], 'privacy': cell['privacy'],
                         'behaviors': cell['behaviors'], 'finite': finite, 'refusals': len(record['meta']['refusals']),
                         'auroc': float(roc_auc_score(y_selection, margin)) if finite else None,
                         **integrity(record, cell)})
            if finite:
                margins[cell['name']] = margin
            print(json.dumps({k: runs[-1].get(k) for k in ('name', 'finite', 'refusals', 'auroc', 'secureSumExact')}),
                  flush=True)
        failures = [r['name'] for r in runs if not r['withinClaim'] or r.get('secureSumExact') is False
                    or r.get('noiseMultiplierMatches') is False]
        usable = {r['name'] for r in runs if r['finite'] and r['refusals'] == 0} - set(failures)
        scores = {**margins, **{f'a17_{k}': a17[k] for k in a17.files}}
        summary = {
            'schema': 'capstone.effusion.distributed_dp.v1', 'protocol': 'effusion-baseline-protocol-v1 amendment A23',
            'gateAuthority': authority, 'representation': 'R0 (A6)', 'recipe': RECIPE, 'headInit': 'zero',
            'rounds': ROUNDS, 'delta': DELTA, 'runs': runs, 'notRun': not_run, 'integrityFailures': failures,
            'unusableCells': sorted({r['name'] for r in runs} - usable),
            'plainFloat32BytesPerClient': next((4 * r['plan']['d'] for r in runs if 'plan' in r), None),
            'contrasts': linear_contrasts(y_selection, selection_patients, scores, contrast_spec(usable),
                                          seed=CONTRAST_SEED),
            'inputs': {'labelInputs': provenance, 'gateSha256': args.gate_sha256, 'a17SummarySha256': args.a17_sha256},
            'sourceSha256': {name: federated._sha256_file(HERE / name) for name in
                             ('run_effusion_distributed_dp.py', 'run_effusion_zero_head.py', 'run_effusion_gdp_recipe.py',
                              'run_effusion_federated.py', 'run_effusion_recipe.py')},
            'frameworkSourceSha256': {name: federated._sha256_file(federated.FRAMEWORK_SRC / 'fedlearn' / name) for name in
                                      ('simulation/engine.py', 'privacy/distributed_skellam.py',
                                       'security/lightsecagg_coded.py')},
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
    for flag in ('--historical-cache', '--lock-dir', '--xrv-cache', '--gate-dir', '--a17-dir', '--output'):
        parser.add_argument(flag, type=Path, required=True)
    for flag in ('--xrv-report-sha256', '--gate-sha256', '--a17-sha256'):
        parser.add_argument(flag, required=True)
    parser.add_argument('--amendment', choices=sorted(federated.FAILED_GATE_AMENDMENTS))
    args = parser.parse_args()
    try:
        summary = main(args)
    except Refusal as error:
        print(json.dumps({'status': 'refused', 'reason': str(error)}))
        raise SystemExit(2)
    print(json.dumps({'status': 'complete', 'integrityFailures': summary['integrityFailures'],
                      'unusable': summary['unusableCells'], 'elapsedSeconds': summary['elapsedSeconds']}))
