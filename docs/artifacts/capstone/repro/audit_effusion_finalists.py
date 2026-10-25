"""Read-only finalized-run reconciliation; exports aggregate audit results only.

No bundle, patient, feature, prediction or model loader is used. This separately
checks execution artifacts and paired-seed inference, not the patient bootstrap.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from scipy.special import ndtr
from scipy.stats import t

import run_effusion_pilots as pilot
from fedlearn.simulation.config import config_hash
from finalist_design import cells


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path, digest):
    path = Path(path)
    _require(not path.is_symlink() and pilot.sha(path) == digest, 'artifact hash differs')
    return json.loads(path.read_text())


def _close(actual, expected, label, tolerance=1e-12):
    _require(type(actual) in (int, float) and math.isfinite(actual)
             and abs(actual - expected) <= tolerance, label + ' differs')
    return abs(actual - expected)


def _archive(directory):
    _require(directory.is_dir() and not directory.is_symlink(), 'source archive required')
    paths = list(directory.rglob('*'))
    _require(not any(p.is_symlink() for p in paths), 'source symbolic link refused')
    return {p.relative_to(directory).as_posix(): pilot.sha(p) for p in paths if p.is_file()}


def _specifications():
    methods = ('mean512', 'projected128', 'trim16', 'center_quarter')
    specs = {}
    def arm(method, noise, attack): return f'{method}:{noise}:{attack}'
    for noise in ('nonprivate', 'private'):
        for attack in ('clean', 'owned_pgd'):
            for method in methods[1:]:
                specs[f'alternative_minus_mean:{method}:{noise}:{attack}'] = [
                    (1, arm(method, noise, attack)), (-1, arm('mean512', noise, attack))]
        for method in methods:
            specs[f'attack_damage:{method}:{noise}'] = [
                (1, arm(method, noise, 'clean')), (-1, arm(method, noise, 'owned_pgd'))]
    for method in methods:
        for attack in ('clean', 'owned_pgd'):
            specs[f'privacy_cost:{method}:{attack}'] = [
                (1, arm(method, 'nonprivate', attack)), (-1, arm(method, 'private', attack))]
    for method in methods[1:]:
        specs[f'interaction:{method}'] = [(sign, arm(policy, noise, attack))
            for noise, direction in (('private', 1), ('nonprivate', -1))
            for policy, attack, sign in ((method, 'owned_pgd', direction),
                ('mean512', 'owned_pgd', -direction), (method, 'clean', -direction),
                ('mean512', 'clean', direction))]
    return specs


def _intervals(result):
    arms = {a['id']: a for a in result['arms']}
    expected_arms = {name for terms in _specifications().values() for _, name in terms}
    _require(len(result['arms']) == 16 and set(arms) == expected_arms, 'analysis arms differ')
    seeds = {str(s) for s in range(13001, 13013)}
    _require(all({str(s) for s in a['per_seed']} == seeds for a in arms.values()), 'analysis seed set differs')
    contrasts = {c['id']: c for c in result['contrasts']}
    _require(len(result['contrasts']) == 31 and set(contrasts) == set(_specifications()), 'contrast set differs')
    largest = 0.
    for name, terms in _specifications().items():
        contrast = contrasts[name]
        _require(contrast['status'] == 'complete' and contrast['n'] == 12 and
                 contrast['components'] == [{'coefficient': c, 'arm': a} for c, a in terms], 'contrast components differ')
        effects = {seed: math.fsum(c * arms[a]['per_seed'][seed]['auroc'] for c, a in terms)
                   for seed in sorted(seeds)}
        point = statistics.mean(effects.values())
        sd = statistics.stdev(effects.values())
        se = sd / math.sqrt(12)
        low, high = t.interval(.95, df=11, loc=point, scale=se) if se else (point, point)
        saved = contrast['seed_interval']
        _require(saved['n'] == 12 and saved['df'] == 11 and set(saved['seed_effects']) == seeds, 'interval seed set differs')
        for key, wanted in dict(point=point, sample_sd=sd, se=se, ci_low=low, ci_high=high).items():
            largest = max(largest, _close(saved[key], float(wanted), 'paired seed ' + key))
        for seed, effect in effects.items():
            largest = max(largest, _close(saved['seed_effects'][seed], effect, 'paired seed effect'))
    primary = contrasts['alternative_minus_mean:projected128:private:clean']['seed_interval']
    _require(result['primary']['seed_interval'] == primary, 'primary interval differs')
    return largest


def audit(run_dir, summary_sha256, analysis_dir, analysis_sha256):
    """Audit completed records without fitting or loading the underlying dataset."""
    run_dir, analysis_dir = Path(run_dir), Path(analysis_dir)
    summary = _read(run_dir / 'summary.json', summary_sha256)
    plan = _read(run_dir / 'frozen-plan.json', summary['plan_sha256'])
    public = _read(run_dir / 'public-manifest.json', plan['public_manifest_sha256'])
    _require(all(plan.get(k) == v for k, v in public.items()) and public['cells'] == cells()
             and public['settings'] == pilot.DEFAULTS and public['stage'] == 'finalists'
             and public['synthetic'] is False and public['bundle_report_sha256'] == pilot.BUNDLE_PIN,
             'frozen public design differs')
    pilot._checkpoint(plan, summary['checkpoint_commit'], pilot.ROOT / summary['checkpoint_manifest'])
    source = public['source_sha256']
    _require(_archive(run_dir / 'sources') == source, 'source archive differs')
    _require({n: pilot.sha(p) for n, p in pilot._paths().items()} == source, 'live source inventory differs')
    expected = {c['name']: c for c in cells()}
    rows = summary['runs']
    _require(summary['status'] == 'complete' and summary['stage'] == 'finalists' and summary['synthetic'] is False
             and len(rows) == 196 and {r['name'] for r in rows} == set(expected)
             and set(plan['configurations']) == set(expected), 'complete exact 196-cell run required')
    _require(public['configuration_sha256'] == {n: pilot._configuration_hash(c)
             for n, c in plan['configurations'].items()}, 'configuration inventory differs')
    rounds, pgd_rounds, private_cells, max_delta, max_z_difference = 0, 0, 0, 0., 0.
    calibrated_z = pilot.calibration(4., rounds=20, delta=1e-5)['gdpNoiseMultiplier']
    for row in rows:
        _require(row['status'] == 'completed' and all(row.get(k) == v for k, v in expected[row['name']].items()), 'cell identity differs')
        _require(row['record'] == f"runs/{row['name']}.json", 'raw record path differs')
        raw = _read(run_dir / row['record'], row['sha256'])
        config, meta = plan['configurations'][row['name']], raw['meta']
        seed = row['seed']
        seeds = dict(partition=seed, init=seed, client=seed, attack=seed + 1000, noise=seed + 2000)
        _require(meta['config'] == config and meta['config_hash'] == config_hash(config)
                 and meta['seeds'] == config['seeds'] == seeds and raw['pilot_metrics'] == row['metrics']
                 and not raw.get('pilot_failure') and meta['refusals'] == [], 'raw run identity differs')
        lines = raw['rounds']
        _require(len(lines) == 20 and [r['round'] for r in lines] == list(range(1, 21)), 'round sequence differs')
        z = meta['noise_multiplier']
        max_z_difference = max(max_z_difference, _close(z, calibrated_z if row['private'] else 0.,
                               'noise multiplier', calibrated_z * 1e-6 if row['private'] else 1e-12))
        sensitivity = 2 * (.0125 if row['method'] == 'center_quarter' else .05) / (48 if row['method'] == 'trim16' else 80)
        if row['private']:
            private_cells += 1
            mu = math.sqrt(20) / z
            delta = float(ndtr(mu / 2 - 4 / mu) - math.exp(4) * ndtr(-mu / 2 - 4 / mu))
            _require(0 <= delta <= 1e-5 + 1e-12, 'independent GDP normal-CDF bound failed')
            max_delta = max(max_delta, delta)
        for line in lines:
            rounds += 1
            p = line['participants']
            _require(not line.get('refusal') and all(p[k] == 80 for k in ('sampled', 'submitted', 'admitted', 'accepted', 'aggregated'))
                     and p['dropped'] == p['straggled'] == 0, 'round cohort differs')
            privacy = line['privacy']
            for key, wanted in dict(sensitivity=sensitivity, noise_std=z * sensitivity,
                                   server_noise_std=z * sensitivity, client_noise_std=0.).items():
                _close(privacy[key], wanted, 'round privacy ' + key)
            diagnostics = line.get('owned_attack_diagnostics', [])
            if row['attack'] == 'owned_pgd':
                _require(len(diagnostics) == 1 and all(diagnostics[0].get(k) == v for k, v in
                    dict(type='owned_pgd', coalition_size=16, total_submitted=80, coalition_fraction=.2,
                         steps=20, restarts=3, step_ratio=.25, gradient_evaluations=60, loss_evaluations=63).items()), 'PGD budget differs')
                pgd_rounds += 1
            else:
                _require(not diagnostics, 'unexpected clean-arm attack')
    result = _read(analysis_dir / 'summary.json', analysis_sha256)
    inputs = result['inputs']
    _require(result['status'] == 'complete' and inputs['summary_sha256'] == summary_sha256
             and inputs['plan_sha256'] == summary['plan_sha256'] and inputs['public_manifest_sha256'] == plan['public_manifest_sha256']
             and inputs['source_sha256'] == source and inputs['record_sha256'] == {r['name']: r['sha256'] for r in rows}, 'analysis provenance differs')
    arms = {a['id']: a for a in result['arms']}
    for row in rows:
        if row['role'] == 'main':
            arm = f"{row['method']}:{'private' if row['private'] else 'nonprivate'}:{row['attack']}"
            _require(arms[arm]['per_seed'][str(row['seed'])] == row['metrics'], 'analysis arm metric differs from raw summary')
    difference = _intervals(result)
    return {'schema': 'capstone.effusion_finalist_execution_audit.v1', 'status': 'verified',
            'counts': dict(cells=196, rounds=rounds, main_cells=192, private_cells=private_cells,
                           pgd_rounds=pgd_rounds, contrasts=31, source_files=len(source)),
            'maximum_seed_interval_difference': difference, 'maximum_delta_at_epsilon4': max_delta,
            'maximum_calibration_multiplier_difference': max_z_difference,
            'hashes': dict(summary_sha256=summary_sha256, analysis_sha256=analysis_sha256,
                           plan_sha256=summary['plan_sha256'], public_manifest_sha256=plan['public_manifest_sha256']),
            'scope': 'Execution/hash reconciliation and independent paired-seed t arithmetic; no private-data access, head rescoring or independent patient-bootstrap verification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'summary-sha256', 'analysis-dir', 'analysis-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('receipt output already exists')
    result = audit(args.run_dir, args.summary_sha256, args.analysis_dir, args.analysis_sha256)
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
