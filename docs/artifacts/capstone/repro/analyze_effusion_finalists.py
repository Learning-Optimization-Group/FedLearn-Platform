"""Rescore pinned finalist heads and export prespecified aggregate-only analyses.

No fitting occurs here. The only data loader is the verified train/selection
bundle loader. Patient identities, features, heads and scores remain in memory.
"""
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np
from scipy.special import ndtr
import torch
from torch.nn.functional import linear

import run_effusion_pilots as pilot
from fedlearn.simulation.behaviors import BehaviorPlan
from fedlearn.simulation.config import config_hash
from fedlearn.simulation.partition import patient_iid_partition
from analyze_effusion_pilots import METRICS, _read, _valid_metrics
from finalist_design import METHODS, SEEDS, cells
from finalist_inference import conditional_patient_interval, paired_seed_interval, weighted_auc


def _stats(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def _arm_id(key):
    method, private, attack = key
    return f"{method}:{'private' if private else 'nonprivate'}:{attack}"


def _ordered_rows(rows):
    declared = {c['name']: c for c in cells()}
    observed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('row must be an object')
        name = row.get('name')
        if name not in declared or name in observed or any(row.get(k) != v for k, v in declared[name].items()):
            raise ValueError('duplicate or changed declared finalist identity')
        if row.get('status') not in ('completed', 'error', 'not_run'):
            raise ValueError('unknown finalist outcome')
        if row['status'] == 'completed' and not _valid_metrics(row.get('metrics')):
            raise ValueError('invalid completed metrics')
        if row.get('metrics') is not None and not _valid_metrics(row['metrics']):
            raise ValueError('invalid available metrics')
        observed[name] = row
    return [observed.get(c['name'], c | {'status': 'missing', 'metrics': None}) for c in cells()]


def summarize(rows):
    """Summarize exact declared identities; any failure suppresses all inference."""
    rows = _ordered_rows(rows)
    complete = all(r['status'] == 'completed' for r in rows)
    grouped = {}
    for row in rows:
        if row['role'] == 'main':
            grouped.setdefault((row['method'], row['private'], row['attack']), []).append(row)
    arms, lookup = [], {}
    for key, members in grouped.items():
        if sorted(r['seed'] for r in members) != list(SEEDS):
            raise ValueError('main arm must contain the twelve exact declared seeds')
        available = {r['seed']: r['metrics'] for r in members if r['status'] == 'completed'}
        arm = {'id': _arm_id(key), 'method': key[0], 'private': key[1], 'attack': key[2],
               'n': len(available), 'planned_n': len(SEEDS),
               'status': 'complete' if len(available) == len(SEEDS) else 'incomplete',
               'per_seed': available,
               'seed_status': {r['seed']: r['status'] for r in members},
               'metrics': {m: _stats([v[m] for v in available.values()]) for m in METRICS}}
        arms.append(arm)
        lookup[key] = available
    contrasts = []

    def add(name, terms, interpretation, primary=False):
        effects = ({s: sum(sign * lookup[key][s]['auroc'] for sign, key in terms) for s in SEEDS}
                   if complete else {})
        contrasts.append({'id': name, 'primary': primary, 'interpretation': interpretation,
                          'components': [{'coefficient': sign, 'arm': _arm_id(key)} for sign, key in terms],
                          'status': 'complete' if complete else 'incomplete',
                          'n': len(effects), 'planned_n': len(SEEDS),
                          'seed_interval': paired_seed_interval(effects) if complete else None,
                          'multiplicity': 'single prespecified primary' if primary else
                          'descriptive unadjusted secondary; no family-wide discovery'})

    for private in (False, True):
        noise = 'private' if private else 'nonprivate'
        for attack in ('clean', 'owned_pgd'):
            for method in METHODS[1:]:
                primary = method == 'projected128' and private and attack == 'clean'
                add(f'alternative_minus_mean:{method}:{noise}:{attack}',
                    [(1, (method, private, attack)), (-1, ('mean512', private, attack))],
                    'alternative minus matched mean AUROC', primary)
        for method in METHODS:
            add(f'attack_damage:{method}:{noise}',
                [(1, (method, private, 'clean')), (-1, (method, private, 'owned_pgd'))],
                'clean minus attacked AUROC; positive means attack damage')
    for method in METHODS:
        for attack in ('clean', 'owned_pgd'):
            add(f'privacy_cost:{method}:{attack}',
                [(1, (method, False, attack)), (-1, (method, True, attack))],
                'zero-noise minus private AUROC; positive means privacy cost')
    for method in METHODS[1:]:
        terms = []
        for private, privacy_sign in ((True, 1), (False, -1)):
            terms.extend([(privacy_sign, (method, private, 'owned_pgd')),
                          (-privacy_sign, ('mean512', private, 'owned_pgd')),
                          (-privacy_sign, (method, private, 'clean')),
                          (privacy_sign, ('mean512', private, 'clean'))])
        add(f'interaction:{method}', terms,
            'private attack-adjusted alternative advantage minus zero-noise advantage; '
            'absolute clean and attacked components are in arms')
    primary = None
    if complete:
        contrast = next(c for c in contrasts if c['primary'])
        interval = contrast['seed_interval']
        lower = interval['ci_low']
        primary = {'contrast': contrast['id'], 'seed_interval': interval, 'patient_interval': None,
                   'technical_margin': .01, 'technical_success': lower > .01,
                   'conclusion': ('practically_positive_seed_mean_gain' if lower > .01 else
                                  'positive_effect_only' if lower > 0 else
                                  'negative_seed_mean_gain' if interval['ci_high'] < 0 else 'unresolved'),
                   'clinical_claim': None}
    outcomes = [{k: r[k] for k in ('name', 'method', 'role', 'seed', 'private', 'attack', 'status', 'metrics')}
                for r in rows]
    return {'schema': 'capstone.effusion_finalist_analysis.v1',
            'status': 'complete' if complete else 'incomplete',
            'accounting': {'planned': len(rows), 'main_planned': 192, 'anchors_planned': 4,
                           'completed': sum(r['status'] == 'completed' for r in rows)},
            'outcomes': outcomes,
            'failures': [{k: r[k] for k in ('name', 'status')} for r in rows if r['status'] != 'completed'],
            'arms': arms, 'contrasts': contrasts, 'primary': primary,
            'scope': 'Frozen selected policies on repeatedly used development-selection cohort; '
                     'separate conditional training-seed and realized-head patient intervals; '
                     'no joint uncertainty, untouched-cohort confirmation, clinical claim, '
                     'family-wide discovery or whole-study DP claim.',
            'metric_direction': {m: 'lower' if m == 'brier' else 'higher' for m in METRICS}}


@lru_cache(maxsize=1)
def _projection():
    return pilot.public_projection(512, 128, seed=20261025)


def _head_tensors(head, width):
    if not isinstance(head, dict) or set(head) != {'head.weight', 'head.bias'}:
        raise ValueError('saved head parameter keys differ')
    try:
        weight = torch.tensor(head['head.weight'], dtype=torch.float32)
        bias = torch.tensor(head['head.bias'], dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('invalid saved head arrays') from error
    if weight.shape != (2, width) or bias.shape != (2,) or not all(
            bool(torch.isfinite(v).all()) for v in (weight, bias)):
        raise ValueError('saved head shape or finiteness differs')
    return weight, bias


def saved_head_margins(features, head, width):
    """Use the runner's float32 linear operation and float64 logit difference."""
    if width not in (128, 512):
        raise ValueError('unknown finalist width')
    x = torch.as_tensor(features)
    if x.ndim != 2 or x.shape[1] != width or x.dtype != torch.float32 or not torch.isfinite(x).all():
        raise ValueError('invalid scoring features')
    weight, bias = _head_tensors(head, width)
    with torch.no_grad():
        logits = linear(x, weight, bias).double()
    return (logits[:, 1] - logits[:, 0]).numpy()


def _close(actual, expected, label, tolerance=1e-10):
    if (isinstance(actual, bool) or not isinstance(actual, (int, float)) or
            not math.isfinite(actual) or abs(actual - expected) > tolerance):
        raise ValueError(label + ' differs')


def _noise_audit(row, raw):
    sensitivity = (2 * .0125 / 80 if row['method'] == 'center_quarter' else
                   2 * .05 / 48 if row['method'] == 'trim16' else 2 * .05 / 80)
    z = raw['meta'].get('noise_multiplier')
    expected_z = _calibration()['gdpNoiseMultiplier'] if row['private'] else 0.
    _close(z, expected_z, 'noise multiplier', max(1e-12, expected_z * 1e-6))
    for round_record in raw['rounds']:
        privacy = round_record.get('privacy', {})
        _close(privacy.get('sensitivity'), sensitivity, 'sensitivity', 1e-12)
        for key in ('noise_std', 'server_noise_std'):
            _close(privacy.get(key), z * sensitivity, 'server noise', 1e-12)
        _close(privacy.get('client_noise_std'), 0., 'client noise', 1e-12)
    result = {'sensitivity': sensitivity, 'noise_multiplier': z, 'noise_std': z * sensitivity,
              'releases': 20}
    if row['private']:
        accounting = raw.get('pilot_accounting', {})
        if accounting.get('releases') != 20:
            raise ValueError('private release accounting differs')
        for key, value in _calibration().items():
            _close(accounting.get(key), value, 'saved GDP calibration', 1e-8)
        exact = pilot.check_calibration(_calibration(), z, 20, 1e-5)
        _close(accounting.get('exact_epsilon'), exact, 'saved exact epsilon', 1e-8)
        mu = math.sqrt(20) / z
        delta = float(ndtr(-4 / mu + mu / 2) - math.exp(4) * ndtr(-4 / mu - mu / 2))
        if delta > 1e-5 + 1e-12 or delta < 0:
            raise ValueError('independent normal-CDF GDP check failed')
        result.update(gdp_mu=mu, delta_at_epsilon4=delta, exact_epsilon=exact)
    return result


@lru_cache(maxsize=1)
def _calibration():
    return pilot.calibration(4., rounds=20, delta=1e-5)


def _load_run(directory, summary_sha256):
    summary = _read(directory / 'summary.json', summary_sha256)
    if (summary.get('schema') != 'capstone.effusion_pilot_result.v1' or
            summary.get('stage') != 'finalists' or summary.get('synthetic') is not False or
            summary.get('status') not in ('complete', 'incomplete')):
        raise ValueError('declared real finalist summary required')
    plan = _read(directory / 'frozen-plan.json', summary.get('plan_sha256'))
    public = _read(directory / 'public-manifest.json', plan.get('public_manifest_sha256'))
    if (any(plan.get(k) != v for k, v in public.items()) or
            public.get('schema') != 'capstone.effusion_pilot_plan.v1' or
            public.get('stage') != 'finalists' or public.get('synthetic') is not False or
            public.get('cells') != cells() or public.get('settings') != pilot.DEFAULTS or
            public.get('bundle_report_sha256') != pilot.BUNDLE_PIN or
            summary.get('reference') != plan.get('reference')):
        raise ValueError('finalist frozen/public plan differs from declared design')
    pilot.validate_design(plan['cells'], stage='finalists', reference=plan.get('reference'))
    source = public.get('source_sha256')
    if not isinstance(source, dict) or not source or any(
            not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in source.values()):
        raise ValueError('source inventory is missing or invalid')
    manifest = summary.get('checkpoint_manifest')
    if not isinstance(manifest, str):
        raise ValueError('checkpoint manifest binding required')
    pilot._checkpoint(plan, summary.get('checkpoint_commit'), pilot.ROOT / manifest)
    configs = plan.get('configurations', {})
    if (set(configs) != {c['name'] for c in cells()} or
            public.get('configuration_sha256') != {n: pilot._configuration_hash(v) for n, v in configs.items()}):
        raise ValueError('frozen configuration hashes differ')
    bundle_path = Path(plan['bundle_path'])
    expected_configs = {c['name']: pilot.configuration(c, plan['settings'], bundle_path, pilot.BUNDLE_PIN)
                        for c in cells()}
    if configs != expected_configs:
        raise ValueError('resolved finalist configurations differ')
    archive = directory / 'sources'
    archive_status = 'not present; committed source binding verified'
    if archive.exists():
        if archive.is_symlink() or any(p.is_symlink() for p in archive.rglob('*')):
            raise ValueError('source archive symbolic link refused')
        actual = {p.relative_to(archive).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in archive.rglob('*') if p.is_file()}
        if actual != source:
            raise ValueError('source archive hashes differ')
        archive_status = 'verified'
    if not isinstance(summary.get('runs'), list):
        raise ValueError('finalist rows must be a list')
    rows = _ordered_rows(summary['runs'])
    complete = all(r['status'] == 'completed' for r in rows)
    if (summary['status'] == 'complete') != complete:
        raise ValueError('summary completion differs from all declared outcomes')
    records = {}
    if (directory / 'runs').is_symlink():
        raise ValueError('run directory symbolic link refused')
    for row in rows:
        if row['status'] == 'missing':
            continue
        relative = f"runs/{row['name']}.json"
        if row.get('record') != relative:
            raise ValueError('undeclared raw record location')
        records[row['name']] = _read(directory / relative, row.get('sha256'))
    return summary, plan, rows, records, archive_status


@lru_cache(maxsize=2)
def _projection_sha256(width):
    matrix = _projection() if width == 128 else pilot.public_projection(512, width, seed=20261025)
    return hashlib.sha256(matrix.numpy().tobytes()).hexdigest()


def _workload_identity(row, config, bundle):
    """Reconstruct deterministic workload metadata without creating a model."""
    train, evaluation, report = bundle['train'], bundle['evaluation'], bundle['report']
    parts = patient_iid_partition(train['patient_ids'], 80, config['seeds']['partition'])
    clients, assignments = {}, []
    for cid, part in enumerate(parts):
        assignments.append({'client': cid, 'images': [train['image_ids'][i] for i in part]})
        positive = int(train['y'][part].sum())
        clients[str(cid)] = {'images': len(part), 'positive_images': positive,
                             'negative_images': len(part) - positive,
                             'patients': len({train['patient_ids'][i] for i in part})}
    width, workload = row['width'], config['workload']
    return {'feat_dim': width, 'n_classes': 2, 'n_train': len(train['y']), 'n_test': len(evaluation['y']),
            'parameters': 2 * width + 2,
            'auc_score': workload['auc_score'], 'auc_score_definition': 'float64(logit_1)-float64(logit_0)',
            'dataset': workload['dataset'], 'source_dataset': report['dataset'], 'target': report['target'],
            'bundle_version': report['bundle_version'], 'backbone': 'resnet18 ImageNet (R0), train-standardised',
            'training_device': 'cpu', 'torch_num_threads': 1, 'pretrained': report['pretrained'],
            'allow_random_features': workload['allow_random_features'],
            'evaluation_split': workload['evaluation_split'],
            'evaluation': 'development selection half; no holdout or official test',
            'features_report_sha256': bundle['report_sha256'], 'bundle_provenance': report['provenance'],
            'split_counts': report['counts'], 'clients': clients, 'partition_algorithm': 'patient-iid-v1',
            'assignment_sha256': pilot._configuration_hash(assignments),
            'assignment_version': 'capstone-pilot-client-assignment-v1',
            'pilot': {'alpha': row['alpha'], 'width': width, 'projection_seed': 20261025,
                      'projection_sha256': _projection_sha256(width),
                      'partition_override': 'actual injected patient-whole partition is bound by cell and assignment hash'}}


def _validate_completed(row, raw, config, expected_workload):
    meta = raw.get('meta', {})
    if meta.get('config') != config or meta.get('config_hash') != config_hash(config):
        raise ValueError('raw config identity differs from exact resolved cell')
    if meta.get('seeds') != config['seeds']:
        raise ValueError('raw seed streams differ from exact resolved cell')
    if meta.get('workload') != expected_workload:
        raise ValueError('raw workload identity differs from pinned bundle and exact resolved cell')
    assignment = BehaviorPlan(config['behaviors'], 80, config['seeds']['attack'], 2).assignment()
    if meta.get('behavior_assignment') != assignment:
        raise ValueError('raw behavior assignment differs from exact resolved cell')
    if raw.get('pilot_failure') or meta.get('refusals') != []:
        raise ValueError('completed run has failures/refusals')
    if raw.get('pilot_metrics') != row['metrics']:
        raise ValueError('raw and summary metrics differ')
    rounds = raw.get('rounds', [])
    if len(rounds) != 20 or [r.get('round') for r in rounds] != list(range(1, 21)):
        raise ValueError('incomplete round sequence')
    for item in rounds:
        participants = item.get('participants', {})
        if (item.get('refusal') or any(participants.get(k) != 80 for k in
                ('sampled', 'submitted', 'admitted', 'accepted', 'aggregated')) or
                any(participants.get(k) != 0 for k in ('dropped', 'straggled'))):
            raise ValueError('failed round or changed 80-client cohort')
    _close(rounds[-1].get('auc'), row['metrics']['auroc'], 'engine AUROC')
    return _noise_audit(row, raw)


def _backend_audit(summary, rows, records):
    comparisons = summary.get('backend_comparisons', [])
    expected = {(seed, private) for seed in SEEDS[:2] for private in (False, True)}
    seen, results = set(), []
    for item in comparisons:
        key = (item.get('seed'), item.get('private'))
        if key not in expected or key in seen:
            raise ValueError('duplicate or unknown backend comparison')
        seen.add(key)
        selected = [r for r in rows if r['method'] == 'mean512' and r['attack'] == 'clean'
                    and (r['seed'], r['private']) == key]
        if len(selected) != 2 or any('finalHead' not in records.get(r['name'], {}).get('meta', {})
                                     for r in selected):
            raise ValueError('backend comparison references incomplete anchors')
        pair_complete = all(r['status'] == 'completed' for r in selected)
        heads = [_head_tensors(records[r['name']]['meta']['finalHead'], 512) for r in selected]
        difference = max(float(torch.max(torch.abs(a-b))) for a, b in zip(*heads))
        _close(item.get('max_parameter_difference'), difference, 'backend reported difference', 1e-12)
        if difference > 1e-6 and pair_complete:
            raise ValueError('native/ByzFL head difference exceeds 1e-6')
        results.append({'seed': key[0], 'private': key[1], 'max_parameter_difference': difference,
                        'status': 'passed' if pair_complete and difference <= 1e-6 else 'failed'})
    if summary['status'] == 'complete' and seen != expected:
        raise ValueError('all four backend comparisons required')
    return results


def analyze(run_dir, summary_sha256, output, *, bootstrap_draws=2000, bootstrap_seed=20261025):
    """Verify every declared record and rescore saved heads; never train models."""
    if (type(bootstrap_draws) is not int or bootstrap_draws != 2000 or
            type(bootstrap_seed) is not int or bootstrap_seed != 20261025):
        raise ValueError('frozen finalist bootstrap requires 2000 draws and seed 20261025')
    if Path(output).exists():
        raise FileExistsError('analysis output already exists')
    summary, plan, rows, records, archive_status = _load_run(Path(run_dir), summary_sha256)
    bundle = pilot.load_effusion_bundle(plan['bundle_path'], expected_report_sha256=pilot.BUNDLE_PIN)
    pilot._validate_bundle_classification(bundle, pilot.BUNDLE_PIN, False)
    evaluation = bundle['evaluation']
    features = {512: torch.as_tensor(evaluation['x'])}
    features[128] = (features[512].double() @ _projection()).float()
    labels = evaluation['y']
    scores, audit_rows, workload_identities = {}, [], {}
    for row in rows:
        raw = records.get(row['name'])
        audit = {'name': row['name'], 'status': row['status'], 'record_sha256': row.get('sha256')}
        if raw is not None and 'finalHead' in raw.get('meta', {}):
            _head_tensors(raw['meta']['finalHead'], row['width'])
        if row['status'] == 'completed':
            config = plan['configurations'][row['name']]
            key = (row['seed'], row['width'])
            if key not in workload_identities:
                workload_identities[key] = _workload_identity(row, config, bundle)
            audit.update(_validate_completed(row, raw, config, workload_identities[key]))
            margins = saved_head_margins(features[row['width']], raw['meta']['finalHead'], row['width'])
            metrics = pilot._metrics(labels, margins)
            for metric in METRICS:
                _close(row['metrics'][metric], metrics[metric], 'recomputed ' + metric)
            _close(row['metrics']['auroc'], weighted_auc(labels, margins), 'independent weighted AUROC')
            scores[row['name']] = margins
            audit.update(scoring='all five saved-head metrics and independent AUROC verified',
                         identity='resolved raw config, config hash, seed streams, workload and behavior assignment verified',
                         head_sha256=hashlib.sha256(json.dumps(raw['meta']['finalHead'], sort_keys=True,
                                                             allow_nan=False).encode()).hexdigest())
        audit_rows.append(audit)
    parity = _backend_audit(summary, rows, records)
    result = summarize(rows)
    if result['status'] == 'complete':
        left, right = [], []
        for seed in SEEDS:
            for method, target in (('projected128', left), ('mean512', right)):
                row = next(r for r in rows if r['role'] == 'main' and r['method'] == method and
                           r['private'] and r['attack'] == 'clean' and r['seed'] == seed)
                target.append(scores[row['name']])
        interval = conditional_patient_interval(labels, evaluation['patient_ids'], np.stack(left), np.stack(right),
                                                draws=bootstrap_draws, seed=bootstrap_seed, max_attempts=20000)
        _close(interval['point'], result['primary']['seed_interval']['point'], 'patient and seed point')
        result['primary']['patient_interval'] = interval
    result['inputs'] = {'summary_sha256': summary_sha256, 'plan_sha256': summary['plan_sha256'],
                        'public_manifest_sha256': plan['public_manifest_sha256'],
                        'bundle_report_sha256': pilot.BUNDLE_PIN,
                        'checkpoint_commit': summary.get('checkpoint_commit'),
                        'source_sha256': plan['source_sha256'],
                        'record_sha256': {r['name']: r['sha256'] for r in rows if 'sha256' in r}}
    audit = {'schema': 'capstone.effusion_finalist_scoring_audit.v1', 'runs': audit_rows,
             'source_archive': archive_status, 'backend_comparisons': parity,
             'scoring': 'float32 linear logits; float64 class-one minus class-zero; cached fixed public projection',
             'bootstrap': {'draws': bootstrap_draws, 'seed': bootstrap_seed, 'max_attempts': 20000},
             'scope': 'aggregate research diagnostics; not covered by released-model privacy guarantee'}
    write_outputs(output, result, audit)
    return result


def _csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True, allow_nan=False) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()} for row in rows)


def _report(result):
    lines = ['# Frozen Effusion finalist comparison', '', f"Status: {result['status']}.", '', result['scope'], '',
             'AUROC gains are left minus right. Native anchors are excluded from all main-arm means.', '',
             '## Primary', '']
    primary = result['primary']
    if primary is None:
        lines.append('Incomplete execution: no primary or partial-pair inference is reported.')
    else:
        interval = primary['seed_interval']
        lines += [f"Projected128 minus mean512, private clean: {interval['point']:.8f} AUROC; "
                  f"approximate paired-seed 95% t interval [{interval['ci_low']:.8f}, {interval['ci_high']:.8f}].",
                  f"Technical margin: +0.01; conclusion: {primary['conclusion']}. No clinical-benefit claim.", '',
                  f"Paired sample SD: {interval['sample_sd']:.8f}; standard error: {interval['se']:.8f}; "
                  f"degrees of freedom: {interval['df']}.", '',
                  'The seed interval assumes independent seed effects and an adequate t approximation. '
                  'Twelve seeds do not certify those assumptions.', '',
                  '| Seed | Paired AUROC gain |', '| --- | ---: |']
        lines += [f'| {seed} | {effect:.8f} |' for seed, effect in interval['seed_effects'].items()]
        patient = primary['patient_interval']
        if patient is not None:
            lines += ['', f"Separate conditional patient interval: [{patient['ci_low']:.8f}, {patient['ci_high']:.8f}]; "
                      f"{patient['draws']} valid draws and {patient['invalid_draws']} skipped single-class draws.",
                      patient['scope']]
    lines += ['', '## Main arm AUROC', '', '| Arm | Completed seeds | Mean | Range | Sample SD |',
              '| --- | ---: | ---: | --- | ---: |']
    for arm in result['arms']:
        s = arm['metrics']['auroc']
        lines.append(f"| {arm['id']} | {arm['n']} | {s['mean']} | {s['min']} to {s['max']} | {s['sample_sd']} |")
    lines += ['', 'All five metrics and every per-seed value are in summary.json and arms.csv. '
              'Secondary paired AUROC intervals in contrasts.csv are descriptive and unadjusted; '
              'the interaction must be interpreted alongside absolute clean/attacked components.', '',
              '## Failure accounting', '']
    lines += [f"- {r['name']}: {r['status']}" for r in result['failures']] or ['All 196 declared outcomes completed.']
    return '\n'.join(lines) + '\n'


def write_outputs(output, result, audit):
    """Write only the explicit aggregate schemas into a fresh directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('summary.json', result), ('scoring-audit.json', audit)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    _csv(output / 'arms.csv', result['arms'])
    _csv(output / 'contrasts.csv', result['contrasts'])
    (output / 'report.md').write_text(_report(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.summary_sha256, args.output)
    print(json.dumps({'status': result['status'], 'accounting': result['accounting']}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
