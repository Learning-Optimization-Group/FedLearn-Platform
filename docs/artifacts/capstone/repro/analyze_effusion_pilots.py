"""Verify pinned diagnostic pilots and export descriptive, aggregate-only results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from run_effusion_pilots import BUNDLE_PIN, DEFAULTS, RECIPES, _configuration_hash, bucket_cells, screen_cells

METRICS = ('auroc', 'average_precision', 'brier', 'sensitivity_at_half', 'specificity_at_half')
SEEDS = (20261211, 20261212)


def _read(path, expected):
    path = Path(path)
    if path.is_symlink() or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('invalid input hash or symbolic link')
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('input hash differs')
    return json.loads(payload)


def _stats(values):
    values = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('nonfinite aggregate input')
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None}


def _series(values):
    return {**_stats(values), 'missing': sum(v is None for v in values),
            'first': values[0] if values else None, 'last': values[-1] if values else None}


def _valid_metrics(value):
    return isinstance(value, dict) and set(value) == set(METRICS) and all(
        isinstance(value[k], (int, float)) and not isinstance(value[k], bool)
        and math.isfinite(value[k]) and 0 <= value[k] <= 1 for k in METRICS)


def _load_stage(directory, pin, stage, screen_pin=None):
    directory = Path(directory)
    summary = _read(directory/'summary.json', pin)
    if summary.get('schema') != 'capstone.effusion_pilot_result.v1' or summary.get('stage') != stage \
            or summary.get('synthetic') is not False or summary.get('status') not in ('complete', 'incomplete'):
        raise ValueError('declared real pilot summary required')
    plan = _read(directory/'frozen-plan.json', summary.get('plan_sha256'))
    public = _read(directory/'public-manifest.json', plan.get('public_manifest_sha256'))
    if any(plan.get(k) != v for k, v in public.items()) or public.get('schema') != 'capstone.effusion_pilot_plan.v1' \
            or public.get('synthetic') is not False or public.get('stage') != stage \
            or public.get('settings') != DEFAULTS or public.get('bundle_report_sha256') != BUNDLE_PIN:
        raise ValueError('declared real pilot plan required')
    reference = summary.get('reference')
    if plan.get('reference') != reference:
        raise ValueError('summary reference differs from frozen plan')
    if stage == 'buckets':
        if not reference or reference.get('summary_sha256') != screen_pin or reference.get('selected_recipe') not in RECIPES:
            raise ValueError('bucket stage requires the explicit pinned screen reference')
        expected = bucket_cells(reference['selected_recipe'])
    else:
        if reference is not None:
            raise ValueError('screen may not have a bucket reference')
        expected = screen_cells()
    if public.get('cells') != expected:
        raise ValueError('nondeclared pilot cells')
    configs = plan.get('configurations', {})
    if set(configs) != {c['name'] for c in expected} or public.get('configuration_sha256') != {
            name: _configuration_hash(config) for name, config in configs.items()}:
        raise ValueError('frozen configuration hashes differ')
    rows = summary.get('runs')
    if not isinstance(rows, list):
        raise ValueError('pilot rows must be a list')
    lookup = {}
    declared = {c['name']: c for c in expected}
    records = {}
    for row in rows:
        name = row.get('name')
        if name in lookup or name not in declared or any(row.get(k) != v for k, v in declared[name].items()):
            raise ValueError('duplicate or nondeclared pilot row')
        if row.get('status') not in ('completed', 'error', 'not_run'):
            raise ValueError('unknown pilot status')
        relative = f'runs/{name}.json'
        if row.get('record') != relative or (directory/'runs').is_symlink():
            raise ValueError('nondeclared run record location')
        raw = _read(directory/relative, row.get('sha256'))
        if row['status'] == 'completed':
            if not _valid_metrics(row.get('metrics')) or raw.get('pilot_metrics') != row['metrics']:
                raise ValueError('completed saved metrics differ or are invalid')
            if raw.get('pilot_failure'):
                raise ValueError('completed record contains a failure')
            if row['kind'] == 'fl':
                rounds = raw.get('rounds', [])
                if len(rounds) != DEFAULTS['rounds'] or [r.get('round') for r in rounds] != list(range(1, DEFAULTS['rounds']+1)):
                    raise ValueError('completed record has incomplete round sequence')
                if not math.isclose(rounds[-1].get('auc', -1), row['metrics']['auroc'], rel_tol=0, abs_tol=1e-10):
                    raise ValueError('completed saved and engine AUROC metrics differ')
        lookup[name] = row
        records[name] = raw
    observed_complete = len(lookup) == len(expected) and all(r['status'] == 'completed' for r in rows)
    if (summary['status'] == 'complete') != observed_complete:
        raise ValueError('summary status differs from declared cell outcomes')
    ordered = [lookup.get(c['name'], c | {'status': 'missing', 'metrics': None}) | {'stage': stage} for c in expected]
    return {'rows': ordered, 'records': records, 'summary': summary, 'plan': plan,
            'identity': {'summary_sha256': pin, 'plan_sha256': summary['plan_sha256'],
                         'public_manifest_sha256': plan['public_manifest_sha256'],
                         'checkpoint_commit': summary.get('checkpoint_commit'),
                         'record_sha256': {r['name']: r['sha256'] for r in rows}}}


def _arm_key(row):
    return row['tag'], row['backend'], row['private']


def _arm_id(key):
    return ':'.join((key[0], key[1], 'private' if key[2] else 'nonprivate'))


def _arms(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(_arm_key(row), []).append(row)
    arms = []
    for key, members in grouped.items():
        successful = [r for r in members if r['status'] == 'completed']
        arm = {k: members[0][k] for k in ('tag', 'backend', 'private', 'kind', 'rule', 'attack', 'fraction', 'width',
                'bucket', 'alpha', 'tau', 'clip_norm', 'lr', 'regularization')}
        arm.update(id=_arm_id(key), n=len(successful), planned_n=len(members),
                   status='complete' if len(successful) == 2 else 'incomplete',
                   reused_from_screen=any(r.get('reused_from_screen', False) for r in members),
                   per_seed={str(r['seed']): r['metrics'] for r in successful},
                   seed_status={str(r['seed']): r['status'] for r in members},
                   metrics={k: _stats([r['metrics'][k] for r in successful]) for k in METRICS})
        arms.append(arm)
    return arms


def _contrasts(arms):
    lookup = {(a['tag'], a['backend'], a['private']): a for a in arms}
    contrasts = []

    def add(label, left, right, meaning):
        if left not in lookup or right not in lookup:
            return
        lhs, rhs = lookup[left], lookup[right]
        seeds = [str(seed) for seed in SEEDS if str(seed) in lhs['per_seed'] and str(seed) in rhs['per_seed']]
        per_seed = {s: {k: lhs['per_seed'][s][k]-rhs['per_seed'][s][k] for k in METRICS} for s in seeds}
        contrasts.append({'id': label, 'left': lhs['id'], 'right': rhs['id'], 'operation': 'left_minus_right',
                          'interpretation': meaning, 'n': len(seeds), 'planned_n': 2,
                          'status': 'complete' if len(seeds) == 2 else 'incomplete', 'per_seed': per_seed,
                          'metrics': {k: _stats([v[k] for v in per_seed.values()]) for k in METRICS}})

    for private in (False, True):
        state = 'private' if private else 'nonprivate'
        mean = ('mean', 'byzfl', private)
        mean_pgd = ('mean_owned_pgd_f16', 'byzfl', private)
        add(f'backend_parity:{state}', ('mean', 'native', private), mean, 'native minus ByzFL mean')
        for tag in ('center_r0.25', 'center_r1', 'center_r4', 'trim16'):
            clean, attack = (f'{tag}_clean', 'byzfl', private), (f'{tag}_owned_pgd', 'byzfl', private)
            add(f'mechanism_vs_mean:{tag}:clean:{state}', clean, mean, 'mechanism minus matched clean mean')
            add(f'mechanism_vs_mean:{tag}:pgd:{state}', attack, mean_pgd, 'mechanism minus matched attacked mean')
            add(f'attack_damage:{tag}:byzfl:{state}', clean, attack, 'clean minus attacked; positive AUROC means damage')
        for attack in ('sign_flip_scale', 'owned_sign_flip', 'owned_pgd'):
            tag = f'mean_{attack}_f16'
            add(f'attack_damage:{tag}:byzfl:{state}', mean, (tag, 'byzfl', private),
                'clean minus attacked; positive AUROC means damage')
        add(f'pgd_vs_fullnorm:{state}', ('mean_owned_sign_flip_f16', 'byzfl', private), mean_pgd,
            'full-norm sign-flip AUROC minus PGD AUROC; PGD is not guaranteed stronger')
        for width in (64, 128, 256):
            add(f'projection_vs_512:k{width}:{state}', (f'mean_k{width}', 'byzfl', private), mean,
                'public projected width minus 512; report clean utility cost alongside private utility')
        for regularization in (.01, .1, 1.):
            add(f'erm_vs_mean:l{regularization:g}:{state}', (f'erm_l{regularization:g}', 'scalar', private), mean,
                'contextual scalar ERM comparator: row bound and parameterization differ; trusted coordinator')
    for arm in arms:
        if arm['private']:
            continue
        tag, backend = arm['tag'], arm['backend']
        add(f'privacy_cost:{tag}:{backend}', (tag, backend, False), (tag, backend, True),
            'clean mechanism minus its private match; positive AUROC is privacy cost')
    for width in (64, 128, 256):
        base = {c['id']: c for c in contrasts}
        projected, full = base[f'privacy_cost:mean_k{width}:byzfl'], base['privacy_cost:mean:byzfl']
        seeds = [s for s in projected['per_seed'] if s in full['per_seed']]
        per_seed = {s: {k: projected['per_seed'][s][k]-full['per_seed'][s][k] for k in METRICS} for s in seeds}
        contrasts.append({'id': f'additional_privacy_cost:k{width}', 'left': projected['id'], 'right': full['id'],
                          'operation': 'difference_of_paired_costs', 'interpretation': 'projection privacy cost minus width-512 privacy cost',
                          'n': len(seeds), 'planned_n': 2, 'status': 'complete' if len(seeds) == 2 else 'incomplete',
                          'per_seed': per_seed, 'metrics': {k: _stats([v[k] for v in per_seed.values()]) for k in METRICS}})
    for arm in arms:
        if arm['tag'].startswith('median_b') and arm['attack'] != 'clean':
            partition = 'iid' if arm['alpha'] is None else 'a0p1'
            clean = (f"median_b{arm['bucket']}_{partition}_clean", 'byzfl', False)
            add(f"bucket_attack_damage:{arm['id']}", clean, (arm['tag'], arm['backend'], False),
                'matched clean minus attacked; Fang is a distinct nonprivate raw-honest oracle stress test')
    return contrasts


def _diagnostics(row, raw):
    rounds = raw.get('rounds', [])
    diagnostic = {k: row[k] for k in ('name', 'tag', 'seed', 'private', 'status', 'stage')}
    diagnostic.update(rounds_observed=len(rounds), oracle_nonprivate=row['attack'] == 'fang_trim',
                      scope='research-only diagnostics outside released-model DP claims')
    if row['kind'] == 'erm':
        diagnostic['comparator'] = 'scalar row-bound-1 regularized logistic ERM; trusted coordinator; one final release'
        accounting = raw.get('accounting', {})
        diagnostic['privacy_accounting'] = {k: accounting.get(k) for k in ('noise_multiplier', 'noise_std', 'sensitivity', 'exact_epsilon', 'releases')}
        return diagnostic
    aggregation = [r.get('aggregation_diagnostics', {}) for r in rounds]
    privacy = [r.get('privacy', {}) for r in rounds]
    diagnostic['outer_clip_rate'] = _series([p.get('clip_rate') for p in privacy])
    diagnostic['noise_std'] = _series([p.get('noise_std') for p in privacy])
    if row['rule'] == 'centered_clip':
        diagnostic['center'] = {k: _series([d.get(k) for d in aggregation]) for k in (
            'inner_clip_rate', 'center_norm', 'center_alignment_with_input_mean', 'mean_input_norm')}
    if row['attack'] in ('owned_pgd', 'owned_sign_flip'):
        owned = [next((d for d in r.get('owned_attack_diagnostics', []) if d.get('type') == row['attack']), {}) for r in rounds]
        expected_loss, expected_grad = (63, 60) if row['attack'] == 'owned_pgd' else (0, 0)
        diagnostic['owned_attack'] = {k: _series([d.get(k) for d in owned]) for k in (
            'loss_evaluations', 'gradient_evaluations', 'initial_norm', 'intended_upload_norm', 'upload_norm')}
        diagnostic['owned_attack'].update(expected_loss_evaluations=expected_loss,
            expected_gradient_evaluations=expected_grad,
            budget_matches_expected=bool(rounds) and all(d.get('loss_evaluations') == expected_loss
                and d.get('gradient_evaluations') == expected_grad for d in owned),
            norm_note='initial_norm is the intended initialization; intended_upload_norm is the final intended norm when recorded; '
                      'upload_norm is represented/pre-server norm. Missing final intended norms are not reconstructed. '
                      'Mandatory server outer clipping bounds reducer inputs; no certified floating-point privacy claim.')
    if row['stage'] == 'buckets':
        attackers = int(round(80*row['fraction']))
        size = row['bucket']
        diagnostic['buckets'] = {'size': size, 'attackers': attackers,
            'expected_contaminated_fraction': 1-math.comb(80-attackers, size)/math.comb(80, size),
            'actual_contaminated_fraction': _series([d.get('contaminated_bucket_fraction') for d in aggregation]),
            'interpretation': 'uniform random grouping expectation; contamination alone does not imply reducer failure'}
    accounting = raw.get('pilot_accounting', {})
    diagnostic['privacy_accounting'] = {k: accounting.get(k) for k in ('gdpNoiseMultiplier', 'exact_epsilon', 'releases')}
    return diagnostic


def _csv(path, items):
    fields = ('id', 'status', 'n', 'metric', 'mean', 'min', 'max', 'sample_sd', 'per_seed')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            for metric in METRICS:
                writer.writerow({**{k: item[k] for k in ('id', 'status', 'n')}, 'metric': metric,
                    **{k: item['metrics'][metric][k] for k in ('mean', 'min', 'max', 'sample_sd')},
                    'per_seed': json.dumps({s: v[metric] for s, v in item['per_seed'].items()}, sort_keys=True)})


def _report(result):
    lines = ['# Effusion diagnostic pilot analysis', '', f"Status: {result['status']}.", '',
             'These are exploratory development-selection results from two paired seeds. Means, observed ranges and sample standard deviations are descriptive. '
             'No confidence intervals, significance, equivalence or clinical/external-validity claims follow from this screen. '
             'The probability-0.5 operating point is fixed and descriptive. No candidate selection is performed here.', '',
             'Contrasts use the stated left-minus-right direction for every metric. Higher AUROC/AP and sensitivity/specificity are better; lower Brier is better. '
             'Attack damage and privacy cost are clean-minus-attacked and clean-minus-private respectively, so positive AUROC means damage/cost.', '',
             'Scalar ERM is contextual: scalar versus two-logit geometry, row-norm bound, regularization and trusted-coordinator assumptions differ. '
             'Fang arms are distinct nonprivate raw-honest oracle stress tests. Diagnostics are research-only and do not extend released-model DP claims.', '',
             'The frozen runner retains initial_norm (intended initialization) and upload_norm (represented final pre-server norm). '
             'It does not separately log the final intended PGD norm. The mandatory server outer clip enforces the reducer input bound; '
             'a represented pre-server norm slightly above the radius is not classified here as a privacy violation.', '',
             '## Accounting', '', '| Stage | Planned | Completed | Failed or unrun |', '| --- | ---: | ---: | ---: |']
    for stage, accounting in result['accounting'].items():
        lines.append(f"| {stage} | {accounting['planned']} | {accounting['completed']} | {accounting['planned']-accounting['completed']} |")
    lines += ['', 'Every seed and all five metrics appear in summary.json and the CSV exports. No incomplete arm is ranked.', '',
              '## Nonprivate mean attack qualification', '',
              'Qualification requires two completed paired seeds and mean clean-to-attack AUROC loss at least 0.02. '
              'Failed qualification does not establish defense robustness. PGD is compared separately with full-norm sign flip and is not guaranteed stronger.', '',
              '| Attack | Paired seeds | Mean AUROC loss | Qualified |', '| --- | ---: | ---: | --- |']
    for item in result['attack_qualification']:
        lines.append(f"| {item['attack']} | {item['n']} | {item['mean_auroc_loss']} | {item['qualified']} |")
    lines += ['', '## Arm AUROC', '', '| Arm | Seeds | Mean | Min | Max | Sample SD |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in result['arms']:
        s = arm['metrics']['auroc']
        lines.append(f"| {arm['id']} | {arm['n']} | {s['mean']} | {s['min']} | {s['max']} | {s['sample_sd']} |")
    lines += ['', '## Failure accounting', '']
    lines += [f"- {r['stage']}: {r['name']} — {r['status']}" for r in result['failures']] or ['No failed or unrun cells.']
    return '\n'.join(lines)+'\n'


def analyze(screen_dir, screen_sha256, output, *, bucket_dir=None, bucket_sha256=None):
    """Read only pinned local results; write a fresh directory of public aggregates."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('analysis output already exists')
    if (bucket_dir is None) != (bucket_sha256 is None):
        raise ValueError('bucket directory and summary hash must be supplied together')
    screen = _load_stage(screen_dir, screen_sha256, 'screen')
    stages = {'screen': screen}
    if bucket_dir is not None:
        bucket = _load_stage(bucket_dir, bucket_sha256, 'buckets', screen_sha256)
        if any(r['status'] != 'completed' for r in screen['rows']):
            raise ValueError('bucket references require a complete pinned screen')
        reference = bucket['summary']['reference']
        if reference.get('plan_sha256') != screen['summary']['plan_sha256']:
            raise ValueError('bucket reference screen plan hash differs')
        stages['buckets'] = bucket
    rows = [r for stage in stages.values() for r in stage['rows']]
    accounting = {name: {'planned': len(stage['rows']), 'completed': sum(r['status'] == 'completed' for r in stage['rows'])}
                  for name, stage in stages.items()}
    if 'buckets' in stages:
        recipe = stages['buckets']['summary']['reference']['selected_recipe']
        for row in screen['rows']:
            if row['tag'] == f'median_recipe_{recipe}':
                rows.append(row | {'tag': 'median_b1_iid_clean', 'reused_from_screen': True})
        accounting['buckets']['reused_screen_references'] = 2
    arms = _arms(rows)
    contrasts = _contrasts(arms)
    failures = [{k: row[k] for k in ('stage', 'name', 'status')} for row in rows if row['status'] != 'completed']
    qualification = []
    for attack in ('sign_flip_scale', 'owned_sign_flip', 'owned_pgd'):
        contrast = next(c for c in contrasts if c['id'] == f'attack_damage:mean_{attack}_f16:byzfl:nonprivate')
        loss = contrast['metrics']['auroc']['mean']
        qualification.append({'attack': attack, 'n': contrast['n'], 'mean_auroc_loss': loss,
                              'per_seed_auroc_loss': {s: v['auroc'] for s, v in contrast['per_seed'].items()},
                              'qualified': contrast['n'] == 2 and loss >= .02, 'threshold': .02})
    result = {'schema': 'capstone.effusion_pilot_analysis.v1', 'status': 'incomplete' if failures else 'complete',
              'inputs': {name: stage['identity'] for name, stage in stages.items()}, 'accounting': accounting,
              'failures': failures, 'arms': arms, 'contrasts': contrasts, 'attack_qualification': qualification,
              'scope': 'exploratory development-selection pilot; descriptive paired seeds only; no finalist selection',
              'metric_direction': {'auroc': 'higher', 'average_precision': 'higher', 'brier': 'lower',
                                   'sensitivity_at_half': 'higher', 'specificity_at_half': 'higher'}}
    diagnostics = {'schema': 'capstone.effusion_pilot_diagnostics.v1', 'scope': 'research-only; aggregate exports',
                   'runs': [_diagnostics(row, stage['records'].get(row['name'], {})) for stage in stages.values() for row in stage['rows']]}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('summary.json', result), ('diagnostics-summary.json', diagnostics)):
        (output/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    _csv(output/'arms.csv', arms)
    _csv(output/'contrasts.csv', contrasts)
    (output/'report.md').write_text(_report(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen-dir', type=Path, required=True)
    parser.add_argument('--screen-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bucket-dir', type=Path)
    parser.add_argument('--bucket-sha256')
    args = parser.parse_args()
    result = analyze(args.screen_dir, args.screen_sha256, args.output,
                     bucket_dir=args.bucket_dir, bucket_sha256=args.bucket_sha256)
    print(json.dumps({'status': result['status'], 'accounting': result['accounting']}))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
