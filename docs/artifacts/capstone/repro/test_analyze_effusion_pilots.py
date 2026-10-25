"""Aggregate-only analysis checks using generated declared records, without fitting."""
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

import run_effusion_pilots as pilot
import analyze_effusion_pilots as analysis


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(cell):
    value = .8 + (cell['seed'] % 2) * .02 - .01 * cell['private']
    if cell['attack'] != 'clean':
        value -= .03 if cell['seed'] % 2 else .05
    return dict(auroc=value, average_precision=value-.1, brier=.2,
                sensitivity_at_half=.7, specificity_at_half=.6)


def stage_fixture(directory, stage='screen', reference=None):
    cells = pilot.screen_cells() if stage == 'screen' else pilot.bucket_cells(reference['selected_recipe'])
    public = dict(schema='capstone.effusion_pilot_plan.v1', stage=stage, synthetic=False,
                  cells=cells, settings=pilot.DEFAULTS, reference=reference, bundle_report_sha256=pilot.BUNDLE_PIN,
                  configuration_sha256={c['name']: pilot._configuration_hash({'name': c['name']}) for c in cells})
    public_sha = write(directory/'public-manifest.json', public)
    plan = public | {'public_manifest_sha256': public_sha,
                     'configurations': {c['name']: {'name': c['name']} for c in cells}}
    plan_sha = write(directory/'frozen-plan.json', plan)
    rows = []
    for c in cells:
        rounds = [dict(round=i+1, auc=metrics(c)['auroc'], privacy={'clip_rate': .2, 'noise_std': .03 if c['private'] else 0.},
                       aggregation_diagnostics={'center_norm': i*.001, 'inner_clip_rate': .1,
                           'center_alignment_with_input_mean': None if i == 0 else .5,
                           'mean_input_norm': .02, 'contaminated_bucket_fraction': c['fraction']},
                       owned_attack_diagnostics=[{'type': c['attack'], 'loss_evaluations': 63,
                           'gradient_evaluations': 60, 'initial_norm': .05, 'upload_norm': .050000001,
                           'intended_upload_norm': .05}] if c['attack'] == 'owned_pgd' else []) for i in range(20)]
        raw = {'pilot_metrics': metrics(c), 'rounds': rounds, 'meta': {'finalHead': {'secret': [12345]}}}
        relative = f"runs/{c['name']}.json"
        rows.append(c | {'metrics': metrics(c), 'status': 'completed', 'record': relative,
                         'sha256': write(directory/relative, raw)})
    summary = dict(schema='capstone.effusion_pilot_result.v1', stage=stage, synthetic=False,
                   status='complete', runs=rows, plan_sha256=plan_sha, reference=reference,
                   checkpoint_commit='abcdef0', checkpoint_manifest='docs/example.json')
    pin = write(directory/'summary.json', summary)
    return summary, pin


def test_complete_screen_exports_all_seeds_and_paired_descriptive_contrasts(tmp_path):
    source = tmp_path/'screen'
    _, pin = stage_fixture(source)
    output = tmp_path/'analysis'
    result = analysis.analyze(source, pin, output)
    assert result['status'] == 'complete'
    assert result['accounting']['screen']['planned'] == result['accounting']['screen']['completed'] == 80
    assert len(result['arms']) == 40
    arm = next(x for x in result['arms'] if x['tag'] == 'mean' and x['backend'] == 'byzfl' and not x['private'])
    assert arm['metrics']['auroc']['mean'] == pytest.approx(.81)
    assert arm['metrics']['auroc']['sample_sd'] == pytest.approx(math.sqrt(.0002))
    assert set(arm['per_seed']) == {'20261211', '20261212'}
    damage = next(x for x in result['contrasts'] if x['id'] == 'attack_damage:mean_owned_pgd_f16:byzfl:nonprivate')
    assert damage['n'] == 2
    assert damage['metrics']['auroc']['mean'] == pytest.approx(.04)
    assert damage['metrics']['auroc']['min'] == pytest.approx(.03)
    assert all(x['qualified'] for x in result['attack_qualification'])
    assert set(p.name for p in output.iterdir()) == {'summary.json', 'arms.csv', 'contrasts.csv', 'diagnostics-summary.json', 'report.md'}
    public = ''.join(p.read_text() for p in output.iterdir())
    assert str(tmp_path) not in public and 'finalHead' not in public and '12345' not in public
    assert 'confidence_interval' not in public and 'p_value' not in public
    with pytest.raises(FileExistsError):
        analysis.analyze(source, pin, output)


def test_summary_pin_record_pin_and_raw_metrics_are_checked_for_every_completed_row(tmp_path):
    source = tmp_path/'screen'
    summary, pin = stage_fixture(source)
    with pytest.raises(ValueError, match='hash'):
        analysis.analyze(source, '0'*64, tmp_path/'wrong')
    row = summary['runs'][0]
    raw = json.loads((source/row['record']).read_text())
    raw['pilot_metrics']['brier'] = .99
    write(source/row['record'], raw)
    with pytest.raises(ValueError, match='hash'):
        analysis.analyze(source, pin, tmp_path/'tampered')
    row['sha256'] = write(source/row['record'], raw)
    pin = write(source/'summary.json', summary)
    with pytest.raises(ValueError, match='metrics'):
        analysis.analyze(source, pin, tmp_path/'changed-metrics')


@pytest.mark.parametrize('change', ['synthetic', 'undeclared', 'duplicate', 'configuration'])
def test_refuses_synthetic_and_non_declared_inputs(tmp_path, change):
    source = tmp_path/'screen'
    summary, _ = stage_fixture(source)
    if change == 'synthetic':
        summary['synthetic'] = True
    elif change == 'undeclared':
        summary['runs'][0]['tau'] = 77
    elif change == 'duplicate':
        summary['runs'].append(copy.deepcopy(summary['runs'][0]))
    else:
        plan = json.loads((source/'frozen-plan.json').read_text())
        plan['configurations'][summary['runs'][0]['name']] = {'changed': True}
        summary['plan_sha256'] = write(source/'frozen-plan.json', plan)
    pin = write(source/'summary.json', summary)
    with pytest.raises(ValueError):
        analysis.analyze(source, pin, tmp_path/'out')


def test_incomplete_screen_preserves_missing_error_and_unrun_cells_without_ranking(tmp_path):
    source = tmp_path/'screen'
    summary, _ = stage_fixture(source)
    summary['runs'][0]['status'] = 'error'
    summary['runs'][1]['status'] = 'not_run'
    summary['runs'].pop()
    summary['status'] = 'incomplete'
    pin = write(source/'summary.json', summary)
    result = analysis.analyze(source, pin, tmp_path/'out')
    assert result['status'] == 'incomplete'
    assert result['accounting']['screen']['completed'] == 77
    assert {f['status'] for f in result['failures']} == {'error', 'not_run', 'missing'}
    assert len(result['arms']) == 40
    assert 'selected_recipe' not in result
    assert any(c['n'] < 2 and c['status'] == 'incomplete' for c in result['contrasts'])


def test_center_attack_and_bucket_diagnostics_keep_research_scope_and_reused_references(tmp_path):
    source = tmp_path/'screen'
    summary, pin = stage_fixture(source)
    reference = dict(selected_recipe='tight', summary_sha256=pin, plan_sha256=summary['plan_sha256'])
    buckets = tmp_path/'buckets'
    _, bucket_pin = stage_fixture(buckets, 'buckets', reference)
    result = analysis.analyze(source, pin, tmp_path/'out', bucket_dir=buckets, bucket_sha256=bucket_pin)
    assert result['accounting']['buckets']['planned'] == 46
    assert result['accounting']['buckets']['reused_screen_references'] == 2
    diagnostic = json.loads((tmp_path/'out/diagnostics-summary.json').read_text())
    center = next(x for x in diagnostic['runs'] if x['tag'] == 'center_r1_clean' and not x['private'])
    assert center['center']['center_norm']['first'] == 0
    assert center['center']['center_norm']['last'] == pytest.approx(.019)
    assert center['center']['center_alignment_with_input_mean']['missing'] == 1
    attack = next(x for x in diagnostic['runs'] if x['tag'] == 'mean_owned_pgd_f16')
    assert attack['owned_attack']['budget_matches_expected'] is True
    bucket = next(x for x in diagnostic['runs'] if x['tag'] == 'median_b4_iid_pgd_f16')
    assert bucket['buckets']['expected_contaminated_fraction'] == pytest.approx(1-math.comb(64,4)/math.comb(80,4))
    assert any(x['tag'] == 'median_b1_iid_clean' and x['reused_from_screen'] for x in result['arms'])
    assert any(x['oracle_nonprivate'] for x in diagnostic['runs'])
    with pytest.raises(ValueError):
        analysis.analyze(source, pin, tmp_path/'other', bucket_dir=buckets)


def change_auroc(source, summary, tag, private, values, backend='byzfl'):
    for row in summary['runs']:
        if row['tag'] == tag and row['private'] == private and row['backend'] == backend:
            value = values[row['seed']]
            row['metrics']['auroc'] = value
            raw = json.loads((source/row['record']).read_text())
            raw['pilot_metrics']['auroc'] = value
            for line in raw['rounds']:
                line['auc'] = value
            row['sha256'] = write(source/row['record'], raw)


def test_paired_contrast_signs_use_actual_matched_seed_differences(tmp_path):
    source = tmp_path/'screen'
    summary, _ = stage_fixture(source)
    first, second = 20261211, 20261212
    for tag, private, values in (
        ('mean', False, (.8, .9)), ('mean', True, (.7, .6)),
        ('mean_owned_pgd_f16', False, (.73, .82)),
        ('center_r1_clean', False, (.85, .83)), ('center_r1_owned_pgd', False, (.75, .65)),
        ('mean_k64', False, (.79, .85)), ('mean_k64', True, (.72, .66)),
    ):
        change_auroc(source, summary, tag, private, dict(zip((first, second), values)))
    # Deliberately scramble storage order; pairing must use identities rather than zip order.
    summary['runs'].reverse()
    result = analysis.analyze(source, write(source/'summary.json', summary), tmp_path/'out')
    contrasts = {c['id']: c for c in result['contrasts']}
    checks = {
        'mechanism_vs_mean:center_r1:clean:nonprivate': (.05, -.07),
        'mechanism_vs_mean:center_r1:pgd:nonprivate': (.02, -.17),
        'privacy_cost:mean:byzfl': (.1, .3),
        'attack_damage:center_r1:byzfl:nonprivate': (.10, .18),
        'attack_damage:mean_owned_pgd_f16:byzfl:nonprivate': (.07, .08),
        'additional_privacy_cost:k64': (-.03, -.11),
    }
    for name, differences in checks.items():
        contrast = contrasts[name]
        assert contrast['n'] == 2
        assert [contrast['per_seed'][str(s)]['auroc'] for s in (first, second)] == pytest.approx(differences)
        assert contrast['metrics']['auroc']['mean'] == pytest.approx(sum(differences)/2)
        assert contrast['metrics']['auroc']['sample_sd'] == pytest.approx(abs(differences[0]-differences[1])/math.sqrt(2))


@pytest.mark.parametrize('mutation', ['different_seed', 'nan', 'absent_metric', 'round_gap', 'claim_complete'])
def test_completed_seed_and_metric_integrity_is_required(tmp_path, mutation):
    source = tmp_path/'screen'
    summary, _ = stage_fixture(source)
    row = summary['runs'][0]
    if mutation == 'different_seed':
        row['seed'] = 123
    elif mutation == 'claim_complete':
        summary['runs'].pop()
    else:
        raw = json.loads((source/row['record']).read_text())
        if mutation == 'nan':
            row['metrics']['auroc'] = float('nan')
            raw['pilot_metrics']['auroc'] = float('nan')
        elif mutation == 'absent_metric':
            del row['metrics']['brier']
            del raw['pilot_metrics']['brier']
        else:
            raw['rounds'][0]['round'] = 2
        row['sha256'] = write(source/row['record'], raw)
    with pytest.raises(ValueError):
        analysis.analyze(source, write(source/'summary.json', summary), tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_partial_attack_pairs_cannot_qualify_from_one_damaging_seed(tmp_path):
    source = tmp_path/'screen'
    summary, _ = stage_fixture(source)
    row = next(r for r in summary['runs'] if r['tag'] == 'mean_owned_pgd_f16' and not r['private'])
    row['status'] = 'not_run'
    summary['status'] = 'incomplete'
    result = analysis.analyze(source, write(source/'summary.json', summary), tmp_path/'out')
    attack = next(a for a in result['attack_qualification'] if a['attack'] == 'owned_pgd')
    assert attack['mean_auroc_loss'] > .02
    assert attack['n'] == 1 and attack['qualified'] is False


def test_bucket_reference_requires_the_same_pinned_screen(tmp_path):
    source = tmp_path/'screen'
    summary, pin = stage_fixture(source)
    buckets = tmp_path/'buckets'
    _, bucket_pin = stage_fixture(buckets, 'buckets', dict(selected_recipe='tight', summary_sha256='f'*64,
                                                        plan_sha256=summary['plan_sha256']))
    with pytest.raises(ValueError, match='pinned screen'):
        analysis.analyze(source, pin, tmp_path/'out', bucket_dir=buckets, bucket_sha256=bucket_pin)


def test_declared_incomplete_summary_cannot_be_promoted_to_complete(tmp_path):
    source = tmp_path/'screen'
    summary, _ = stage_fixture(source)
    summary['status'] = 'incomplete'
    pin = write(source/'summary.json', summary)
    with pytest.raises(ValueError, match='status'):
        analysis.analyze(source, pin, tmp_path/'out')
