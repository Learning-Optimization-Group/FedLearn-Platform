"""Generated-only checks for the independent finalized-artifact audit."""
import importlib
import json

import pytest

import analyze_effusion_finalists as analysis
from test_analyze_effusion_finalists import artifact_fixture


@pytest.fixture
def finalized(artifact_fixture, tmp_path, monkeypatch):
    directory, _, summary, save, _ = artifact_fixture
    source = tmp_path / 'generated_fixture.py'
    source.write_text('fixture = 1\n')
    inventory = {'generated_fixture.py': analysis.pilot.sha(source)}
    monkeypatch.setattr(analysis.pilot, '_paths', lambda: {'generated_fixture.py': source})
    archive = directory / 'sources'
    archive.mkdir()
    (archive / 'generated_fixture.py').write_bytes(source.read_bytes())
    public = json.loads((directory / 'public-manifest.json').read_text())
    plan = json.loads((directory / 'frozen-plan.json').read_text())
    public['source_sha256'] = plan['source_sha256'] = inventory
    plan['public_manifest_sha256'] = save(directory / 'public-manifest.json', public)
    summary['plan_sha256'] = save(directory / 'frozen-plan.json', plan)
    for row in summary['runs']:
        raw = json.loads((directory / row['record']).read_text())
        if row['attack'] == 'owned_pgd':
            for line in raw['rounds']:
                line['owned_attack_diagnostics'] = [{'type': 'owned_pgd', 'coalition_size': 16,
                    'total_submitted': 80, 'coalition_fraction': .2, 'steps': 20, 'restarts': 3,
                    'step_ratio': .25, 'gradient_evaluations': 60, 'loss_evaluations': 63}]
        row['sha256'] = save(directory / row['record'], raw)
    raw_pin = save(directory / 'summary.json', summary)
    result = analysis.summarize(summary['runs'])
    result['inputs'] = {'summary_sha256': raw_pin, 'plan_sha256': summary['plan_sha256'],
                        'public_manifest_sha256': plan['public_manifest_sha256'],
                        'source_sha256': inventory,
                        'record_sha256': {r['name']: r['sha256'] for r in summary['runs']}}
    output = tmp_path / 'analysis'
    output.mkdir()
    result_pin = save(output / 'summary.json', result)
    return directory, raw_pin, output, result_pin, summary, result, save, source


def module():
    return importlib.import_module('audit_effusion_finalists')


def test_generated_complete_artifacts_reconcile_without_private_data(finalized):
    directory, raw_pin, output, result_pin, *_ = finalized
    receipt = module().audit(directory, raw_pin, output, result_pin)
    assert receipt['status'] == 'verified'
    assert receipt['counts'] == {'cells': 196, 'rounds': 3920, 'main_cells': 192,
                                 'private_cells': 98, 'pgd_rounds': 1920, 'contrasts': 31,
                                 'source_files': 1}
    assert receipt['maximum_seed_interval_difference'] < 1e-12
    assert 0 <= receipt['maximum_delta_at_epsilon4'] <= 1e-5 + 1e-12
    text = json.dumps(receipt)
    for forbidden in ('head.weight', 'patient_ids', 'generated_fixture_bundle', '/Users/', '/private/', '/tmp/'):
        assert forbidden not in text


@pytest.mark.parametrize('mutation', ['raw_hash', 'seed', 'cohort', 'pgd', 'gdp'])
def test_rejects_raw_hash_or_rehashed_execution_corruption(finalized, mutation):
    directory, raw_pin, output, result_pin, summary, result, save, _ = finalized
    row = next(r for r in summary['runs'] if r['attack'] == 'owned_pgd' and r['private'])
    raw = json.loads((directory / row['record']).read_text())
    if mutation == 'raw_hash': raw['extra'] = True
    elif mutation == 'seed': raw['meta']['seeds']['client'] += 1
    elif mutation == 'cohort': raw['rounds'][0]['participants']['submitted'] = 79
    elif mutation == 'pgd': raw['rounds'][0]['owned_attack_diagnostics'][0]['gradient_evaluations'] = 59
    else: raw['meta']['noise_multiplier'] /= 2
    digest = save(directory / row['record'], raw)
    if mutation != 'raw_hash':
        row['sha256'] = digest
        raw_pin = save(directory / 'summary.json', summary)
        result['inputs']['summary_sha256'] = raw_pin
        result['inputs']['record_sha256'][row['name']] = digest
        result_pin = save(output / 'summary.json', result)
    with pytest.raises(ValueError):
        module().audit(directory, raw_pin, output, result_pin)


@pytest.mark.parametrize('mutation', ['endpoint', 'effect', 'components', 'arm'])
def test_rejects_repinned_final_analysis_corruption(finalized, mutation):
    directory, raw_pin, output, _, _, result, save, _ = finalized
    contrast = result['contrasts'][0]
    if mutation == 'endpoint': contrast['seed_interval']['ci_high'] += .02
    elif mutation == 'effect': contrast['seed_interval']['seed_effects'][13001] += .02
    elif mutation == 'components': contrast['components'][0]['coefficient'] = -1
    else: result['arms'][0]['per_seed'][13001]['auroc'] += .02
    result_pin = save(output / 'summary.json', result)
    with pytest.raises(ValueError):
        module().audit(directory, raw_pin, output, result_pin)


@pytest.mark.parametrize('where', ['live', 'archive'])
def test_changed_frozen_source_is_refused(finalized, where):
    directory, raw_pin, output, result_pin, _, _, _, source = finalized
    target = source if where == 'live' else directory / 'sources' / 'generated_fixture.py'
    target.write_text('changed = True\n')
    with pytest.raises(ValueError, match='source'):
        module().audit(directory, raw_pin, output, result_pin)


def test_cli_writes_exclusive_aggregate_receipt(finalized, tmp_path, monkeypatch):
    directory, raw_pin, output, result_pin, *_ = finalized
    receipt = tmp_path / 'receipt.json'
    monkeypatch.setattr('sys.argv', ['audit', '--run-dir', str(directory), '--summary-sha256', raw_pin,
        '--analysis-dir', str(output), '--analysis-sha256', result_pin, '--output', str(receipt)])
    assert module().main() == 0
    assert json.loads(receipt.read_text())['status'] == 'verified'
    with pytest.raises(FileExistsError):
        module().main()


def test_engine_rdp_roundtrip_noise_is_accepted_and_difference_recorded(finalized):
    directory, _, output, _, summary, result, save, _ = finalized
    row = next(r for r in summary['runs'] if r['private'])
    raw = json.loads((directory / row['record']).read_text())
    # Actual public-calibration roundtrip through the engine's RDP search.
    raw['meta']['noise_multiplier'] = 4.835102780756167
    sensitivity = raw['rounds'][0]['privacy']['sensitivity']
    for line in raw['rounds']:
        line['privacy']['noise_std'] = line['privacy']['server_noise_std'] = 4.835102780756167 * sensitivity
    row['sha256'] = save(directory / row['record'], raw)
    raw_pin = save(directory / 'summary.json', summary)
    result['inputs']['summary_sha256'] = raw_pin
    result['inputs']['record_sha256'][row['name']] = row['sha256']
    result_pin = save(output / 'summary.json', result)
    receipt = module().audit(directory, raw_pin, output, result_pin)
    assert 3e-10 < receipt['maximum_calibration_multiplier_difference'] < 4e-10


def test_nonconstant_seed_effects_use_sample_sd_and_student_t():
    from test_analyze_effusion_finalists import rows
    data = rows()
    for row in data:
        if row['method'] == 'projected128' and row['private'] and row['attack'] == 'clean':
            row['metrics']['auroc'] += .01 * (row['seed'] - 13001)
    result = json.loads(json.dumps(analysis.summarize(data)))
    assert module()._intervals(result) < 1e-12
    result['contrasts'][0]['seed_interval']['df'] = 12
    with pytest.raises(ValueError, match='seed set'):
        module()._intervals(result)
