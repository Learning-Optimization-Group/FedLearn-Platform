"""Final-stage gates and actual projected-head execution on generated inputs."""
import hashlib
import json

import pytest

import finalist_design
import run_effusion_pilots as pilot
from test_run_effusion_byzfl_qualification import inputs


@pytest.fixture
def final_reference(tmp_path, monkeypatch):
    receipt = {'status': 'complete', 'total_new_cells_completed': 126,
               'screen_summary_sha256': 'a'*64, 'bucket_summary_sha256': 'b'*64}
    path = tmp_path/'phase3.json'
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(pilot, 'PHASE3', path, raising=False)
    return {'screen_summary_sha256': 'a'*64, 'bucket_summary_sha256': 'b'*64,
            'phase3_verification_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'selected_width': 128, 'selected_center_tau': .0125}


def test_final_stage_accepts_only_declared_budget_and_verified_parent_identity(final_reference):
    pilot.validate_design(finalist_design.cells(), stage='finalists', reference=final_reference)
    with pytest.raises(ValueError):
        pilot.validate_design(finalist_design.cells()[:-1], stage='finalists', reference=final_reference)
    with pytest.raises(ValueError):
        pilot.validate_design(finalist_design.cells(), stage='finalists', reference=final_reference | {'screen_summary_sha256': 'c'*64})


def test_final_stage_requires_completed_diagnostics(final_reference, monkeypatch):
    path = pilot.PHASE3
    receipt = json.loads(path.read_text()) | {'status': 'incomplete'}
    path.write_text(json.dumps(receipt))
    reference = final_reference | {'phase3_verification_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):
        pilot.validate_design(finalist_design.cells(), stage='finalists', reference=reference)


def test_generated_final_stage_keeps_projected_dimensions_and_calibration(inputs, tmp_path):
    cell = next(c for c in finalist_design.cells() if c['method'] == 'projected128' and c['private'] and c['attack'] == 'clean')
    plan_dir = tmp_path/'plan'
    pilot.preflight(inputs['bundle'], inputs['bundle_sha256'], plan_dir, cells=[cell],
                    settings=inputs['settings'], synthetic=True, stage='finalists')
    pin = hashlib.sha256((plan_dir/'plan.json').read_bytes()).hexdigest()
    output = tmp_path/'runs'
    result = pilot.execute(plan_dir, pin, output)
    assert result['status'] == 'complete' and result['stage'] == 'finalists'
    record = json.loads((output/result['runs'][0]['record']).read_text())
    assert len(record['meta']['finalHead']['head.weight']) == 2
    assert len(record['meta']['finalHead']['head.weight'][0]) == 128
    assert record['pilot_accounting']['exact_epsilon'] <= 4.+1e-6
