import json

import check_effusion_dp_noise_records as chk


def _record(tmp_path, rule, trim, std):
    meta = {'privacy_mode': 'central', 'noise_multiplier': 2.0,
            'config': {'server': {'rule': rule, 'clip_norm': 0.05, 'rule_params': {'trim_ratio': trim}}}}
    line = {'round': 1, 'participants': {'accepted': 80}, 'privacy': {'noise_std': std}}
    (tmp_path / 'r.json').write_text(json.dumps({'meta': meta, 'rounds': [line]}))


def test_trimmed_mean_uses_the_trimmed_denominator(tmp_path):
    _record(tmp_path, 'trimmed_mean', 0.2, 2.0 * 2 * 0.05 / 48)
    assert chk.check([tmp_path])['mismatches'] == []


def test_a_wrong_std_is_reported(tmp_path):
    _record(tmp_path, 'mean', 0.2, 2.0 * 0.05 / 80)  # add/remove scale on a replace-one run
    assert len(chk.check([tmp_path])['mismatches']) == 1
