import run_effusion_encoder_dp as dp


def test_only_passing_representations_run_and_pair_with_a17():
    summary = {'results': {'R0': {'auroc': 0.8}, 'R1': {'passed': False}, 'R2': {'passed': True}, 'R3': {'passed': True}}}
    assert dp.passing(summary) == ['R2', 'R3']
    cells = dp.cells(['R2'])
    assert len(cells) == 20 and cells[0]['r0'] == 'zero_T20_clean_20261101'
    assert {c['r0'] for c in cells if c['epsilon'] == 4.0} == {f'zero_T20_eps4_{s}' for s in dp.SEEDS}
    assert dp.passing({'results': {'R1': {'passed': False}}}) == []
