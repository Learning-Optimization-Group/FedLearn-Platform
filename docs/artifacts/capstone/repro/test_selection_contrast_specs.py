"""Synthetic contract checks for the selection-only contrast reconstruction."""

import ast
from pathlib import Path

import pytest

from selection_contrast_specs import SEEDS, build_specs, incremental_dp_cost_specs


def summary(study):
    if study == 'A11':
        names = [f'local_eps{e}_{s}' for e in (1, 4, 8, 32, 128) for s in SEEDS]
    elif study == 'A12':
        names = [f'local_eps{e}_{r}_{c}_{s}' for e in (8, 32, 128)
                 for r in ('mean', 'trimmed_mean', 'median')
                 for c in ('clean', 'sign_flip', 'label_flip') for s in SEEDS]
    elif study == 'A19':
        groups = ('nodp_mean_clean', 'nodp_trim16_clean', 'nodp_median_clean',
                  'nodp_mean_fang_trim_m16', 'nodp_median_fang_trim_m16', 'dp4_mean_clean')
        names = [f'{a}_{g}_{s}' for a in ('a1', 'a0p1') for g in groups for s in SEEDS]
    elif study == 'A22':
        names = [f'R2_T20_{p}_{s}' for p in ('clean', 'eps8', 'eps4', 'eps1') for s in SEEDS]
    else:
        names = [f'dist_b{b}_eps{e}_{s}' for b in (12, 14, 16, 20) for e in (8, 4, 1)
                 if (b, e) != (12, 1) for s in SEEDS]
        names += [f'rdp_eps{e}_{s}' for e in (8, 4, 1) for s in SEEDS]
        names += [f'{arm}_eps4_signflip{m}_{s}' for arm in ('dist_b16', 'rdp')
                  for m in (4, 16) for s in SEEDS]
    return {'runs': [{'name': n, 'finite': True, 'refusals': 0, 'auroc': 0.7,
                      'withinClaim': True} for n in names],
            'representations': ['R2'], 'contrasts': {'untrusted': {'point': -999}}}


@pytest.mark.parametrize('study,count', [('A11', 13), ('A12', 63), ('A19', 34), ('A22', 14), ('A23', 49)])
def test_complete_synthetic_studies_preserve_all_contrast_groups(study, count):
    specs = build_specs(study, summary(study))
    assert len(specs) == count
    assert 'untrusted' not in specs
    assert all(len(terms) == 5 for terms in specs.values())
    for terms in specs.values():
        for seed, components in zip(SEEDS, terms):
            assert all(name.endswith(f'_{seed}') for name in components)
            assert all(isinstance(coefficient, float) for coefficient in components.values())


def test_cross_study_aliases_and_four_term_cost_signs():
    seed = SEEDS[0]
    assert build_specs('A11', summary('A11'))['local-central|eps4'][0] == {
        f'local_eps4_{seed}': 1.0, f'a8_dp_eps4_{seed}': -1.0}
    assert build_specs('A19', summary('A19'))['nonIidMinusIid|a0p1|dp4|mean|clean'][0] == {
        f'a0p1_dp4_mean_clean_{seed}': 1.0, f'iid_dp4_mean_clean_{seed}': -1.0}
    assert build_specs('A22', summary('A22'))['dpCostMinusR0dpCost|R2|eps4'][0] == {
        f'R2_T20_clean_{seed}': 1.0, f'R2_T20_eps4_{seed}': -1.0,
        f'a17_zero_T20_clean_{seed}': -1.0, f'a17_zero_T20_eps4_{seed}': 1.0}


def test_a23_exactly_matches_original_pure_function_without_importing_harness():
    path = Path(__file__).with_name('run_effusion_distributed_dp.py')
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'contrast_spec')
    namespace = {'SEEDS': SEEDS, 'PRIVACY': (8.0, 4.0, 1.0), 'BITS': (12, 14, 16, 20),
                 'INFEASIBLE': {(12, 1.0)}, 'ATTACK_EPSILON': 4.0, 'ATTACK_BITS': 16,
                 'ATTACKERS': (4, 16), 'privacy_label': lambda e: f'eps{e:g}'}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    data = summary('A23')
    assert build_specs('A23', data) == namespace['contrast_spec']({r['name'] for r in data['runs']})


@pytest.mark.parametrize('study,filename', [
    ('A11', 'run_effusion_local_dp.py'), ('A12', 'run_effusion_local_robust.py'),
    ('A19', 'run_effusion_noniid.py'), ('A22', 'run_effusion_encoder_dp.py')])
def test_definitions_exactly_match_original_source_blocks(study, filename):
    """Execute only public coefficient-generation statements, without importing harnesses."""
    path = Path(__file__).with_name(filename)
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    # Original contrast blocks are inside the output-staging try block.
    body = next(node.body for node in main.body if isinstance(node, ast.Try))
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, node in enumerate(body) if assigns(node, 'contrasts'))
    end = next(i for i, node in enumerate(body) if assigns(node, 'summary'))
    data = summary(study)
    namespace = {'SEEDS': SEEDS, 'usable': {r['name'] for r in data['runs']},
                 'EPSILONS': (1., 4., 8., 32., 128.) if study == 'A11' else (8., 32., 128.),
                 'CENTRAL_EPSILONS': (1., 4., 8.), 'RULES': ('mean', 'trimmed_mean', 'median'),
                 'BEHAVIORS': dict.fromkeys(('clean', 'sign_flip', 'label_flip')),
                 'ALPHAS': (1., .1), 'representations': ['R2'], 'PRIVACY': (None, 8., 4., 1.),
                 'privacy_label': lambda e: 'clean' if e is None else f'eps{e:g}'}
    groups = [('nodp', label, label, None, 'clean', []) for label in ('mean', 'trim16', 'median')]
    groups += [('nodp', label, label, None, 'fang_trim_m16', []) for label in ('mean', 'median')]
    namespace['GROUPS'] = groups + [('dp4', 'mean', 'mean', None, 'clean', [])]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('alpha_label', 'cells')]
    exec(compile(ast.Module(body=functions + body[start:end], type_ignores=[]), str(path), 'exec'), namespace)
    assert build_specs(study, data) == namespace['contrasts']


@pytest.mark.parametrize('change', [{'finite': False}, {'refusals': 1}, {'auroc': float('nan')},
                                   {'withinClaim': False}, {'secureSumExact': False},
                                   {'noiseMultiplierMatches': False}])
def test_unqualified_a23_cell_removes_every_dependent_contrast(change):
    data = summary('A23')
    name = f'dist_b16_eps4_{SEEDS[0]}'
    next(r for r in data['runs'] if r['name'] == name).update(change)
    result = build_specs('A23', data)
    assert not any(name in components for terms in result.values() for components in terms)
    assert 'auroc|b20|eps4' in result


def test_missing_clean_arm_suppresses_cost_but_preserves_qualified_private_auc():
    data = summary('A22')
    data['runs'] = [r for r in data['runs'] if r['name'] != f'R2_T20_clean_{SEEDS[0]}']
    specs = build_specs('A22', data)
    assert 'auroc|R2|eps4' in specs
    assert not any(key.startswith('dpCost') for key in specs)


def test_duplicate_identity_and_unknown_study_are_refused():
    data = summary('A11')
    data['runs'].append(dict(data['runs'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        build_specs('A11', data)
    with pytest.raises(ValueError, match='unsupported'):
        build_specs('A99', {'runs': []})


def test_incremental_privacy_cost_is_separate_and_uses_four_matched_components():
    data = summary('A19')
    for run in data['runs']:
        run['iid'] = run['name'].split('_', 1)[1]
    extra = incremental_dp_cost_specs(data)
    assert set(extra) == {'incrementalDpCost|a1', 'incrementalDpCost|a0p1'}
    assert set(extra).isdisjoint(build_specs('A19', data))
    seed = SEEDS[0]
    assert extra['incrementalDpCost|a0p1'][0] == {
        f'a0p1_nodp_mean_clean_{seed}': 1.0, f'a0p1_dp4_mean_clean_{seed}': -1.0,
        f'iid_nodp_mean_clean_{seed}': -1.0, f'iid_dp4_mean_clean_{seed}': 1.0}
    metrics = dict(zip(extra['incrementalDpCost|a0p1'][0], (0.8132, 0.6563, 0.819, 0.6947)))
    assert sum(metrics[n] * c for n, c in extra['incrementalDpCost|a0p1'][0].items()) == pytest.approx(0.0326)
    data['runs'] = [r for r in data['runs'] if r['name'] != f'a0p1_nodp_mean_clean_{seed}']
    assert set(incremental_dp_cost_specs(data)) == {'incrementalDpCost|a1'}
