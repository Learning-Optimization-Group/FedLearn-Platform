"""Reconstruct original selection-study AUROC contrasts without loading study data.

Definitions follow the contrast-building blocks in run_effusion_local_dp.py (A11),
run_effusion_local_robust.py (A12), run_effusion_noniid.py (A19),
run_effusion_encoder_dp.py (A22), and run_effusion_distributed_dp.py (A23).
Stored contrast estimates and intervals are never used. Baseline aliases identify
external score records whose qualification and metrics the caller must validate.
"""

from math import isfinite

SEEDS = tuple(range(20261101, 20261106))
Specs = dict[str, list[dict[str, float]]]


def _qualified_names(study: str, summary: dict) -> set[str]:
    excluded = set(summary.get('unusableCells', [])) | set(summary.get('integrityFailures', []))
    seen, usable = set(), set()
    for run in summary['runs']:
        name = run['name']
        if not isinstance(name, str) or not name:
            raise ValueError('run name must be a nonempty string')
        if name in seen:
            raise ValueError(f'duplicate run name: {name}')
        seen.add(name)
        auc = run.get('auroc')
        if (name in excluded or run.get('finite') is not True or run.get('refusals') != 0
                or isinstance(auc, bool) or not isinstance(auc, (int, float))
                or not isfinite(auc) or not 0.0 <= auc <= 1.0):
            continue
        if study == 'A23' and (run.get('withinClaim') is not True
                              or run.get('secureSumExact') is False
                              or run.get('noiseMultiplierMatches') is False):
            continue
        usable.add(name)
    return usable


def build_specs(study: str, summary: dict) -> Specs:
    """Return original contrast keys and five ordered per-seed coefficient maps.

    A missing or unqualified local component suppresses its entire contrast;
    complete-pair inference never substitutes seeds. The caller must independently
    require exact original-key coverage and validate every external baseline alias.
    A22 refers to its federated DP-follow-up summary, not its centralized gate.
    """
    if study not in ('A11', 'A12', 'A19', 'A22', 'A23'):
        raise ValueError(f'unsupported selection study: {study}')
    usable = _qualified_names(study, summary)
    specs: Specs = {}
    baseline_prefix = {'A11': 'a8_', 'A19': 'iid_', 'A22': 'a17_', 'A23': 'a17_'}.get(study)

    def add(key: str, terms: list[dict[str, float]]) -> None:
        if all(name in usable or (baseline_prefix and name.startswith(baseline_prefix))
               for components in terms for name in components):
            specs[key] = terms

    if study == 'A11':
        for epsilon in (1, 4, 8, 32, 128):
            p = f'eps{epsilon}'
            add(f'auroc|local|{p}', [{f'local_{p}_{s}': 1.0} for s in SEEDS])
            add(f'localCost|{p}', [{f'a8_clean_{s}': 1.0, f'local_{p}_{s}': -1.0} for s in SEEDS])
            if epsilon in (1, 4, 8):
                add(f'local-central|{p}',
                    [{f'local_{p}_{s}': 1.0, f'a8_dp_{p}_{s}': -1.0} for s in SEEDS])

    elif study == 'A12':
        for epsilon in (8, 32, 128):
            p = f'eps{epsilon}'
            for rule in ('mean', 'trimmed_mean', 'median'):
                for condition in ('clean', 'sign_flip', 'label_flip'):
                    names = [f'local_{p}_{rule}_{condition}_{s}' for s in SEEDS]
                    key = f'{p}|{rule}|{condition}'
                    add(f'auroc|{key}', [{name: 1.0} for name in names])
                    if condition != 'clean':
                        add(f'drop|{key}', [{f'local_{p}_{rule}_clean_{s}': 1.0, name: -1.0}
                                           for s, name in zip(SEEDS, names)])
                    if rule != 'mean':
                        add(f'ruleVsMean|{key}', [{name: 1.0, f'local_{p}_mean_{condition}_{s}': -1.0}
                                                 for s, name in zip(SEEDS, names)])

    elif study == 'A19':
        groups = [('nodp', label, 'clean') for label in ('mean', 'trim16', 'median')]
        groups += [('nodp', label, 'fang_trim_m16') for label in ('mean', 'median')]
        groups += [('dp4', 'mean', 'clean')]
        for alpha in ('a1', 'a0p1'):
            for tag, label, condition in groups:
                stem = f'{tag}_{label}_{condition}'
                names = [f'{alpha}_{stem}_{s}' for s in SEEDS]
                key = f'{alpha}|{tag}|{label}|{condition}'
                add(f'auroc|{key}', [{name: 1.0} for name in names])
                add(f'nonIidMinusIid|{key}', [{name: 1.0, f'iid_{stem}_{s}': -1.0}
                                            for s, name in zip(SEEDS, names)])
                if label != 'mean':
                    add(f'ruleVsMean|{key}', [{name: 1.0, f'{alpha}_{tag}_mean_{condition}_{s}': -1.0}
                                             for s, name in zip(SEEDS, names)])
                if condition != 'clean':
                    add(f'drop|{key}', [{f'{alpha}_{tag}_{label}_clean_{s}': 1.0, name: -1.0}
                                       for s, name in zip(SEEDS, names)])

    elif study == 'A22':
        representations = summary['representations']
        if (not isinstance(representations, list) or len(set(representations)) != len(representations)
                or any(rep not in ('R1', 'R2', 'R3') for rep in representations)):
            raise ValueError('invalid A22 passing representations')
        for rep in representations:
            for p in ('clean', 'eps8', 'eps4', 'eps1'):
                names = [f'{rep}_T20_{p}_{s}' for s in SEEDS]
                key = f'{rep}|{p}'
                add(f'auroc|{key}', [{name: 1.0} for name in names])
                add(f'minusR0|{key}', [{name: 1.0, f'a17_zero_T20_{p}_{s}': -1.0}
                                      for s, name in zip(SEEDS, names)])
                if p != 'clean':
                    add(f'dpCost|{key}', [{f'{rep}_T20_clean_{s}': 1.0, name: -1.0}
                                         for s, name in zip(SEEDS, names)])
                    add(f'dpCostMinusR0dpCost|{key}',
                        [{f'{rep}_T20_clean_{s}': 1.0, name: -1.0, f'a17_zero_T20_clean_{s}': -1.0,
                          f'a17_zero_T20_{p}_{s}': 1.0} for s, name in zip(SEEDS, names)])

    else:  # A23
        for epsilon in (8, 4, 1):
            p = f'eps{epsilon}'
            add(f'auroc|rdp|{p}', [{f'rdp_{p}_{s}': 1.0} for s in SEEDS])
            add(f'rdpMinusGdp|{p}', [{f'rdp_{p}_{s}': 1.0, f'a17_zero_T20_{p}_{s}': -1.0} for s in SEEDS])
            for bits in (12, 14, 16, 20):
                if (bits, epsilon) == (12, 1):
                    continue
                names = [f'dist_b{bits}_{p}_{s}' for s in SEEDS]
                key = f'b{bits}|{p}'
                add(f'auroc|{key}', [{name: 1.0} for name in names])
                add(f'distMinusRdp|{key}', [{name: 1.0, f'rdp_{p}_{s}': -1.0}
                                          for s, name in zip(SEEDS, names)])
                add(f'distMinusGdp|{key}', [{name: 1.0, f'a17_zero_T20_{p}_{s}': -1.0}
                                          for s, name in zip(SEEDS, names)])
        for attackers in (4, 16):
            for label, stem in (('b16', 'dist_b16'), ('rdp', 'rdp')):
                add(f'auroc|{label}|eps4|signflip{attackers}',
                    [{f'{stem}_eps4_signflip{attackers}_{s}': 1.0} for s in SEEDS])
                add(f'attackDamage|{label}|signflip{attackers}',
                    [{f'{stem}_eps4_{s}': 1.0, f'{stem}_eps4_signflip{attackers}_{s}': -1.0} for s in SEEDS])
            add(f'damageDistMinusRdp|signflip{attackers}',
                [{f'dist_b16_eps4_{s}': 1.0, f'dist_b16_eps4_signflip{attackers}_{s}': -1.0,
                  f'rdp_eps4_{s}': -1.0, f'rdp_eps4_signflip{attackers}_{s}': 1.0} for s in SEEDS])
    return specs


def incremental_dp_cost_specs(summary: dict) -> Specs:
    """Additional exploratory A19 contrasts, separate from the original contrasts.

    Compute (non-IID clean - non-IID private) - (IID clean - IID private)
    for the mean rule. The two external IID aliases come from the run metadata.
    """
    usable = _qualified_names('A19', summary)
    records = {run['name']: run for run in summary['runs']}
    specs: Specs = {}
    for alpha in ('a1', 'a0p1'):
        names = [(f'{alpha}_nodp_mean_clean_{s}', f'{alpha}_dp4_mean_clean_{s}') for s in SEEDS]
        if not all(name in usable for pair in names for name in pair):
            continue
        terms = []
        for clean, private in names:
            clean_iid, private_iid = records[clean]['iid'], records[private]['iid']
            if clean_iid != clean.split('_', 1)[1] or private_iid != private.split('_', 1)[1]:
                raise ValueError(f'A19 IID baseline identity mismatch: {clean}, {private}')
            terms.append({clean: 1.0, private: -1.0, f'iid_{clean_iid}': -1.0, f'iid_{private_iid}': 1.0})
        specs[f'incrementalDpCost|{alpha}'] = terms
    return specs
