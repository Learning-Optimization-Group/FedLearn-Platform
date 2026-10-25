"""Prospective finalist budget; fitting is gated by a separate source checkpoint."""
SEEDS = tuple(range(13001, 13013))
METHODS = ('mean512', 'projected128', 'trim16', 'center_quarter')


def cells():
    def make(method, seed, private, attack, backend='byzfl', role='main'):
        rule = 'trimmed_mean' if method == 'trim16' else 'centered_clip' if method == 'center_quarter' else 'mean'
        width = 128 if method == 'projected128' else 512
        tag = ('mean' if attack == 'clean' else 'mean_owned_pgd_f16') if method == 'mean512' else f'{method}_{attack}'
        return {'name': f"{method}_{backend}_{attack}_{'gdp4' if private else 'clean'}_s{seed}",
                'tag': tag, 'method': method, 'role': role, 'seed': seed, 'private': private,
                'kind': 'fl', 'backend': backend, 'rule': rule, 'width': width,
                'clip_norm': .05, 'lr': .03, 'attack': attack, 'fraction': .2 if attack == 'owned_pgd' else 0.,
                'bucket': 1, 'tau': .0125 if method == 'center_quarter' else .05,
                'alpha': None, 'regularization': None, 'row_bound': None}

    # Complete four backend comparisons before fitting any attacked/candidate arm.
    result = []
    for seed in SEEDS[:2]:
        for private in (False, True):
            result.append(make('mean512', seed, private, 'clean', 'native', 'anchor'))
            result.append(make('mean512', seed, private, 'clean'))
    names = {c['name'] for c in result}
    for seed in SEEDS:
        for private in (False, True):
            for method in METHODS:
                for attack in ('clean', 'owned_pgd'):
                    cell = make(method, seed, private, attack)
                    if cell['name'] not in names:
                        result.append(cell)
                        names.add(cell['name'])
    return result
