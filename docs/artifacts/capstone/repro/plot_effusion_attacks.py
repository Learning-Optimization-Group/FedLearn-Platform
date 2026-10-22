"""Render the A9 attack-study heatmaps from the hash-pinned record: absolute AUROC and attack-induced drop.

Every plotted number is read from the record's contrasts and written beside the figure as JSON, so the
figure is a rendering of the record, never a hand-entered claim. Unavailable cells are drawn hatched
with a reason, never as zero.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROWS = [('nodp', 'mean', 'No DP · mean'), ('nodp', 'trimmed_mean', 'No DP · trimmed mean (0.2)'),
        ('nodp', 'median', 'No DP · median'), ('dp4', 'mean', 'DP ε=4 · mean'),
        ('dp4', 'trimmed_mean', 'DP ε=4 · trimmed mean (0.2)'), ('dp4', 'median', 'DP ε=4 · median')]
CONDITIONS = [('clean', 'No attack'), ('sign_flip', 'Sign flip ×−4'), ('label_flip', 'Label flip'), ('alie', 'ALIE z=0.49')]
UNAVAILABLE = {('dp4', 'median'): 'unavailable: no replace-one\nsensitivity bound for central DP'}
SEQUENTIAL = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']  # blue 100..700
DIVERGING = ['#1c5cab', '#f0efec', '#e34948']  # blue pole, neutral midpoint, red pole
INK, MUTED, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'


def _luminance(rgb):
    channel = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    r, g, b = (channel(c) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ink(rgb):
    """The text token (dark ink or surface) with the higher WCAG contrast against the cell fill."""
    fill = _luminance(rgb)
    contrast = lambda hexcode: (max(fill, _luminance(tuple(int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5)))) + 0.05) / \
        (min(fill, _luminance(tuple(int(hexcode[i:i + 2], 16) / 255 for i in (1, 3, 5)))) + 0.05)  # noqa: E731
    return max((INK, SURFACE), key=contrast)


def _clean(text):
    return text.replace('-0.000', '0.000').replace('+0.000', '0.000')


def cell_values(contrasts):
    absolute, drop = {}, {}
    for tag, rule, _ in ROWS:
        for condition, _ in CONDITIONS:
            a = contrasts.get(f'auroc|{tag}|{rule}|{condition}')
            if a is not None:
                absolute[(tag, rule, condition)] = a
            d = contrasts.get(f'drop|{tag}|{rule}|{condition}')
            if d is not None:
                drop[(tag, rule, condition)] = d
    return absolute, drop


def render(summary_path, expected_sha256, output_stem):
    raw = Path(summary_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SystemExit('summary hash mismatch')
    summary = json.loads(raw)
    if summary['unusableCells']:
        raise SystemExit(f"unusable cells must be shown, not dropped: {summary['unusableCells']}")
    absolute, drop = cell_values(summary['contrasts'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'svg.hashsalt': 'capstone-a9',
                         'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})
    seq = LinearSegmentedColormap.from_list('seq', SEQUENTIAL)
    div = LinearSegmentedColormap.from_list('div', DIVERGING)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), gridspec_kw={'width_ratios': [4, 3]})
    panels = [(axes[0], CONDITIONS, absolute, seq, Normalize(0.55, 0.85), 'A. Selection AUROC (5-seed mean)', '{:.3f}'),
              (axes[1], CONDITIONS[1:], drop, div, TwoSlopeNorm(0.0, -0.07, 0.07),
               'B. Attack-induced drop (clean − attacked, same rule & seed)', '{:+.3f}')]
    plotted = []
    for ax, conditions, values, cmap, norm, title, fmt in panels:
        ax.set_xlim(0, len(conditions)); ax.set_ylim(len(ROWS), 0)
        for i, (tag, rule, label) in enumerate(ROWS):
            for j, (condition, _) in enumerate(conditions):
                key = (tag, rule, condition)
                if (tag, rule) in UNAVAILABLE:
                    ax.add_patch(plt.Rectangle((j + .02, i + .04), .96, .92, facecolor=SURFACE, edgecolor='#b8b7b1',
                                               hatch='////', linewidth=.6))
                    if j == 0:
                        ax.text(len(conditions) / 2, i + .5, UNAVAILABLE[(tag, rule)].replace('\n', ' '),
                                ha='center', va='center', fontsize=7, color=MUTED,
                                bbox={'facecolor': SURFACE, 'edgecolor': 'none', 'pad': 1.5})
                    plotted.append({'panel': title[:1], 'row': label, 'column': condition, 'status': 'unavailable'})
                    continue
                v = values[key]
                colour = cmap(norm(v['point']))
                ax.add_patch(plt.Rectangle((j + .02, i + .04), .96, .92, facecolor=colour, edgecolor=SURFACE, linewidth=2))
                ink = _ink(colour)
                ax.text(j + .5, i + .42, _clean(fmt.format(v['point'])), ha='center', va='center', fontsize=9,
                        fontweight='bold', color=ink)
                ax.text(j + .5, i + .72, _clean('[{:.3f}, {:.3f}]'.format(*v['interval95'])), ha='center', va='center',
                        fontsize=6.5, color=ink)
                plotted.append({'panel': title[:1], 'row': label, 'column': condition, 'point': v['point'],
                                'interval95': v['interval95']})
        ax.set_xticks([j + .5 for j in range(len(conditions))], [name for _, name in conditions])
        ax.set_yticks([i + .5 for i in range(len(ROWS))], [label for _, _, label in ROWS] if ax is axes[0] else [])
        ax.tick_params(length=0, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.xaxis.tick_top()
        ax.set_title(title, loc='left', fontweight='bold', color=INK, pad=22)
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=.035, pad=.02)
        bar.outline.set_visible(False); bar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    fig.text(.01, .01, 'R0 linear head, 80 whole-patient IID clients, 20 rounds, 16 attacker slots (20%). No-DP cells: LR 0.1, clip 1.0; '
             'DP cells: LR 0.03, clip 0.05, central ε=4, δ=1e-5, client-level. Brackets: 95% patient-cluster bootstrap of the '
             '5-seed mean (selection half). Negative drop = attacked run scored higher.', fontsize=6.8, color=MUTED, wrap=True)
    fig.tight_layout(rect=(0, .05, 1, 1))
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{output_stem}.png', dpi=200)
    fig.savefig(f'{output_stem}.svg', metadata={'Date': None})
    values_path = Path(f'{output_stem}-values.json')
    values_path.write_text(json.dumps({'summarySha256': expected_sha256, 'cells': plotted}, indent=2, sort_keys=True) + '\n')
    return plotted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output-stem', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({'cells': len(render(args.summary, args.summary_sha256, args.output_stem))}))
