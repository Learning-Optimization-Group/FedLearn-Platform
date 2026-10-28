"""Render aggregate-only projection study figures in the existing plotting environment."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.audit.read_text())['comparisons']
    args.output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for kind,title,xlabel in [('retention','Controlled signal retention','Retained teacher energy'),
                              ('random_bases','Projection choice variability','Public projection basis'),
                              ('attack_budgets','Attack optimization effort','Ascent steps per restart')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        if kind in ('retention','random_bases'):
            for private in (False,True):
                rows=[r for r in data[kind] if r['private']==private]
                xs=[r['eta'] if kind=='retention' else r['projection_seed']-20261100 for r in rows]
                ax.plot(xs,[r['mean_gain'] for r in rows],marker='o',label='GDP4' if private else 'No noise')
                for x,r in zip(xs,rows):ax.scatter([x]*4,[v['gain'] for v in r['points']],alpha=.35,s=18)
            ax.axhline(0,color='black',lw=.8);ax.set_ylabel('AUROC difference from Full512')
        else:
            for private in (False,True):
                for family in ('full','random'):
                    rows=[r for r in data[kind] if r['private']==private and r['kind']==family]
                    ax.plot([r['steps'] for r in rows],[r['mean_auroc'] for r in rows],marker='o',
                            label=f'{family}, '+('GDP4' if private else 'no noise'))
            ax.set_ylabel('Attacked AUROC')
        ax.set(title=title,xlabel=xlabel);ax.legend();fig.tight_layout()
        fig.savefig(args.output/f'{kind}.png',dpi=160);fig.savefig(args.output/f'{kind}.svg');plt.close(fig)
    (args.output/'plot-runtime.json').write_text(json.dumps({'matplotlib':importlib.metadata.version('matplotlib')},indent=2)+'\n')


if __name__=='__main__':main()
