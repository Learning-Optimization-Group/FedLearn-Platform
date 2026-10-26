"""Render descriptive paired CPU timing from the public benchmark export."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(source,output):
    data=json.loads(Path(source).read_text())
    cells={row['name']:row for row in data['cells'] if row['status']=='complete'}
    methods=('mean512','projected128');counts=(3,6,24,80)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for method,color,offset in zip(methods,('#176b87','#b24f2a'),(-.08,.08)):
        for axis,key in zip(axes,('elapsed_speedup','steady_round_speedup')):
            values=[]
            for n in counts:
                values.append([p[key] for p in data['placement_pairs']
                    if cells[p['three_host_cell']]['stage']=='systems'
                    and cells[p['three_host_cell']]['method']==method
                    and cells[p['three_host_cell']]['num_clients']==n])
            if any(len(v)!=3 for v in values):raise ValueError('plot requires every prespecified three-seed placement pair')
            values=np.asarray(values)
            med=np.median(values,axis=1);x=np.arange(4)+offset
            axis.errorbar(x,med,yerr=np.array([med-values.min(1),values.max(1)-med]),
                          fmt='o-',capsize=4,color=color,label=method,linewidth=1.4)
            for column in range(3):axis.scatter(x,values[:,column],s=12,color=color,alpha=.45)
    for axis,title in zip(axes,('Whole-cell elapsed time','Steady-round time')):
        axis.axhline(1,color='#555555',linewidth=.8,linestyle='--')
        axis.set(xticks=np.arange(4),xticklabels=counts,xlabel='Logical clients',ylabel='Mac-only / three-host time',title=title)
        axis.grid(axis='y',alpha=.2);axis.legend(frameon=False)
    fig.suptitle('CPU training on three physical hosts: paired timing ratios',fontsize=13)
    fig.savefig(output/'cpu-placement-speedup.png')
    plt.close(fig)
    return output/'cpu-placement-speedup.png'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(render(args.source,args.output))
