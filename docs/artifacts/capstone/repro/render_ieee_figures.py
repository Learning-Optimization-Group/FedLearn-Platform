"""Column-sized figures from frozen public aggregate records; no model fitting."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=[args.artifacts/name for name in ('effusion-finalists-2026-10-25.json',
           'multihost-results-2026-10-26.json','projection-study-results-2026-10-28.json')]
    medical,cpu,synthetic=[json.loads(p.read_text()) for p in paths]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
                         'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    colors=['#1764ab','#c74d13']

    def save(fig,name):
        fig.savefig(args.output/f'{name}.pdf')
        fig.savefig(args.output/f'{name}.png',dpi=220)
        plt.close(fig)

    methods=['mean512','projected128','trim16','center_quarter'];xs=np.arange(4)
    medical_low=min(r['metrics']['auroc']['mean']-r['metrics']['auroc']['sample_sd'] for r in medical['arms'])-.02
    medical_high=max(.85,max(r['metrics']['auroc']['mean']+r['metrics']['auroc']['sample_sd'] for r in medical['arms'])+.02)
    fig,axes=plt.subplots(2,1,figsize=(3.5,3.65),sharex=True)
    for ax,private in zip(axes,(False,True)):
        for offset,attack,color,label in [(-.11,'clean',colors[0],'Clean'),(.11,'owned_pgd',colors[1],'Owned PGD')]:
            rows=[next(r for r in medical['arms'] if r['method']==m and r['attack']==attack and
                       r['id'].split(':')[1]==('private' if private else 'nonprivate')) for m in methods]
            values=[r['metrics']['auroc'] for r in rows]
            ax.errorbar(xs+offset,[v['mean'] for v in values],yerr=[v['sample_sd'] for v in values],
                        fmt='o',ms=3,capsize=2,color=color,label=label,lw=.9)
        ax.axhline(.5,color='.6',ls='--',lw=.7);ax.set_ylabel('AUROC')
        ax.set_title('GDP4' if private else 'No noise');ax.set_ylim(medical_low,medical_high)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.57,.995),ncol=2,frameon=False)
    axes[-1].set_xticks(xs,['Mean\n512','Project\n128','Trim\n16','Center\n0.25 C'])
    fig.subplots_adjust(left=.16,right=.98,bottom=.12,top=.86,hspace=.35)
    save(fig,'medical-policies')

    fig,axes=plt.subplots(2,1,figsize=(3.5,3.35),gridspec_kw={'height_ratios':[1.1,1]})
    primary=medical['primary'];seed=primary['seed_interval'];patient=primary['patient_interval']
    vals=seed['seed_effects'];order=sorted(vals,key=int)
    axes[0].plot(np.arange(1,13),[vals[s] for s in order],'o',ms=3,color=colors[0])
    axes[0].axhline(0,color='.2',lw=.7);axes[0].axhline(.01,color=colors[1],ls='--',lw=.8)
    axes[0].set(xlabel='Training seed index (13001–13012)',ylabel='Paired AUROC gain')
    axes[0].set_xticks([1,4,8,12]);axes[0].set_title('Medical private-clean projection gain')
    for y,ci,color in [(1,seed,colors[0]),(0,patient,colors[1])]:
        axes[1].errorbar(ci['point'],y,xerr=[[ci['point']-ci['ci_low']],[ci['ci_high']-ci['point']]],
                         fmt='o',ms=3,capsize=3,color=color,lw=1.1)
    axes[1].set_yticks([0,1],['Fixed-head\npatient bootstrap','Across\ntraining seeds'])
    axes[1].axvline(.01,color='.6',ls='--',lw=.7)
    axes[1].set(xlim=(0,.033),ylim=(-.6,1.6),xlabel='Paired AUROC gain (separate 95% intervals)')
    fig.subplots_adjust(left=.28,right=.98,bottom=.13,top=.92,hspace=.72)
    save(fig,'medical-primary')

    fig,axes=plt.subplots(2,1,figsize=(3.5,3.55),sharex=True)
    for ax,key,title in zip(axes,['elapsed_speedup','steady_round_speedup'],['Whole-cell time','Steady-round time']):
        for method,color,offset in [('mean512',colors[0],-.045),('projected128',colors[1],.045)]:
            groups=[]
            for n in (3,6,24,80):
                values=[r[key] for r in cpu['placement_pairs'] if r['three_host_cell'].startswith('systems_')
                        and re.search(rf'_n{n:03}_{method}_',r['three_host_cell'])]
                assert len(values)==3
                groups.append(values)
            med=np.median(groups,axis=1);lo=np.min(groups,axis=1);hi=np.max(groups,axis=1)
            ax.errorbar(np.arange(4)+offset,med,yerr=[med-lo,hi-med],fmt='o-',ms=3,lw=.9,capsize=2,
                        label=method,color=color)
        ax.axhline(1,color='.4',ls='--',lw=.7);ax.set_ylabel('Mac-only / three-host time');ax.set_title(title)
    axes[0].set_ylim(.25,1.05);axes[0].legend(loc='upper right',frameon=False)
    axes[-1].set_xticks(range(4),['3','6','24','80']);axes[-1].set_xlabel('Logical clients')
    fig.subplots_adjust(left=.19,right=.98,bottom=.12,top=.92,hspace=.35)
    save(fig,'cpu-placement')

    comparisons=synthetic['comparisons']
    for kind,name,xlabel,title in [('retention','signal-retention','Retained teacher energy','Controlled signal retention'),
                                  ('random_bases','random-bases','Public projection basis','Projection choice variability')]:
        fig,ax=plt.subplots(figsize=(3.5,2.65))
        for private,color in zip((False,True),colors):
            rows=[r for r in comparisons[kind] if r['private']==private]
            x=[r['eta'] if kind=='retention' else r['projection_seed']-20261100 for r in rows]
            ax.plot(x,[r['mean_gain'] for r in rows],'-o',ms=3,color=color,lw=1,label='GDP4' if private else 'No noise')
            for xpos,row in zip(x,rows):
                ax.scatter([xpos]*4,[p['gain'] for p in row['points']],color=color,alpha=.35,s=8)
        ax.axhline(0,color='.3',lw=.7);ax.set(xlabel=xlabel,ylabel='AUROC gain vs Full512',title=title)
        if kind=='random_bases':ax.set_xticks(range(1,6))
        else:ax.set_xticks([0,.25,.75,1.])
        ax.legend(loc='best',frameon=False);fig.subplots_adjust(left=.18,right=.97,bottom=.2,top=.88)
        save(fig,name)

    fig,ax=plt.subplots(figsize=(3.5,2.65))
    for private in (False,True):
        for family,color,style in [('full',colors[0],'-'),('random',colors[1],'--')]:
            rows=[r for r in comparisons['attack_budgets'] if r['private']==private and r['kind']==family]
            ax.plot([r['steps'] for r in rows],[r['mean_auroc'] for r in rows],style+'o',ms=3,lw=1,
                    color=color,alpha=.65 if private else 1,
                    label=('Full512' if family=='full' else 'Random128')+(', GDP4' if private else ', no noise'))
    ax.axhline(.5,color='.5',ls=':',lw=.7);ax.set_xticks([20,80,160])
    ax.set(xlabel='Ascent steps per restart',ylabel='Attacked AUROC',title='Attack optimization effort')
    ax.legend(loc='center',bbox_to_anchor=(.63,.64),frameon=False,fontsize=6.6)
    fig.subplots_adjust(left=.17,right=.97,bottom=.2,top=.88);save(fig,'attack-effort')
    receipt=dict(inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 matplotlib=importlib.metadata.version('matplotlib'),
                 scope='Layout-only redraws from frozen public aggregate records; no training or new estimation target.')
    (args.output/'figure-inputs.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
