"""Scientific exports from independently reconciled final NIH study evidence."""

if not __debug__:
    raise RuntimeError("Verification requires Python assertions; rerun without -O or PYTHONOPTIMIZE.")

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import textwrap



def main(study,audit_path,output):
    study=Path(study).resolve();audit_path=Path(audit_path).resolve();target=Path(output)
    if target.exists() or target.is_symlink():raise FileExistsError('refusing existing figure output')
    if target.resolve().is_relative_to(study):raise ValueError('figure output must be outside immutable study')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    audit=json.loads(audit_path.read_text())
    summary=json.loads((study/'summary.json').read_text())
    assert hashlib.sha256((study/'summary.json').read_bytes()).hexdigest()==audit['raw_summary_sha256']
    assert audit['source_commit']==summary['source_identity']['git_commit']
    assert audit['qualified']==sum(c['status']=='qualified' for c in summary['cases'])
    assert audit['primary']==summary['analysis']['primary']
    # Reconcile the exact plotted default metrics with the separately computed audit.
    checked={c['name']:c for c in audit['cases']}
    actual={c['name']:c for c in summary['cases'] if c['status']=='qualified'}
    assert set(checked)==set(actual)
    for name,cell in actual.items():
        for key in ('auc','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity'):
            assert abs(cell['analysis']['test'][key]-checked[name]['final'][key])<=1e-10
    target.mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9,'legend.fontsize':7.5,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180,'svg.hashsalt':'nih-phase2-study-v1'})
    colors={'mean':'#176B87','trimmed_mean':'#AC4032'}
    labels={'mean':'Clipped mean','trimmed_mean':'Trimmed mean'}
    figure_files=[];plotted=[]
    def save(fig,name):
        assert name==current_figure
        # Reserve caption space before rendering at a report-sized width.
        fig.set_size_inches(7.2,3.8)
        for label in fig.texts:
            if label.get_position()[1] < 0:
                label.set_position((.015,.02))
                label.set_text(textwrap.fill(label.get_text(),width=135))
                label.set_fontsize(7)
                label.set_verticalalignment('bottom')
        if fig._suptitle is not None:
            fig._suptitle.set_position((.5,.98))
            fig._suptitle.set_fontsize(9.5)
        fig.tight_layout(rect=(0,.10,1,1))
        for axes in fig.axes:
            legend=axes.get_legend()
            if legend is not None:
                legend.set_frame_on(True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('none')
                legend.get_frame().set_alpha(.95)
        fig.savefig(target/f'{name}.png',metadata={'Software':None},bbox_inches='tight')
        fig.savefig(target/f'{name}.svg',metadata={'Creator':None,'Date':None},bbox_inches='tight')
        figure_files.extend([f'{name}.png',f'{name}.svg']);plt.close(fig)
    def key(family,seed,head,n,clip,condition,rule,privacy):
        return f'{family}_s{seed}_{head}_n{n}_c{clip}_{condition}_{rule}_{privacy}'
    def observations(names):
        cells=[actual[name] for name in names if name in actual]
        return cells,[checked[c['name']]['final']['auc'] for c in cells]
    def draw(ax,x,cells,values,color,label=None):
        if not values:return
        offsets=np.linspace(-.035,.035,len(values)) if len(values)>1 else np.array([0.])
        ax.scatter(x+offsets,values,color=color,alpha=.45,s=15,zorder=2)
        mean=statistics.mean(values)
        ax.errorbar(x,mean,yerr=[[mean-min(values)],[max(values)-mean]],fmt='o',color=color,capsize=3,label=label,zorder=3)
        for cell,value in zip(cells,values):plotted.append({'figure':current_figure,'figure_value':'test_auc','name':cell['name'],'value':value})
    # Every descriptive primary endpoint: same fixed seeds, means and observed ranges.
    current_figure='nih-phase2-primary-components'
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.3),sharey=True)
    for ax,condition,title in zip(axes,['clean','sign_flip','label_flip'],['Clean','Own-update sign flip','Own-data label flip']):
        for j,rule in enumerate(['mean','trimmed_mean']):
            for i,privacy in enumerate(['none','eps1','eps4','eps8']):
                names=[key('primary',seed,'linear',80,'005',condition,rule,privacy) for seed in range(20261101,20261111)]
                cells,values=observations(names)
                draw(ax,i+(j-.5)*.16,cells,values,colors[rule],labels[rule] if i==0 else None)
        ax.set_xticks(range(4),['No DP','ε1','ε4','ε8']);ax.set_title(title);ax.set_xlabel('Privacy setting (per run)')
        ax.axhline(.5,color='#666666',linestyle=':',linewidth=1)
    axes[0].set_ylabel('Final official-test image AUROC');axes[-1].legend(frameon=False,loc='best')
    fig.suptitle('Primary endpoint components: all qualified fixed seeds',y=1.02,fontweight='bold')
    fig.text(.01,-.015,'Points: individual seeds. Large points and bars: mean and observed range; these bars are not confidence intervals. Missing cells remain in the study ledger.',fontsize=8)
    save(fig,'nih-phase2-primary-components')
    primary=audit['primary'];rows=primary['per_seed']
    current_figure='nih-phase2-primary-contrast'
    fig,ax=plt.subplots(figsize=(8,4.4))
    for index,row in enumerate(rows):
        if row['effect'] is None:
            ax.text(index,0,'missing',rotation=90,va='bottom',ha='center',fontsize=8,color='#777777')
        else:ax.scatter(index,row['effect'],s=45,color='#176B87');plotted.append({'figure':current_figure,'figure_value':'primary_D','seed':row['seed'],'value':row['effect']})
    ax.axhline(0,color='#555555',linewidth=1)
    if primary['mean'] is not None:
        ax.axhline(primary['mean'],color='#AC4032',label='Mean of complete seed effects')
    if primary['ci95'] is not None:
        low,high=primary['ci95'];ax.axhspan(low,high,color='#AC4032',alpha=.12,label='Prespecified paired 95% t interval (df9)')
    ax.set_xticks(range(10),[str(r['seed'])[-2:] for r in rows]);ax.set_xlabel('Matched seed suffix (202611xx)')
    ax.set_ylabel('D: DP change in attack-adjusted\ntrimming advantage (AUROC)')
    ax.set_title(f'Primary contrast: {primary["n_complete"]}/10 complete seed effects',fontweight='bold');ax.legend(frameon=False,fontsize=8)
    fig.text(.01,-.025,'Interval describes conditional simulation-seed variation on fixed cohorts; no clinical-population coverage. A positive D does not imply higher absolute private utility.',fontsize=8)
    save(fig,'nih-phase2-primary-contrast')
    # Capacity compares the three prespecified sensitivity seeds and matched baselines.
    current_figure='nih-phase2-head-privacy-penalty'
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    heads=[('linear',1026),('mlp8',4122),('mlp128',65922),('mlp512',263682)]
    for ax,condition,title in zip(axes,['clean','sign_flip'],['Clean','Own-update sign flip']):
        for j,rule in enumerate(['mean','trimmed_mean']):
            for i,(head,parameters) in enumerate(heads):
                diffs=[];members=[]
                for seed in range(20261101,20261104):
                    family='primary' if head=='linear' else 'head'
                    names=[key(family,seed,head,80,'005',condition,rule,p) for p in ['none','eps4']]
                    if all(n in checked for n in names):
                        delta=checked[names[1]]['final']['auc']-checked[names[0]]['final']['auc'];diffs.append(delta)
                        members.append({'name':names[1]});plotted.append({'figure':current_figure,'figure_value':'head_private_minus_control','name':names[1],'control':names[0],'value':delta})
                if diffs:
                    mean=statistics.mean(diffs);x=i+(j-.5)*.16
                    ax.errorbar(x,mean,yerr=[[mean-min(diffs)],[max(diffs)-mean]],color=colors[rule],fmt='o',capsize=3,label=labels[rule] if i==0 else None)
                    ax.scatter(x+np.linspace(-.025,.025,len(diffs)),diffs,s=15,color=colors[rule],alpha=.5)
        ax.set_xticks(range(4),[f'{h}\n{n:,}' for h,n in heads]);ax.axhline(0,color='#555555',linewidth=1)
        ax.set_title(title);ax.set_xlabel('Head / trainable parameters')
    axes[0].set_ylabel('Matched ε4 minus no-DP final test AUROC');axes[-1].legend(frameon=False)
    fig.suptitle('Head-size sensitivity: matched privacy penalties',y=1.02,fontweight='bold')
    fig.text(.01,-.025,'First three fixed seeds only; mean and observed range, descriptive. Capacity and local optimization change with width; this does not isolate a pure dimensional effect.',fontsize=8)
    save(fig,'nih-phase2-head-privacy-penalty')
    # All modeled communication and observed child costs, no physical-network implication.
    current_figure='nih-phase2-head-costs'
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax in axes:ax.set_xscale('log')
    for head,_ in heads:
        cells=[c for n,c in actual.items() if ('_clean_mean_none' in n) and (f'_{head}_n80_c005_' in n) and c['name'].split('_s',1)[1].split('_',1)[0] in ['20261101','20261102','20261103']]
        for cell in cells:
            p=cell['summary']['parameters'];byte=cell['summary']['totals']['up_bytes']
            axes[0].scatter(p,cell['child_execution_seconds'],color='#176B87',s=25)
            axes[1].scatter(p,byte/1e6,color='#176B87',s=25)
            plotted.append({'figure':current_figure,'figure_value':'head_clean_mean_cost','name':cell['name'],'parameters':p,'child_seconds':cell['child_execution_seconds'],'up_megabytes':byte/1e6})
    axes[0].set_xlabel('Trainable parameters');axes[0].set_ylabel('Observed child process seconds');axes[0].set_title('CPU time: startup through artifact export')
    axes[1].set_xlabel('Trainable parameters');axes[1].set_ylabel('Modeled upload MB across all rounds/clients');axes[1].set_title('Modeled payload, not network traffic')
    fig.text(.01,-.025,'Clean no-DP mean, first three seeds; fixed CPU recipe/order. Timing includes startup/export; memory checks sample engine-completion high-water RSS.',fontsize=8)
    save(fig,'nih-phase2-head-costs')
    current_figure='nih-phase2-cohort-sensitivity'
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,privacy in zip(axes,['none','eps4']):
        for j,rule in enumerate(['mean','trimmed_mean']):
            for i,n in enumerate([10,80,1000]):
                family='primary' if n==80 else 'cohort'
                names=[key(family,seed,'linear',n,'005','clean',rule,privacy) for seed in range(20261101,20261104)]
                cells,values=observations(names)
                draw(ax,i+(j-.5)*.16,cells,values,colors[rule],labels[rule] if i==0 else None)
        ax.set_xticks(range(3),['10','80','1,000']);ax.axhline(.5,color='#666666',linestyle=':',linewidth=1)
        ax.set_title('No DP' if privacy=='none' else 'Central ε4');ax.set_xlabel('Accepted simulated clients per round')
    axes[0].set_ylabel('Final official-test image AUROC');axes[-1].legend(frameon=False)
    fig.suptitle('Cohort-size sensitivity: clean linear head',y=1.02,fontweight='bold')
    fig.text(.01,-.025,'First three matched seeds; mean and observed range. Cohort size changes local data and minibatch trajectories as well as aggregate sensitivity.',fontsize=8)
    save(fig,'nih-phase2-cohort-sensitivity')
    current_figure='nih-phase2-clipping-sensitivity'
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
    for ax,condition,title in zip(axes,['clean','sign_flip'],['Clean','Own-update sign flip']):
        for j,rule in enumerate(['mean','trimmed_mean']):
            for i,privacy in enumerate(['none','eps4']):
                deltas=[]
                for seed in range(20261101,20261104):
                    low=key('primary',seed,'linear',80,'005',condition,rule,privacy)
                    high=key('clipping',seed,'linear',80,'04',condition,rule,privacy)
                    if low in checked and high in checked:
                        delta=checked[high]['final']['auc']-checked[low]['final']['auc'];deltas.append(delta)
                        plotted.append({'figure':current_figure,'figure_value':'clipping_high_minus_low','name':high,'control':low,'value':delta})
                if deltas:
                    mean=statistics.mean(deltas);x=i+(j-.5)*.16
                    ax.errorbar(x,mean,yerr=[[mean-min(deltas)],[max(deltas)-mean]],color=colors[rule],fmt='o',capsize=3,label=labels[rule] if i==0 else None)
                    ax.scatter(x+np.linspace(-.025,.025,len(deltas)),deltas,s=15,color=colors[rule],alpha=.5)
        ax.set_xticks(range(2),['No DP','ε4']);ax.axhline(0,color='#555555',linewidth=1);ax.set_title(title)
    axes[0].set_ylabel('C0.4 minus C0.05 final test AUROC');axes[-1].legend(frameon=False)
    fig.suptitle('Clipping sensitivity: matched signal/noise trade-off',y=1.02,fontweight='bold')
    fig.text(.01,-.025,'First three fixed seeds; mean and observed range, descriptive. Changing C changes signal truncation and calibrated noise together.',fontsize=8)
    save(fig,'nih-phase2-clipping-sensitivity')
    # Machine-readable registry is reconciled again before the public files are copied.
    registry={'renderer_versions':{'matplotlib':matplotlib.__version__,'numpy':np.__version__},'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'independent_audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),'source_commit':audit['source_commit'],'qualified':audit['qualified'],'primary_complete':primary['complete'],
        'plotted_values':plotted,'files':{name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in figure_files}}
    (target/'figure-values.json').write_text(json.dumps(registry,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'output':str(target),'figures':len(figure_files)//2,'plotted_values':len(plotted)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.study,args.audit,args.output)
