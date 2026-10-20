"""Curated numerical tables from the independently checked fixed NIH study."""

if not __debug__:
    raise RuntimeError("Verification requires Python assertions; rerun without -O or PYTHONOPTIMIZE.")

import argparse
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path
import statistics



def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summary_values(values):
    return (statistics.mean(values),min(values),max(values)) if values else (None,None,None)


def main(study,audit_path,output):
    study=Path(study).resolve();audit_path=Path(audit_path).resolve();output=Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('refusing existing table output')
    if output.resolve().is_relative_to(study):raise ValueError('table output must be outside immutable study')
    audit=json.loads(audit_path.read_text())
    assert sha(study/'summary.json')==audit['raw_summary_sha256']
    assert sha(study/'index.json')==audit['raw_index_sha256']
    assert sha(study/'protocol.json')==audit['raw_protocol_sha256']
    result=json.loads((study/'summary.json').read_text());protocol=json.loads((study/'protocol.json').read_text())
    assert audit['source_commit']==result['source_identity']['git_commit'] and audit['primary']==result['analysis']['primary']
    configs={cfg['name']:cfg for cfg in protocol['configs']};cases={c['name']:c for c in result['cases']}
    independent={c['name']:c for c in audit['cases']}
    assert len(configs)==len(cases)==360 and set(independent)=={n for n,c in cases.items() if c['status']=='qualified'}
    for name,cell in independent.items():
        for key,value in cell['final'].items():assert abs(cases[name]['analysis']['test'][key]-value)<=1e-10
    output.mkdir()
    registry={}
    def write(name,fields,rows):
        path=output/name
        with path.open('x',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields,extrasaction='raise',lineterminator='\n');writer.writeheader();writer.writerows(rows)
        registry[name]={'sha256':sha(path),'rows':len(rows),'fields':fields}
    def identity(cfg):
        condition='clean' if not cfg['behaviors'] else 'sign_flip' if cfg['behaviors'][0]['type']=='sign_flip_scale' else 'label_flip'
        privacy='none' if cfg['privacy']['mode']=='none' else f'eps{cfg["privacy"]["epsilon"]:g}'
        h=cfg['workload']['head'];head='linear' if h['type']=='linear' else f'mlp{h["hidden"]}'
        return {'family':cfg['name'].split('_',1)[0],'seed':cfg['seed'],'head':head,'clients':cfg['federation']['num_clients'],
            'clip':cfg['server']['clip_norm'],'condition':condition,'rule':cfg['server']['rule'],'privacy':privacy}
    ids={name:identity(cfg) for name,cfg in configs.items()}
    metrics=['auc','loss','ece','accuracy','balanced_accuracy','sensitivity','specificity','tp','tn','fp','fn']
    status_rows=[]
    for family in ['primary','head','cohort','clipping']:
        group=[c for name,c in cases.items() if ids[name]['family']==family]
        status_rows.append({'family':family,'planned':len(group),**{s:sum(c['status']==s for c in group) for s in ['qualified','failed','timeout','not_run']}})
    assert sum(r['planned'] for r in status_rows)==360
    write('status.csv',['family','planned','qualified','failed','timeout','not_run'],status_rows)
    every=[]
    for name,cfg in configs.items():
        c=cases[name]
        reason=''
        if c['status']=='timeout':reason='TIMEOUT'
        elif c['status']=='not_run':reason=c.get('reason') if c.get('reason') in ('source_integrity','global_budget_exhausted','pending') else 'UNRUN_PRIVATE_DETAILS_RETAINED'
        elif c['status']=='failed':
            reason='FAILED_PRIVATE_DETAILS_RETAINED'
            execution=study/'runs'/name/'execution.json'
            if execution.is_file():
                kind=json.loads(execution.read_text()).get('error_kind','')
                if isinstance(kind,str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}',kind):reason=kind
        # Raw exception messages can contain host paths/commands. Keep those private.
        row={'name':name,**ids[name],'status':c['status'],'reason':reason}
        if c['status']=='qualified':
            row.update({key:independent[name]['final'][key] for key in metrics})
            row.update(child_seconds=c['child_execution_seconds'],engine_completion_rss_mib=c['summary']['peak_rss_bytes']/2**20)
        every.append(row)
    write('all_360_outcomes.csv',['name','family','seed','head','clients','clip','condition','rule','privacy','status','reason',*metrics,'child_seconds','engine_completion_rss_mib'],every)
    evaluation=[]
    modes=['validation','test','threshold_validation','threshold_test','patient_weighted_validation','patient_weighted_test','test_ambiguous_excluded']
    for name,cfg in configs.items():
        c=cases[name]
        for mode in modes:
            row={'name':name,**ids[name],'status':c['status'],'evaluation_mode':mode}
            if c['status']=='qualified':
                a=c['analysis']
                if mode.startswith('threshold_'):
                    data=a['threshold'][mode.removeprefix('threshold_')];row['validation_selected_threshold']=a['validation_threshold']
                elif mode.startswith('patient_weighted_'):data=a['patient_weighted'][mode.removeprefix('patient_weighted_')]
                else:data=a[mode]
                row.update({key:data[key] for key in [*metrics,'n_images','weight_sum']})
            evaluation.append(row)
    assert len(evaluation)==2520
    write('all_360_evaluation_modes.csv',['name','family','seed','head','clients','clip','condition','rule','privacy','status','evaluation_mode','validation_selected_threshold',*metrics,'n_images','weight_sum'],evaluation)
    component_keys=[f'{condition}/{rule}/{privacy}' for condition,rule,privacy in itertools.product(['clean','sign_flip'],['mean','trimmed_mean'],['none','eps4'])]
    primary=[]
    for seed in audit['primary']['per_seed']:
        c=seed['components'];row={'seed':seed['seed'],'complete':seed['effect'] is not None,'effect':seed['effect'],'missing':';'.join(seed['missing'])}
        row.update({key:c.get(key) for key in component_keys})
        for privacy in ['none','eps4']:
            needed=[f'{condition}/{rule}/{privacy}' for condition,rule in itertools.product(['clean','sign_flip'],['mean','trimmed_mean'])]
            if all(key in c for key in needed):row[f'B_{privacy}']=(c[f'sign_flip/trimmed_mean/{privacy}']-c[f'sign_flip/mean/{privacy}'])-(c[f'clean/trimmed_mean/{privacy}']-c[f'clean/mean/{privacy}'])
        primary.append(row)
    write('primary_per_seed.csv',['seed','complete','effect','B_none','B_eps4','missing',*component_keys],primary)
    groups={}
    for name,cfg in configs.items():
        key=name.replace(f's{cfg["seed"]}','matched_seeds',1);groups.setdefault(key,[]).append(name)
    assert len(groups)==64
    grouped=[];sensitivities=[];costs=[];update_diagnostics=[]
    for key,names in groups.items():
        factor={k:v for k,v in ids[names[0]].items() if k!='seed'}
        available=[name for name in names if name in independent]
        row={'group':key,**factor,'planned':len(names),'qualified':len(available)}
        for metric in metrics:
            mean,low,high=summary_values([independent[n]['final'][metric] for n in available])
            row.update({f'{metric}_mean':mean,f'{metric}_min':low,f'{metric}_max':high})
        grouped.append(row)
        row={'group':key,**factor,'planned':len(names),'qualified':len(available)}
        for label,get in [('patient_minus_image_auc',lambda a:a['patient_weighted']['test']['auc']-a['test']['auc']),
            ('mask_minus_primary_auc',lambda a:a['test_ambiguous_excluded']['auc']-a['test']['auc']),
            ('validation_threshold',lambda a:a['validation_threshold']),
            ('default_test_recall',lambda a:a['test']['sensitivity']),
            ('threshold_test_recall',lambda a:a['threshold']['test']['sensitivity']),
            ('threshold_test_specificity',lambda a:a['threshold']['test']['specificity']),
            ('threshold_test_balanced_accuracy',lambda a:a['threshold']['test']['balanced_accuracy']),
            ('threshold_validation_recall',lambda a:a['threshold']['validation']['sensitivity'])]:
            mean,low,high=summary_values([get(cases[n]['analysis']) for n in available])
            row.update({label+'_mean':mean,label+'_min':low,label+'_max':high})
        sensitivities.append(row)
        row={'group':key,**factor,'planned':len(names),'qualified':len(available)}
        for label,get in [('child_seconds',lambda c:c['child_execution_seconds']),
            ('parent_control_seconds',lambda c:c['parent_control_seconds']),
            ('core_seconds',lambda c:c['summary']['core_execution_seconds']),
            ('federation_loop_seconds',lambda c:c['summary']['federation_loop_seconds']),
            ('artifact_seconds',lambda c:c['summary']['artifact_execution_seconds']),
            ('engine_completion_rss_mib',lambda c:c['summary']['peak_rss_bytes']/2**20),
            ('modeled_upload_mb',lambda c:c['summary']['totals']['up_bytes']/1e6),
            ('modeled_download_mb',lambda c:c['summary']['totals']['down_bytes']/1e6),
            ('client_matmul_flops',lambda c:c['summary']['totals']['client_flops'])]:
            mean,low,high=summary_values([get(cases[n]) for n in available]);row.update({label+'_mean':mean,label+'_min':low,label+'_max':high})
        costs.append(row)
        row={'group':key,**factor,'planned':len(names),'qualified':len(available)}
        for label,field in [('clip_rate','clip_rate_mean'),('shrink_factor','mean_shrink_mean'),('noise_to_signal','noise_to_signal_mean'),('noise_rms_per_round','noise_rms_per_round')]:
            values=[independent[n]['source_bound_update_diagnostics'][field] for n in available if independent[n]['source_bound_update_diagnostics'][field] is not None]
            mean,low,high=summary_values(values);row.update({label+'_mean':mean,label+'_min':low,label+'_max':high,label+'_n':len(values)})
        update_diagnostics.append(row)
    common=['group','family','head','clients','clip','condition','rule','privacy','planned','qualified']
    write('all_64_recipe_groups.csv',common+[f'{m}_{s}' for m in metrics for s in ['mean','min','max']],grouped)
    write('threshold_patient_content_sensitivity.csv',common+[f'{m}_{s}' for m in ['patient_minus_image_auc','mask_minus_primary_auc','validation_threshold','default_test_recall','threshold_test_recall','threshold_test_specificity','threshold_test_balanced_accuracy','threshold_validation_recall'] for s in ['mean','min','max']],sensitivities)
    write('source_bound_update_diagnostics.csv',common+[f'{m}_{stat}' for m in ['clip_rate','shrink_factor','noise_to_signal','noise_rms_per_round'] for stat in ['mean','min','max','n']],update_diagnostics)
    write('observed_and_modeled_costs.csv',common+[f'{m}_{s}' for m in ['child_seconds','parent_control_seconds','core_seconds','federation_loop_seconds','artifact_seconds','engine_completion_rss_mib','modeled_upload_mb','modeled_download_mb','client_matmul_flops'] for s in ['mean','min','max']],costs)
    def name(family,seed,head,n,clip,condition,rule,privacy):return f'{family}_s{seed}_{head}_n{n}_c{clip}_{condition}_{rule}_{privacy}'
    def paired_row(factors,pairs):
        values=[];private=[];control=[]
        for high,low in pairs:
            if high in independent and low in independent:
                hi,lo=independent[high]['final']['auc'],independent[low]['final']['auc'];values.append(hi-lo);private.append(hi);control.append(lo)
        row={**factors,'planned_pairs':len(pairs),'qualified_pairs':len(values)}
        for label,v in [('difference',values),('treatment_auc',private),('control_auc',control)]:
            mean,low,high=summary_values(v);row.update({label+'_mean':mean,label+'_min':low,label+'_max':high})
        return row
    head_rows=[]
    for head,condition,rule in itertools.product(['linear','mlp8','mlp128','mlp512'],['clean','sign_flip'],['mean','trimmed_mean']):
        family='primary' if head=='linear' else 'head'
        pairs=[(name(family,s,head,80,'005',condition,rule,'eps4'),name(family,s,head,80,'005',condition,rule,'none')) for s in range(20261101,20261104)]
        head_rows.append(paired_row({'head':head,'condition':condition,'rule':rule},pairs))
    pair_fields=['planned_pairs','qualified_pairs']+[f'{m}_{s}' for m in ['difference','treatment_auc','control_auc'] for s in ['mean','min','max']]
    write('head_private_minus_no_dp.csv',['head','condition','rule',*pair_fields],head_rows)
    clipping=[]
    for condition,rule,privacy in itertools.product(['clean','sign_flip'],['mean','trimmed_mean'],['none','eps4']):
        pairs=[(name('clipping',s,'linear',80,'04',condition,rule,privacy),name('primary',s,'linear',80,'005',condition,rule,privacy)) for s in range(20261101,20261104)]
        clipping.append(paired_row({'condition':condition,'rule':rule,'privacy':privacy},pairs))
    write('clipping_high_minus_low.csv',['condition','rule','privacy',*pair_fields],clipping)
    attack=[]
    for condition,rule,privacy in itertools.product(['sign_flip','label_flip'],['mean','trimmed_mean'],['none','eps1','eps4','eps8']):
        pairs=[(name('primary',s,'linear',80,'005',condition,rule,privacy),name('primary',s,'linear',80,'005','clean',rule,privacy)) for s in range(20261101,20261111)]
        attack.append(paired_row({'condition':condition,'rule':rule,'privacy':privacy},pairs))
    write('attack_minus_clean.csv',['condition','rule','privacy',*pair_fields],attack)
    cohort=[]
    for n,rule,privacy in itertools.product([10,80,1000],['mean','trimmed_mean'],['none','eps4']):
        family='primary' if n==80 else 'cohort';names=[name(family,s,'linear',n,'005','clean',rule,privacy) for s in range(20261101,20261104)]
        values=[independent[n]['final']['auc'] for n in names if n in independent];mean,low,high=summary_values(values)
        cohort.append({'clients':n,'rule':rule,'privacy':privacy,'planned':3,'qualified':len(values),'auc_mean':mean,'auc_min':low,'auc_max':high})
    write('cohort_matched_first_three_seeds.csv',['clients','rule','privacy','planned','qualified','auc_mean','auc_min','auc_max'],cohort)
    payload={'version':'nih-phase2-tables-v1','source_commit':audit['source_commit'],'independent_audit_sha256':sha(audit_path),'exporter_sha256':sha(Path(__file__)),
        'primary':audit['primary'],'statuses':result['analysis']['statuses'],'files':registry,
        'notes':['Only the prespecified complete-ten-seed primary contrast has a formal interval.','All other ranges and paired differences are descriptive, with qualified/planned counts retained.','Head/cohort/clipping sensitivities use first three matched seeds and reuse primary controls.','All360evaluationmodes retains confusion counts, including fractional inverse-patient weights; threshold loss/ECE remain argmax-confidence diagnostics.','Update diagnostics are source-bound reported summaries, not an independent reconstruction of raw updates. Noise RMS is sigma times sqrt(d), not mean Gaussian vector length.','Costs distinguish observed scope from modeled payload/FLOPs; RSS is engine-completion process high-water.','Research aggregates do not establish clinical, patient-level or whole-matrix privacy protection.']}
    (output/'table-registry.json').write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'tables':len(registry),'rows':sum(x['rows'] for x in registry.values())}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.study,args.audit,args.output)
