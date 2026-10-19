"""Prepare bounded, duplicate-separated Wisconsin Diagnostic Breast Cancer CSVs."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from fedlearn.tree.schema import BREAST_CANCER_FEATURES


def partition_rows(features, labels, *, seed):
    x=np.asarray(features,dtype='float32');y=np.asarray(labels)
    if x.ndim!=2 or x.shape[1]!=30 or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isin(y,[0,1]).all():
        raise ValueError('requires finite thirty-feature binary rows')
    groups={};targets={}
    for i,row in enumerate(x):
        identity=hashlib.sha256(row.astype('<f4').tobytes()).hexdigest()
        if identity in targets and targets[identity]!=int(y[i]):
            raise ValueError('identical features have conflicting labels')
        groups.setdefault(identity,[]).append(i);targets[identity]=int(y[i])
    keys=sorted(groups)
    if any(sum(targets[k]==label for k in keys)<20 for label in (0,1)):
        raise ValueError('requires at least twenty distinct groups per class')
    remaining,test=train_test_split(keys,test_size=.2,random_state=seed,stratify=[targets[k] for k in keys])
    train,validation=train_test_split(remaining,test_size=.2,random_state=seed,stratify=[targets[k] for k in remaining])
    first,second=train_test_split(train,test_size=.5,random_state=seed,stratify=[targets[k] for k in train])
    result={name:sorted(i for k in selected for i in groups[k]) for name,selected in
            [('client-0',first),('client-1',second),('validation',validation),('test',test)]}
    if sorted(i for rows in result.values() for i in rows)!=list(range(len(x))):
        raise ValueError('partition coverage differs')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('output already exists')
    from sklearn.datasets import load_breast_cancer
    from sklearn.datasets import _base
    dataset=load_breast_cancer()
    if dataset.data.shape!=(569,30) or list(dataset.target_names)!=['malignant','benign']:
        raise ValueError('unexpected diagnostic dataset schema')
    # The source orders mean measurements, measurement errors, then worst measurements.
    measures=('radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','fractal dimension')
    source_names=[prefix+' '+name for prefix in ('mean','worst') for name in measures]
    expected=source_names[:10]+[name+' error' for name in measures]+source_names[10:]
    if list(dataset.feature_names)!=expected: raise ValueError('source feature order differs')
    x=dataset.data.astype('float32');y=(1-dataset.target).astype('int64')
    partitions=partition_rows(x,y,seed=42)
    source=Path(_base.__file__).parent/'data/breast_cancer.csv'
    raw_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    args.out.mkdir(parents=True)
    report=dict(dataset='Wisconsin Diagnostic Breast Cancer',source='scikit-learn bundled UCI diagnostic dataset',
                source_url='https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html',
                sklearn_version=version('scikit-learn'),source_csv_sha256=raw_sha,records=569,
                feature_names=list(BREAST_CANCER_FEATURES),source_feature_names=list(dataset.feature_names),
                label_mapping={'0':'benign','1':'malignant'},source_target_mapping={'0':'malignant','1':'benign'},
                conversion='raw numeric measurements cast to float32; no fitted preprocessing',seed=42,
                split='stratified duplicate-feature groups: 20% test, 20% of remainder validation, remaining training split in half',
                patient_ids_available=False,patient_disjointness_claimed=False,pairwise_float32_feature_group_overlap=0,partitions={})
    for name,rows in partitions.items():
        path=args.out/(name+'.csv')
        with path.open('w') as file:
            file.write(','.join((*BREAST_CANCER_FEATURES,'label'))+'\n')
            for i in rows: file.write(','.join(str(v) for v in x[i])+','+str(y[i])+'\n')
        report['partitions'][name]=dict(indices=rows,records=len(rows),label_counts=np.bincount(y[rows],minlength=2).tolist(),
             csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),snapshot_sha256=hashlib.sha256(x[rows].astype('<f4').tobytes()+y[rows].astype('<f4').tobytes()).hexdigest())
    (args.out/'partition-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:entry['records'] for name,entry in report['partitions'].items()}),flush=True)


if __name__=='__main__': main()
