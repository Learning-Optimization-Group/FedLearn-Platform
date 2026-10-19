import importlib.util
from pathlib import Path
import numpy as np
import pytest


def benchmark():
    path=Path(__file__).resolve().parents[2]/'research/benchmarks/breast_cancer.py'
    spec=importlib.util.spec_from_file_location('breast_cancer',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_duplicate_feature_groups_stay_in_one_partition_and_split_is_reproducible():
    x=np.repeat(np.arange(80,dtype='float32')[:,None],30,axis=1)
    x=np.repeat(x,2,axis=0);y=np.repeat(np.arange(80)%2,2)
    first=benchmark().partition_rows(x,y,seed=42)
    second=benchmark().partition_rows(x,y,seed=42)
    assert first==second
    assert sorted(i for rows in first.values() for i in rows)==list(range(len(x)))
    identities=[{x[i].tobytes() for i in rows} for rows in first.values()]
    assert all(not a & b for i,a in enumerate(identities) for b in identities[i+1:])
    assert all(set(y[rows])=={0,1} for rows in first.values())
    y[1]=1-y[0]
    with pytest.raises(ValueError,match='conflicting'):
        benchmark().partition_rows(x,y,seed=42)
