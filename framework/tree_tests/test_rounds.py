import hashlib
import json
import uuid

import numpy as np
import pytest

from fedlearn.communication.generated import execution_contract_pb2 as c
from fedlearn.communication.generated import tree_pb2 as p
from fedlearn.tree.contract import validate_contract
from fedlearn.tree.coordinator import TreeCoordinator
from fedlearn.tree.model import initial_model, load_model, train_delta


def contract():
    initial = bytes(load_model(initial_model()).save_raw(raw_format="ubj"))
    value = c.ExecutionContract(contract_version=2, min_client_protocol_version=3,
        run_id=str(uuid.uuid4()), project_id=str(uuid.uuid4()), recipe=c.RECIPE_XGBOOST,
        strategy=c.STRATEGY_XGB_BAGGING, num_rounds=2, clients_per_round=2,
        partitioning=c.PARTITIONING_LOCAL, seed=42,
        round=c.RoundPolicy(timeout_ms=1000, one_accepted_update_per_round=True,
                            max_transient_retries=3, retry_backoff_ms=100),
        security=c.SecurityPolicy(transport=c.TRANSPORT_PLAINTEXT_DEV,
            client_auth=c.CLIENT_AUTH_DISABLED_DEV, secure_aggregation=c.SECAGG_NONE),
        tree_training=c.TreeTraining(xgboost_version="3.4.1", objective="binary:logistic",
            booster="gbtree", feature_names=["f0","f1","f2","f3"], classes=["0","1"],
            base_score=.5, learning_rate=.3, max_depth=3, local_boost_rounds=1,
            tree_method="hist", initial_model_sha256=hashlib.sha256(initial).hexdigest(),
            input_dtype="float32", missing_policy="reject", contribution_codec="xgboost-json-v1",
            data_source=c.DATA_SOURCE_LOCAL_SNAPSHOT))
    return value, initial


def request(coordinator, client, round_number=1, **overrides):
    x = np.random.default_rng(42).normal(size=(32,4)).astype("float32")
    delta = train_delta(coordinator.parent, x, (x[:,0]>0).astype("float32"), cohort_size=2, seed=1)
    fields = dict(client_id=client, run_id=coordinator.contract.run_id,
        contract_id=coordinator.contract_id, server_round=round_number,
        parent_sha256=hashlib.sha256(coordinator.parent).hexdigest(),
        delta_sha256=hashlib.sha256(delta).hexdigest(), new_trees_json=delta, num_examples=32)
    fields.update(overrides)
    return p.TreeContribution(**fields)


def test_fixed_cohort_identity_and_exact_retries():
    value, initial = contract()
    coordinator = TreeCoordinator(value, initial)
    coordinator.register("a", partition=0)
    with pytest.raises(ValueError): coordinator.register("fake", partition=0)
    coordinator.register("b", partition=1)
    with pytest.raises(ValueError): coordinator.register("c", partition=2)
    a, b = request(coordinator,"a"), request(coordinator,"b")
    assert coordinator.submit(a) == p.TreeContributionAck.ACCEPTED
    assert coordinator.submit(a) == p.TreeContributionAck.ALREADY_ACCEPTED
    conflicting = p.TreeContribution(); conflicting.CopyFrom(a); conflicting.num_examples=31
    with pytest.raises(ValueError): coordinator.submit(conflicting)
    assert coordinator.submit(b) == p.TreeContributionAck.ACCEPTED
    assert coordinator.server_round == 2
    assert coordinator.submit(a) == p.TreeContributionAck.ALREADY_ACCEPTED
    assert load_model(coordinator.parent).num_boosted_rounds() == 2


@pytest.mark.parametrize("override", [
    {"server_round":0}, {"server_round":2}, {"run_id":str(uuid.uuid4())},
    {"contract_id":"0"*64}, {"parent_sha256":"0"*64}, {"delta_sha256":"0"*64},
    {"num_examples":0}, {"num_examples":1_000_001}, {"client_id":"unknown"},
])
def test_wrong_round_parent_schema_identity_and_counts_cannot_mutate_model(override):
    value, initial = contract(); coordinator = TreeCoordinator(value, initial)
    coordinator.register("a",partition=0); coordinator.register("b",partition=1)
    before=coordinator.parent
    with pytest.raises(ValueError): coordinator.submit(request(coordinator,"a",**override))
    assert coordinator.parent == before and coordinator.accepted_count == 0


def test_timeout_does_not_aggregate_partial_cohort():
    value, initial = contract(); now=[100.0]
    coordinator = TreeCoordinator(value, initial, clock=lambda:now[0])
    coordinator.register("a",partition=0); coordinator.register("b",partition=1)
    coordinator.submit(request(coordinator,"a")); before=coordinator.parent
    now[0]+=2
    with pytest.raises(TimeoutError): coordinator.check_deadline()
    assert coordinator.parent==before and coordinator.state==p.TreeModel.FAILED


@pytest.mark.parametrize("field,value", [
    ("contract_version",1), ("min_client_protocol_version",2), ("partitioning",c.PARTITIONING_SHARDED),
    ("recipe",c.RECIPE_MLP), ("strategy",c.STRATEGY_FEDAVG), ("clients_per_round",0),
])
def test_contract_refuses_unsupported_workloads(field,value):
    original,_=contract(); setattr(original,field,value)
    with pytest.raises(ValueError): validate_contract(original)


def test_initial_artifact_is_hash_bound():
    value, initial=contract()
    with pytest.raises(ValueError): TreeCoordinator(value, initial+b"corrupt")


def test_failed_merge_does_not_record_an_accepted_last_contribution(monkeypatch):
    value, initial = contract(); coordinator=TreeCoordinator(value,initial)
    coordinator.register("a",partition=0); coordinator.register("b",partition=1)
    a,b=request(coordinator,"a"),request(coordinator,"b")
    coordinator.submit(a); before=coordinator.parent
    def refuse(parent,contributions): raise ValueError("ensemble size limit")
    monkeypatch.setattr("fedlearn.tree.coordinator.merge_trees",refuse)
    with pytest.raises(ValueError):coordinator.submit(b)
    assert coordinator.parent==before and coordinator.accepted_count==1
    assert ("b",1) not in coordinator.receipts
