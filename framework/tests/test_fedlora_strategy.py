# framework/tests/test_fedlora_strategy.py
import pytest
import torch
from collections import OrderedDict
from fedlearn.server.strategy import FedLoRA


def ffa_initial():
    # Global carries A (frozen+shared) + B + head.
    return OrderedDict([
        ("m.lora_A.weight", torch.tensor([[1.0, 1.0]])),
        ("m.lora_B.weight", torch.tensor([[0.0], [0.0]])),
        ("score.weight", torch.tensor([0.0])),
    ])


def ffa_upload(bval, hval):
    # Client uploads only B + head under FFA.
    return OrderedDict([
        ("m.lora_B.weight", torch.tensor([[bval], [bval]])),
        ("score.weight", torch.tensor([hval])),
    ])


def test_ffa_reattaches_frozen_a_and_averages_b_head():
    s = FedLoRA(initial_parameters=ffa_initial(), aggregation="FFA_LORA")
    out = s.aggregate_fit(1, [(ffa_upload(2.0, 2.0), 100), (ffa_upload(4.0, 4.0), 100)])
    assert torch.allclose(out["m.lora_B.weight"], torch.tensor([[3.0], [3.0]]))   # averaged
    assert torch.allclose(out["score.weight"], torch.tensor([3.0]))               # averaged
    assert torch.allclose(out["m.lora_A.weight"], torch.tensor([[1.0, 1.0]]))     # re-attached unchanged


def test_assert_homogeneous_raises_on_shape_mismatch():
    s = FedLoRA(initial_parameters=ffa_initial(), aggregation="FFA_LORA")
    bad = OrderedDict([("m.lora_B.weight", torch.tensor([[1.0]])), ("score.weight", torch.tensor([1.0]))])
    with pytest.raises(ValueError, match="[Hh]eterogeneous"):
        s.aggregate_fit(1, [(ffa_upload(1.0, 1.0), 100), (bad, 100)])


def test_fedit_averages_all_keys_no_reattach():
    init = OrderedDict([
        ("m.lora_A.weight", torch.tensor([[1.0, 1.0]])),
        ("m.lora_B.weight", torch.tensor([[0.0], [0.0]])),
        ("score.weight", torch.tensor([0.0])),
    ])
    s = FedLoRA(initial_parameters=init, aggregation="FEDIT")

    def up(v):
        return OrderedDict([
            ("m.lora_A.weight", torch.tensor([[v, v]])),
            ("m.lora_B.weight", torch.tensor([[v], [v]])),
            ("score.weight", torch.tensor([v])),
        ])

    out = s.aggregate_fit(1, [(up(2.0), 100), (up(4.0), 100)])
    assert torch.allclose(out["m.lora_A.weight"], torch.tensor([[3.0, 3.0]]))  # A averaged in FedIT
    assert torch.allclose(out["m.lora_B.weight"], torch.tensor([[3.0], [3.0]]))


def test_empty_results_returns_none():
    s = FedLoRA(initial_parameters=ffa_initial(), aggregation="FFA_LORA")
    assert s.aggregate_fit(1, []) is None


def test_ffa_weighting_is_num_examples_weighted():
    s = FedLoRA(initial_parameters=ffa_initial(), aggregation="FFA_LORA")
    # B/head weighted by num_examples: (2*300 + 4*100)/400 = 2.5
    out = s.aggregate_fit(1, [(ffa_upload(2.0, 2.0), 300), (ffa_upload(4.0, 4.0), 100)])
    assert torch.allclose(out["score.weight"], torch.tensor([2.5]))
    assert torch.allclose(out["m.lora_B.weight"], torch.tensor([[2.5], [2.5]]))


def test_ffa_init_without_lora_a_raises():
    no_a = OrderedDict([("m.lora_B.weight", torch.zeros(2, 1)), ("score.weight", torch.zeros(1))])
    with pytest.raises(ValueError, match="lora_A"):
        FedLoRA(initial_parameters=no_a, aggregation="FFA_LORA")
