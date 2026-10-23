"""--head-init zero: zero the output layer of a subset arm's initial federated head (capstone A17)."""
import json
import os
import sys
from collections import OrderedDict

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fl_server  # noqa: E402
from test_fl_server_eval_card import _BASE_ARGV, _HISTORY  # noqa: E402


def _mlp_head():
    return OrderedDict([("classifier.1.weight", torch.ones(4, 6)), ("classifier.1.bias", torch.ones(4)),
                        ("classifier.4.weight", torch.ones(2, 4)), ("classifier.4.bias", torch.ones(2))])


def test_a_single_linear_head_is_zeroed_whole():
    params = OrderedDict([("fc.weight", torch.randn(10, 512)), ("fc.bias", torch.randn(10))])
    out, keys = fl_server.zero_output_layer(params, "FROZEN_HEAD")
    assert keys == ["fc.weight", "fc.bias"]
    assert all(torch.count_nonzero(v) == 0 for v in out.values())


def test_only_the_output_layer_of_an_mlp_head_is_zeroed():
    out, keys = fl_server.zero_output_layer(_mlp_head(), "OVA_LP")
    assert keys == ["classifier.4.weight", "classifier.4.bias"]
    assert torch.equal(out["classifier.1.weight"], torch.ones(4, 6))  # hidden layer keeps its init
    assert torch.count_nonzero(out["classifier.4.weight"]) == 0


@pytest.mark.parametrize("arm, init_path, message", [
    ("FULL", None, "arms"),
    ("FROZEN_HEAD", "/registry/head.npz", "continued run"),
])
def test_refusals(arm, init_path, message):
    with pytest.raises(ValueError, match=message):
        fl_server.zero_output_layer(_mlp_head(), arm, init_path)


def test_a_head_without_a_matrix_is_refused():
    with pytest.raises(ValueError, match="no 2-D"):
        fl_server.zero_output_layer(OrderedDict(b=torch.ones(3)), "FROZEN_HEAD")


def test_flag_defaults_to_default_and_the_card_records_it():
    parser = fl_server.build_arg_parser()
    default = parser.parse_args(_BASE_ARGV)
    zero = parser.parse_args([*_BASE_ARGV, "--head-init", "zero", "--training-arm", "FROZEN_HEAD"])
    assert default.head_init == "default" and zero.head_init == "zero"
    assert json.loads(fl_server.build_eval_card(zero, _HISTORY, None))["head_init"] == "zero"
    assert json.loads(fl_server.build_eval_card(default, _HISTORY, None))["head_init"] == "default"


def test_apply_head_init_updates_the_federated_and_full_state_together():
    from argparse import Namespace
    full = OrderedDict([("conv.weight", torch.ones(3, 3)), *_mlp_head().items()])
    federated = OrderedDict((k, v.clone()) for k, v in full.items() if k.startswith("classifier."))
    args = Namespace(head_init="zero", training_arm="FROZEN_HEAD", init_model_path=None)
    out = fl_server.apply_head_init(args, federated, full)
    assert torch.count_nonzero(out["classifier.4.weight"]) == 0
    assert torch.count_nonzero(full["classifier.4.weight"]) == 0       # the saved merge sees the same head
    assert torch.equal(full["conv.weight"], torch.ones(3, 3))           # backbone untouched
    untouched = fl_server.apply_head_init(Namespace(head_init="default"), federated, full)
    assert untouched is federated
