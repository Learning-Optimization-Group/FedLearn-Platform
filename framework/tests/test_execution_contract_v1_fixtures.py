"""Execution contract v1 golden fixtures through the generated Python reader.

The golden contract is committed twice -- canonical protobuf bytes and ProtoJSON -- and every
language's generated reader must recover the same message from both. These tests pin the Python
side: both encodings agree, protobuf bytes survive a parse/serialize round trip unchanged, the
ProtoJSON rendering is exactly the committed document, explicit-presence zeros survive, and the
committed files are what the generator produces (so a hand edit cannot drift from the generator).
"""
from __future__ import annotations

import importlib.util
import json
import os

from google.protobuf import json_format

from fedlearn.communication.generated import execution_contract_pb2 as pb

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "execution_contract_v1")


def _generator():
    spec = importlib.util.spec_from_file_location(
        "execution_contract_v1_generate", os.path.join(FIXTURES, "generate.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golden_bytes() -> bytes:
    with open(os.path.join(FIXTURES, "golden_tinynet_fedavg.binpb"), "rb") as fh:
        return fh.read()


def _golden_json_text() -> str:
    with open(os.path.join(FIXTURES, "golden_tinynet_fedavg.json"), encoding="utf-8") as fh:
        return fh.read()


def _parse_binary() -> pb.ExecutionContract:
    contract = pb.ExecutionContract()
    contract.ParseFromString(_golden_bytes())
    return contract


def test_binary_and_protojson_goldens_decode_to_the_same_contract():
    from_json = json_format.Parse(_golden_json_text(), pb.ExecutionContract(),
                                  ignore_unknown_fields=True)
    assert from_json == _parse_binary()


def test_binary_golden_reserializes_to_identical_bytes():
    assert _parse_binary().SerializeToString(deterministic=True) == _golden_bytes()


def test_protojson_rendering_matches_the_committed_document():
    rendered = json_format.MessageToDict(_parse_binary())
    assert rendered == json.loads(_golden_json_text())


def test_explicit_presence_zeros_survive_both_encodings():
    for contract in (_parse_binary(),
                     json_format.Parse(_golden_json_text(), pb.ExecutionContract())):
        sgd = contract.model_training.local_training.sgd
        assert sgd.HasField("momentum") and sgd.momentum == 0.0
        assert sgd.HasField("weight_decay") and sgd.weight_decay == 0.0
        assert sgd.HasField("nesterov") and sgd.nesterov is False
        local = contract.model_training.local_training
        assert local.HasField("drop_last") and local.drop_last is False
        assert not local.HasField("max_local_steps")
        assert not local.HasField("gradient_clip_norm")
        assert not contract.security.HasField("secure_agg_threshold")
        assert not contract.security.HasField("central_dp")


def test_golden_describes_the_tinynet_fedavg_emission():
    contract = _parse_binary()
    training = contract.model_training
    assert contract.contract_version == 1
    assert contract.recipe == pb.RECIPE_TINYNET_GOLDEN
    assert contract.strategy == pb.STRATEGY_FEDAVG
    assert training.arm == pb.ARM_FULL
    assert training.update_protocol == pb.UPDATE_TRAINABLE_STATE_F32
    assert [(t.name, list(t.shape)) for t in training.trainable] == [
        ("fc1.weight", [5, 4]), ("fc1.bias", [5])]
    assert [t.identity_vector.width for t in training.data.transforms] == [4]
    assert training.data.class_count == 3


def test_committed_golden_files_are_the_generator_output():
    generator = _generator()
    contract = generator.build_golden()
    assert generator.to_binary(contract) == _golden_bytes()
    assert generator.to_json_text(contract) == _golden_json_text()
