"""Local vector data must match the published contract and execute its batch order."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from google.protobuf import json_format

import client
import execution_plan
import recipes
from fedlearn.contract import parse_contract_binary
from fedlearn.contract.batch_order import seeded_permutation


def plan():
    return execution_plan.resolve_model_training("TINYNET_GOLDEN", "FedAvg", "FULL",
                                                 data_source="LOCAL_SNAPSHOT")


def csv_file(path, n=19):
    path.write_text("f0,f1,f2,f3,label\n" + "".join(
        f"{i},0,0,0,c{i % 3}\n" for i in range(n)))
    return path


def snapshot(path, n=19):
    path.mkdir()
    x = np.arange(n * 4, dtype="<f4").reshape(n, 4)
    y = (np.arange(n) % 3).astype("<i8")
    (path / "inputs.f32").write_bytes(x.tobytes())
    (path / "targets.i64").write_bytes(y.tobytes())
    meta = dict(schemaVersion=1, modality="vector", inputShape=[4], inputDtype="f32",
                classNames=["c0", "c1", "c2"], labelSchemaId=plan().data.label_schema_id,
                recordCount=n, inputsSha256=hashlib.sha256(x.tobytes()).hexdigest(),
                targetsSha256=hashlib.sha256(y.tobytes()).hexdigest())
    meta["snapshotId"] = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (path / "snapshot.json").write_text(json.dumps(meta))
    return path, x, y


def test_csv_keeps_every_local_row_and_named_label_without_partitioning(tmp_path):
    from local_snapshot import load_vector_snapshot
    local = load_vector_snapshot(csv_file(tmp_path / "own.csv"), plan().data, ["c0", "c1", "c2"])
    x, y = local.dataset.tensors
    assert x.shape == (19, 4)
    assert x[:, 0].tolist() == list(range(19))
    assert y.tolist() == [i % 3 for i in range(19)]
    assert len(local.snapshot_id) == 64


def test_android_snapshot_reads_identical_tensors_and_verifies_its_content(tmp_path):
    from local_snapshot import load_vector_snapshot
    path, x, y = snapshot(tmp_path / "own")
    local = load_vector_snapshot(path, plan().data, ["c0", "c1", "c2"])
    np.testing.assert_array_equal(local.dataset.tensors[0], x)
    np.testing.assert_array_equal(local.dataset.tensors[1], y)
    (path / "inputs.f32").write_bytes(b"\x00" * x.nbytes)
    with pytest.raises(ValueError, match="digest"):
        load_vector_snapshot(path, plan().data, ["c0", "c1", "c2"])


@pytest.mark.parametrize("text", ["f0,f1,f2,label\n1,2,3,c0\n",
                                 "f0,f1,f2,f3,label\n1,2,3,4,Normal\n",
                                 "f0,f1,f2,f3,label\nNaN,2,3,4,c0\n",
                                 "f0,f1,f2,f3,label\n1e100,2,3,4,c0\n",
                                 "f0,f1,f2,f3,label\n"])
def test_bad_local_csv_is_refused(tmp_path, text):
    from local_snapshot import load_vector_snapshot
    p = tmp_path / "bad.csv"; p.write_text(text)
    with pytest.raises(ValueError):
        load_vector_snapshot(p, plan().data, ["c0", "c1", "c2"])


def test_each_epoch_and_round_uses_contract_permutation_with_partial_batch(tmp_path):
    from local_snapshot import load_vector_snapshot, SeededSnapshotLoader
    local = load_vector_snapshot(csv_file(tmp_path / "own.csv"), plan().data, ["c0", "c1", "c2"])
    loader = SeededSnapshotLoader(local.dataset, batch_size=8, seed=2**64-1, drop_last=False)
    loader.set_round(7)
    for epoch in (0, 1):
        seen = [int(i) for x, _ in loader for i in x[:, 0]]
        assert seen == seeded_permutation(19, 2**64-1, 7, epoch)
    loader.set_round(8)
    batches = list(loader)
    assert [len(x) for x, _ in batches] == [8, 8, 3]
    assert torch.cat([x for x, _ in batches])[:, 0].tolist() == seeded_permutation(19, 2**64-1, 8, 0)


def local_contract(tmp_path):
    root = Path(__file__).resolve().parents[2]
    c = parse_contract_binary((root / "framework/tests/fixtures/execution_contract_v1/golden_tinynet_fedavg.binpb").read_bytes())
    c.model_training.data.CopyFrom(plan().data)
    c.model_training.local_training.CopyFrom(plan().local_training)
    p = tmp_path / "contract.json"; p.write_text(json_format.MessageToJson(c))
    return c, p


def test_cli_binds_validated_local_data_to_contract_and_loads_all_rows(tmp_path, monkeypatch):
    c, contract = local_contract(tmp_path)
    csv = csv_file(tmp_path / "own.csv")
    monkeypatch.setattr(client, "LOCAL_DATASET", None, raising=False)
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    args = client.parse_args(["--project-id", c.project_id, "--server-address", "localhost:50000",
                             "--partition-id", "1", "--model-type", "TINYNET_GOLDEN",
                             "--execution-contract", str(contract), "--run-id", c.run_id,
                             "--dataset-path", str(csv)])
    assert client.enforce_execution_contract(args, "TINYNET_GOLDEN", "FULL") == c
    monkeypatch.setattr(client, "MODEL_TYPE", "TINYNET_GOLDEN")
    loader, _ = client.load_data(1, "cb", str(csv), 10)
    assert len(loader.dataset) == 19
    assert execution_plan.check_contract(c, "TINYNET_GOLDEN", "FedAvg", "FULL",
                                         data_source="LOCAL_SNAPSHOT") == []


def test_ecg_override_preserves_explicit_dataset_path():
    args = client.parse_args(["--project-id", "p", "--server-address", "localhost:50000",
                             "--partition-id", "0", "--model-type", "MLP",
                             "--dataset-path", "/data/own.csv"])
    path, _ = client.apply_ecg_run_settings(args)
    assert path == "/data/own.csv"


def test_local_path_without_model_type_is_refused_before_model_construction(tmp_path, monkeypatch):
    args = client.parse_args(["--project-id", "p", "--server-address", "localhost:50000",
                             "--partition-id", "0", "--dataset-path", str(tmp_path)])
    monkeypatch.setattr(client, "parse_args", lambda: args)
    def no_model(*args, **kwargs):
        raise AssertionError("model constructed before validating local-data intent")
    monkeypatch.setattr(client, "ZOSLClient", no_model)
    with pytest.raises(SystemExit, match="model-type"):
        client.main()


def test_local_path_without_execution_contract_is_refused(tmp_path):
    args = client.parse_args(["--project-id", "p", "--server-address", "localhost:50000",
                             "--partition-id", "0", "--model-type", "TINYNET_GOLDEN",
                             "--dataset-path", str(tmp_path)])
    with pytest.raises(SystemExit, match="execution contract"):
        client.enforce_execution_contract(args, "TINYNET_GOLDEN", "FULL")


def test_local_fit_matches_manual_steps_for_the_actual_round_and_rejects_missing_round(tmp_path, monkeypatch):
    from local_snapshot import load_vector_snapshot, SeededSnapshotLoader
    local = load_vector_snapshot(csv_file(tmp_path / "own.csv"), plan().data, ["c0", "c1", "c2"])
    c, _ = local_contract(tmp_path)
    loader = SeededSnapshotLoader(local.dataset, batch_size=8, seed=int(c.seed), drop_last=False)
    for key in ("USE_LLM", "USE_MLP", "USE_PNEUMONIA", "USE_LLM_LORA", "USE_DERIVED"):
        monkeypatch.setattr(client, key, False)
    monkeypatch.setattr(client, "MODEL_TYPE", "TINYNET_GOLDEN")
    monkeypatch.setattr(client, "TRAINING_ARM", "FULL")
    monkeypatch.setattr(client, "USE_WIRE_SUBSET", True)
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"))
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", c)
    monkeypatch.setattr(client, "LOCAL_DATASET", loader)
    participant = client.ZOSLClient(partition_id=0, dataset_name="cb")
    initial = participant.get_parameters()
    with pytest.raises(ValueError, match="round"):
        participant.fit(initial, {})
    actual, count = participant.fit(initial, {"server_round": 2})

    expected = recipes.build_tinynet_golden()
    expected.load_state_dict(initial, strict=False)
    optimizer = torch.optim.SGD([p for p in expected.parameters() if p.requires_grad], lr=0.001)
    x, y = local.dataset.tensors
    order = seeded_permutation(19, int(c.seed), 2, 0)
    for offset in range(0, 19, 8):
        indices = order[offset:offset + 8]
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(expected(x[indices]), y[indices]).backward()
        optimizer.step()
    assert count == 19
    for key, value in actual.items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("case", ["classes", "shape", "labels", "nonfinite"])
def test_rehashed_but_incompatible_snapshot_is_refused(tmp_path, case):
    from local_snapshot import load_vector_snapshot
    path, x, y = snapshot(tmp_path / "own")
    meta = json.loads((path / "snapshot.json").read_text())
    if case == "classes":
        meta["classNames"] = list(reversed(meta["classNames"]))
    elif case == "shape":
        meta["inputShape"] = [2, 2]
    elif case == "labels":
        y[0] = 999; (path / "targets.i64").write_bytes(y.tobytes())
        meta["targetsSha256"] = hashlib.sha256(y.tobytes()).hexdigest()
    else:
        x[0, 0] = np.nan; (path / "inputs.f32").write_bytes(x.tobytes())
        meta["inputsSha256"] = hashlib.sha256(x.tobytes()).hexdigest()
    content = {k: v for k, v in meta.items() if k != "snapshotId"}
    meta["snapshotId"] = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (path / "snapshot.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_vector_snapshot(path, plan().data, ["c0", "c1", "c2"])
