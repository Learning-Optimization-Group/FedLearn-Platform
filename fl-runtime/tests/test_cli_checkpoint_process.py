"""Run the real runtime and real TinyNet clients through an interrupted training round."""
from collections import OrderedDict
import os
import shutil
from pathlib import Path
import socket
import subprocess
import sys
import time

import grpc
import numpy as np
import pytest
import torch

import client
import recipes
from fedlearn.communication.generated import fedlearn_pb2 as pb, fedlearn_pb2_grpc as rpc
from fedlearn.communication.serializer import parameters_to_proto, proto_to_parameters


@pytest.fixture
def participants(monkeypatch):
    for name in ("USE_LLM_LORA", "USE_LLM", "USE_MLP", "USE_PNEUMONIA", "USE_DERIVED"):
        monkeypatch.setattr(client, name, False)
    monkeypatch.setattr(client, "MODEL_TYPE", "TINYNET_GOLDEN")
    monkeypatch.setattr(client, "TRAINING_ARM", "FULL")
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    monkeypatch.setattr(client, "USE_WIRE_SUBSET", recipes.federates_trainable_subset("TINYNET_GOLDEN", "FULL"))
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"))
    return [client.ZOSLClient(partition_id=index, dataset_name="cb", num_clients=2) for index in range(2)]


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def await_round(stub, number):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if stub.GetServerStatus(pb.GetServerStatusRequest(), timeout=2).current_round == number:
            return
        time.sleep(.02)
    raise AssertionError(f"runtime did not reach round {number}")


def fit_and_submit(stub, participant, index, round_):
    model = stub.GetGlobalModel(pb.GetGlobalModelRequest(client_id=str(index)), timeout=3)
    assert model.current_round == round_
    parameters, _ = proto_to_parameters(model.parameters)
    config = dict(model.config) | {"server_round": round_}
    # The same generated-data client round gets the same local shuffle in both placements.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10_000 + 10 * round_ + index)
        # This fixture loader owns a separate generator; an interrupted attempt advances it too.
        participant.trainloader.generator.manual_seed(10_000 + 10 * round_ + index)
        updated, examples = participant.fit(parameters, config)
    assert stub.SubmitModelUpdate(pb.SubmitModelUpdateRequest(
        client_id=str(index), trained_on_round=round_, parameters=parameters_to_proto(updated, examples)),
        timeout=3).received


@pytest.mark.parametrize("strategy,boundary", [
    ("FedAvg", "partial_round"), ("FedOpt", "partial_round"), ("FedProx", "partial_round"),
    ("FedOpt", "partial_final_save"), ("FedAvg", "after_final_save"),
])
def test_cli_recovery_matches_uninterrupted_training_with_real_clients(tmp_path, participants, strategy, boundary):
    initial = tmp_path / "initial.npz"
    environment = os.environ.copy()
    environment.update(FEDLEARN_REQUIRE_CLIENT_AUTH="0", FEDLEARN_REQUIRE_TLS="0",
                       FEDLEARN_GRPC_USE_TLS="0", FEDLEARN_COMPLETION_DRAIN_SECONDS="0",
                       FEDLEARN_INTERNAL_API_KEY="", FEDLEARN_INTERNAL_RUN_TOKEN="",
                       FEDLEARN_BACKEND_URL="http://127.0.0.1:1", CUDA_VISIBLE_DEVICES="",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.pop("FEDLEARN_RECORD_CLIENT_UPDATES", None)
    runtime = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(runtime) + os.pathsep + environment.get("PYTHONPATH", "")
    # Use the production initializer: this recipe's frozen fc2 must not enter the wire state.
    subprocess.run([sys.executable, str(runtime / "init_model.py"), "--model-type", "TINYNET_GOLDEN",
                    "--model-name", "tinynet_golden", "--optimizer", "SGD", "--out", str(initial)],
                   cwd=runtime, env=environment, check=True, capture_output=True, timeout=30)
    outputs = []
    for interrupted, run_id in ((False, "ad17a470-4412-4b96-bda6-d0e0daa5b966"),
                                (True, "7ca425ee-203e-45a1-aeaf-e7339dac64fe")):
        root = tmp_path / ("interrupted" if interrupted else "control")
        root.mkdir()
        output = root / "final.npz"
        if boundary != "partial_round":
            shutil.copyfile(initial, output)  # Exercise the default initialization/write-target alias.
        environment["FEDLEARN_RUN_ID"] = run_id
        process = channel = None
        with (root / "runtime.log").open("w") as log:
            def launch(resume=False, connect=True):
                port = free_port()
                runner = "import fl_server; fl_server.bind_address='127.0.0.1'; fl_server.main()"
                if interrupted and not resume and boundary != "partial_round":
                    runner = "\n".join([
                        "import os", "from pathlib import Path", "import fl_server",
                        "original_save = fl_server.np.savez", "def fail_final_save(path, **weights):",
                        "    Path(path).write_bytes(b'partial final model')" if boundary == "partial_final_save"
                        else "    original_save(path, **weights)",
                        "    os._exit(90)", "fl_server.np.savez = fail_final_save",
                        "fl_server.bind_address = '127.0.0.1'", "fl_server.main()",
                    ])
                command = [sys.executable, "-u", "-c",
                           runner, "--model-path", str(output),
                           "--project-id", "17c01f7b-cb3a-419a-be1e-ceb3abc17f6b", "--model-type", "TINYNET_GOLDEN",
                           "--model-name", "tinynet_golden", "--strategy", strategy, "--num-rounds", "3",
                           "--min-clients", "2", "--seed", "1729", "--port", str(port),
                           "--checkpoint-directory", str(root / "private")]
                if boundary == "partial_round":
                    command += ["--init-model-path", str(initial)]
                if resume:
                    command.append("--resume-from-checkpoint")
                child = subprocess.Popen(command, cwd=runtime, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if not connect:
                    return child, None, None
                connection = grpc.insecure_channel(f"127.0.0.1:{port}")
                try:
                    grpc.channel_ready_future(connection).result(timeout=25)
                    stub = rpc.FederatedLearningServiceStub(connection)
                    for index in range(2):
                        reply = stub.RegisterClient(pb.RegisterClientRequest(client_id=str(index), protocol_version=2),
                                                    timeout=3)
                        assert reply.status == pb.RegisterClientResponse.ACCEPTED
                except BaseException:
                    connection.close()
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)
                    raise
                return child, connection, stub

            try:
                process, channel, stub = launch()
                for index, participant in enumerate(participants):
                    fit_and_submit(stub, participant, index, 1)
                await_round(stub, 2)
                if interrupted and boundary == "partial_round":
                    fit_and_submit(stub, participants[0], 0, 2)
                    assert stub.GetServerStatus(pb.GetServerStatusRequest(), timeout=2).received_updates_this_round == 1
                    process.kill()
                    process.wait(timeout=5)
                    channel.close()
                    process, channel, stub = launch(resume=True)
                    await_round(stub, 2)
                    assert stub.GetServerStatus(pb.GetServerStatusRequest(), timeout=2).received_updates_this_round == 0
                for round_ in (2, 3):
                    for index, participant in enumerate(participants):
                        fit_and_submit(stub, participant, index, round_)
                    if round_ < 3:
                        await_round(stub, round_ + 1)
                if interrupted and boundary != "partial_round":
                    assert process.wait(timeout=20) == 90
                    channel.close()
                    process, channel, stub = launch(resume=True, connect=False)
                assert process.wait(timeout=25) == 0
            finally:
                if channel is not None:
                    channel.close()
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        with np.load(output, allow_pickle=False) as saved:
            outputs.append(OrderedDict((key, saved[key].copy()) for key in saved.files))
    assert outputs[0].keys() == outputs[1].keys()
    for name in outputs[0]:
        np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
