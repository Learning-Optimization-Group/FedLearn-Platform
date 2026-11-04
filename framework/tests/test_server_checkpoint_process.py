"""Real-process recovery discards a partial round and preserves committed FedOpt state."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import grpc
import torch

from fedlearn.communication.generated import fedlearn_pb2 as pb, fedlearn_pb2_grpc as rpc
from fedlearn.communication.serializer import parameters_to_proto
from fedlearn.server.checkpoint import CheckpointIdentity, RoundCheckpointStore
from fedlearn.server.strategy import FedOpt


SERVER = """
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sys
import torch
from fedlearn.server.checkpoint import CheckpointIdentity
from fedlearn.server.server import ServerConfig, start_server
from fedlearn.server.strategy import FedOpt

directory, result_path, address, resume = sys.argv[1:]
strategy = FedOpt(OrderedDict(w=torch.zeros(1)), min_fit_clients=2, clients_per_round=2,
                  server_learning_rate=.2, bias_correction=True,
                  evaluate_fn=lambda number, model: (float(model['w'].square().mean()), {'round': number}))
config = ServerConfig(num_rounds=3, checkpoint_directory=directory,
                      checkpoint_identity=CheckpointIdentity('process-run', hashlib.sha256(b'process-config').hexdigest()),
                      resume_from_checkpoint=resume == 'yes')
history, model = start_server(address, config, strategy)
if not model:
    raise RuntimeError('training did not complete')
Path(result_path).write_text(json.dumps({'history': history, 'w': model['w'].tolist()}))
"""


def parameters(value):
    return OrderedDict(w=torch.tensor([value], dtype=torch.float32))


def free_address():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{listener.getsockname()[1]}"


def wait_for_round(stub, expected):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        status = stub.GetServerStatus(pb.GetServerStatusRequest(), timeout=2)
        if status.current_round == expected:
            return
        time.sleep(.02)
    raise AssertionError(f"server did not reach round {expected}")


def submit(stub, client, round_, value):
    response = stub.SubmitModelUpdate(pb.SubmitModelUpdateRequest(
        client_id=client, trained_on_round=round_, parameters=parameters_to_proto(parameters(value), 2)), timeout=3)
    return response.received


def test_killed_server_resumes_from_committed_round_and_matches_uninterrupted_fedopt(tmp_path):
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    environment.update(FEDLEARN_REQUIRE_CLIENT_AUTH="0", FEDLEARN_REQUIRE_TLS="0",
                       FEDLEARN_GRPC_USE_TLS="0", FEDLEARN_COMPLETION_DRAIN_SECONDS="0")
    environment.pop("FEDLEARN_RECORD_CLIENT_UPDATES", None)
    directory = tmp_path / "private"
    result = tmp_path / "result.json"
    process = None
    channel = None
    with (tmp_path / "server.log").open("w") as log:
        def launch(resume):
            address = free_address()
            child = subprocess.Popen(
                [sys.executable, "-u", "-c", SERVER, str(directory), str(result), address, resume],
                env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            connection = grpc.insecure_channel(address)
            try:
                grpc.channel_ready_future(connection).result(timeout=20)
                stub = rpc.FederatedLearningServiceStub(connection)
                for client in ("a", "b"):
                    reply = stub.RegisterClient(pb.RegisterClientRequest(client_id=client), timeout=3)
                    assert reply.status == pb.RegisterClientResponse.ACCEPTED
            except BaseException:
                connection.close()
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)
                raise
            return child, connection, stub

        try:
            process, channel, stub = launch("no")
            assert submit(stub, "a", 1, 1)
            assert submit(stub, "b", 1, 3)
            wait_for_round(stub, 2)
            assert submit(stub, "a", 2, 99)  # This incomplete round must never be committed.
            assert stub.GetServerStatus(pb.GetServerStatusRequest(), timeout=2).received_updates_this_round == 1
            process.kill()
            process.wait(timeout=5)
            channel.close()
            process, channel, stub = launch("yes")
            wait_for_round(stub, 2)
            # The legacy received bit acknowledges transport, including a stale retry.
            # The authoritative contribution count must remain zero.
            submit(stub, "a", 1, 99)
            assert stub.GetServerStatus(pb.GetServerStatusRequest(), timeout=2).received_updates_this_round == 0
            for round_ in (2, 3):
                assert submit(stub, "a", round_, round_)
                assert submit(stub, "b", round_, round_ + 2)
                if round_ < 3:
                    wait_for_round(stub, round_ + 1)
            assert process.wait(timeout=15) == 0
        finally:
            if channel is not None:
                channel.close()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    recovered = json.loads(result.read_text())
    assert [number for number, _ in recovered["history"]] == [1, 2, 3]
    reference = FedOpt(parameters(0), min_fit_clients=2, clients_per_round=2,
                       server_learning_rate=.2, bias_correction=True)
    for round_ in (1, 2, 3):
        expected = reference.aggregate_fit(round_, [(parameters(round_), 2), (parameters(round_ + 2), 2)])
    assert torch.equal(torch.tensor(recovered["w"]), expected["w"])
    store = RoundCheckpointStore(directory, CheckpointIdentity(
        "process-run", hashlib.sha256(b"process-config").hexdigest()))
    assert store.load_latest().completed_round == 3
