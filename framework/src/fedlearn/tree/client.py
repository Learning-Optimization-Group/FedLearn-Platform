"""Tree client adapter used by the canonical platform client entry point."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import grpc

from fedlearn.communication.generated import execution_contract_pb2 as c, tree_pb2 as p, tree_pb2_grpc as rpc
from fedlearn.security.client_interceptor import maybe_wrap_channel
from .contract import contract_id, parse_contract
from .data import load_csv
from .model import _xgb, canonical, train_delta, validate_model


def _channel(address, contract):
    use_tls = os.environ.get("FEDLEARN_GRPC_USE_TLS") == "1"
    if use_tls != (contract.security.transport == c.TRANSPORT_TLS_REQUIRED):
        raise ValueError("tree client transport differs from the contract")
    if contract.security.client_auth == c.CLIENT_AUTH_CONNECTION_TOKEN and not os.environ.get("FEDLEARN_CONNECTION_TOKEN"):
        raise ValueError("tree run requires its enrollment connection token")
    options = [("grpc.max_send_message_length", 128 * 1024), ("grpc.max_receive_message_length", 32 * 1024 * 1024)]
    if use_tls:
        def read(name):
            path = os.environ.get(name)
            return Path(path).read_bytes() if path else None
        credentials = grpc.ssl_channel_credentials(root_certificates=read("FEDLEARN_GRPC_ROOT_CERT"),
            private_key=read("FEDLEARN_GRPC_CLIENT_KEY"), certificate_chain=read("FEDLEARN_GRPC_CLIENT_CERT"))
        channel = grpc.secure_channel(address, credentials, options=options)
    else:
        channel = grpc.insecure_channel(address, options=options)
    return channel, maybe_wrap_channel(channel)


def _call(method, request, policy):
    for attempt in range(policy.max_transient_retries + 1):
        try:
            return method(request, timeout=10)
        except grpc.RpcError as exc:
            if exc.code() not in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED) or attempt == policy.max_transient_retries:
                raise
            time.sleep(policy.retry_backoff_ms / 1000 * (2 ** attempt))


def run_client(args):
    contract = parse_contract(Path(args.execution_contract).read_text(), run_id=args.run_id, project_id=args.project_id)
    identifier = contract_id(contract)
    if args.strategy != "XgbBagging" or args.training_arm != "FULL" or args.device != "cpu":
        raise ValueError("XGBoost requires XgbBagging, FULL and cpu")
    features, labels, fingerprint = load_csv(args.dataset_path)
    client_id = f"project_{args.project_id}_client_{args.partition_id}"
    raw_channel, channel = _channel(args.server_address, contract)
    try:
        stub = rpc.FederatedTreeServiceStub(channel)
        registration = p.TreeRegistration(client_id=client_id, run_id=contract.run_id, contract_id=identifier,
                                          protocol_version=3, partition_id=args.partition_id)
        ack = _call(stub.RegisterClient, registration, contract.round)
        if ack.protocol_version != 3:
            raise ValueError("server does not speak the tree protocol")
        print(json.dumps(dict(client=client_id, run=contract.run_id, contract=identifier,
                              local_rows=len(features), local_snapshot_sha256=fingerprint)), flush=True)
        accepted = set()
        highest_round = 0
        initial_digest = None
        finalization_started = None
        request = p.TreeModelRequest(client_id=client_id, run_id=contract.run_id, contract_id=identifier)
        while True:
            state = _call(stub.GetTreeModel, request, contract.round)
            if state.contract_id != identifier or not 1 <= state.server_round <= contract.num_rounds:
                raise ValueError("tree model response identity/round mismatch")
            if state.state == p.TreeModel.FAILED:
                raise RuntimeError("tree run failed")
            if state.state == p.TreeModel.COMPLETE:
                print(json.dumps(dict(complete=True, accepted_rounds=sorted(accepted),
                                      final_model_sha256=state.parent_sha256)), flush=True)
                return 0
            if state.state == p.TreeModel.FINALIZING:
                if finalization_started is None:
                    finalization_started = time.monotonic()
                if time.monotonic() - finalization_started > 180:
                    raise TimeoutError("tree artifact finalization exceeded its callback deadline")
                time.sleep(.25)
                continue
            if state.state not in (p.TreeModel.WAITING, p.TreeModel.TRAINING):
                raise ValueError("unknown tree server state")
            if state.deadline_unix_ms <= int(time.time() * 1000):
                raise TimeoutError("tree round deadline elapsed")
            if state.server_round < highest_round:
                raise ValueError("tree server moved backwards")
            highest_round = state.server_round
            if state.state == p.TreeModel.WAITING or state.server_round in accepted:
                time.sleep(.25)
                continue
            initial = bytes(state.initial_model_ubj)
            if initial_digest is None:
                if hashlib.sha256(initial).hexdigest() != contract.tree_training.initial_model_sha256:
                    raise ValueError("initial tree artifact is not the contracted model")
                booster = _xgb().Booster(params={"device": "cpu", "nthread": 1})
                booster.load_model(bytearray(initial))
                initial_json = canonical(validate_model(bytes(booster.save_raw(raw_format="json"))))
                if state.server_round == 1 and initial_json != bytes(state.model_json):
                    raise ValueError("first-round model differs from the contracted initial ensemble")
                initial_digest = hashlib.sha256(initial).hexdigest()
            parent = bytes(state.model_json)
            if hashlib.sha256(parent).hexdigest() != state.parent_sha256:
                raise ValueError("tree parent model digest mismatch")
            validate_model(parent)
            started = time.monotonic()
            delta = train_delta(parent, features, labels, cohort_size=contract.clients_per_round,
                                seed=(contract.seed + state.server_round) % (2**64))
            contribution = p.TreeContribution(client_id=client_id, run_id=contract.run_id,
                contract_id=identifier, server_round=state.server_round, parent_sha256=state.parent_sha256,
                delta_sha256=hashlib.sha256(delta).hexdigest(), new_trees_json=delta, num_examples=len(features))
            response = _call(stub.SubmitTreeUpdate, contribution, contract.round)
            if response.result not in (p.TreeContributionAck.ACCEPTED, p.TreeContributionAck.ALREADY_ACCEPTED):
                raise ValueError("tree contribution was not accepted")
            accepted.add(state.server_round)
            print(json.dumps(dict(round=state.server_round, result=p.TreeContributionAck.Result.Name(response.result),
                parent_sha256=state.parent_sha256, delta_sha256=contribution.delta_sha256,
                uplink_bytes=contribution.ByteSize(), downlink_bytes=state.ByteSize(),
                compute_ms=round((time.monotonic() - started) * 1000, 3))), flush=True)
    finally:
        raw_channel.close()


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on the participant's own numeric CSV")
    parser.add_argument("--model-type", choices=["XGBOOST"], required=True)
    parser.add_argument("--model-name", choices=["xgboost_binary"], default="xgboost_binary")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--strategy", choices=["XgbBagging"], default="XgbBagging")
    parser.add_argument("--training-arm", choices=["FULL"], default="FULL")
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    args = parser.parse_args()
    if not 0 <= args.partition_id <= 9999:
        parser.error("partition-id must be the nonnegative enrolled partition")
    return run_client(args)
