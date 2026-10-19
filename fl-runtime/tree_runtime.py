"""Platform entry points for the pinned CPU XGBoost workload."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import grpc
import numpy as np
import requests

from fedlearn.communication.generated import execution_contract_pb2 as c, tree_pb2 as p
from fedlearn.security.identity import partition_extractor_from_env
from fedlearn.security.interceptor import interceptor_from_env
from fedlearn.security.tls import check_server_tls_policy
from fedlearn.tree.contract import validate_contract
from fedlearn.tree.coordinator import TreeCoordinator
from fedlearn.tree.model import _xgb, canonical, initial_model, load_model, validate_model
from fedlearn.tree.schema import MODEL_SCHEMAS, model_schema
from fedlearn.tree.transport import add_tree_services


def _base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-type", choices=["XGBOOST"], required=True)
    parser.add_argument("--model-name", choices=list(MODEL_SCHEMAS), required=True)
    return parser


def init_main():
    parser = _base_parser("Initialize the zero-tree XGBoost ensemble")
    parser.add_argument("--optimizer", choices=["TreeBoosting"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pretrain-epochs", type=int, choices=[0], default=0)
    parser.add_argument("--task-type", choices=["SEQ_CLASSIFICATION"], default="SEQ_CLASSIFICATION")
    args = parser.parse_args()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(load_model(initial_model(feature_names=model_schema(args.model_name))).save_raw(raw_format="ubj")))
    print("Initialized XGBoost 3.4.1 binary model with zero trained trees", flush=True)
    return 0


def read_checkpoint(path, *, feature_names=None):
    raw = Path(path).read_bytes()
    if not 0 < len(raw) <= 16 * 1024 * 1024:
        raise ValueError("tree checkpoint exceeds byte limit")
    # Only platform-owned, hash-verified registry/init artifacts enter this path.
    booster = _xgb().Booster(params={"device": "cpu", "nthread": 1})
    booster.load_model(bytearray(raw))
    validate_model(bytes(booster.save_raw(raw_format="json")), feature_names=feature_names)
    return booster


def infer_main():
    parser = _base_parser("Infer from a registered XGBoost ensemble")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-type", choices=["SEQ_CLASSIFICATION"], default="SEQ_CLASSIFICATION")
    args = parser.parse_args()
    input_error = False
    try:
        names = model_schema(args.model_name)
        booster = read_checkpoint(args.model_path, feature_names=names)
        payload = json.loads(Path(args.input_path).read_text())
        values = payload.get("values")
        if payload.get("kind") != "vector" or not isinstance(values, list) or len(values) != len(names) \
                or any(type(v) not in (float, int) or not np.isfinite(v)
                       or abs(v) > float(np.finfo(np.float32).max) for v in values):
            input_error = True
            raise ValueError(f"XGBoost input requires {len(names)} finite float32 values in the preset feature order")
        probability = float(booster.inplace_predict(np.array([values], dtype="float32"))[0])
        index = int(probability > .5)
        margin = float(booster.inplace_predict(np.array([values], dtype="float32"), predict_type="margin")[0])
        result = dict(ok=True, modelType="XGBOOST", predictedIndex=index, predictedLabel=str(index),
                      classes=["0", "1"], probabilities=[1 - probability, probability], logits=[0, margin])
        Path(args.out).write_text(json.dumps(result))
        return 0
    except Exception as exc:
        Path(args.out).write_text(json.dumps(dict(ok=False, errorKind="input" if input_error else "internal", error=str(exc))))
        return 1


def _headers():
    key, token = os.environ.get("FEDLEARN_INTERNAL_API_KEY"), os.environ.get("FEDLEARN_INTERNAL_RUN_TOKEN")
    if not key or not token:
        raise ValueError("platform tree server requires scoped internal callback credentials")
    return {"X-Internal-Key": key, "X-Internal-Run-Token": token}


def server_main():
    parser = _base_parser("Run the platform's fixed-cohort XGBoost server")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-model-path")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--strategy", choices=["XgbBagging"], required=True)
    parser.add_argument("--num-rounds", type=int, required=True)
    parser.add_argument("--min-clients", type=int, required=True)
    parser.add_argument("--clients-per-round", type=int)
    parser.add_argument("--training-arm", choices=["FULL"], default="FULL")
    args = parser.parse_args()
    contract = c.ExecutionContract.FromString(base64.b64decode(os.environ["FEDLEARN_TREE_CONTRACT_B64"], validate=True))
    validate_contract(contract, project_id=args.project_id, run_id=os.environ.get("FEDLEARN_RUN_ID"))
    if tuple(contract.tree_training.feature_names) != model_schema(args.model_name):
        raise ValueError("tree launch model preset disagrees with the contract")
    if args.num_rounds != contract.num_rounds or args.min_clients != contract.clients_per_round \
            or (args.clients_per_round is not None and args.clients_per_round != contract.clients_per_round):
        raise ValueError("tree launch arguments disagree with the published contract")
    use_tls = check_server_tls_policy()
    if use_tls != (contract.security.transport == c.TRANSPORT_TLS_REQUIRED) \
            or (os.environ.get("FEDLEARN_REQUIRE_CLIENT_AUTH") == "1") != (contract.security.client_auth == c.CLIENT_AUTH_CONNECTION_TOKEN):
        raise ValueError("tree transport/auth configuration disagrees with contract")
    callback_headers = _headers()
    backend = os.environ["FEDLEARN_BACKEND_URL"].rstrip("/")
    initial = Path(args.init_model_path or args.model_path).read_bytes()
    coordinator = TreeCoordinator(contract, initial)
    interceptor = interceptor_from_env()
    server = grpc.server(ThreadPoolExecutor(max_workers=contract.clients_per_round * 2 + 10),
        interceptors=[interceptor] if interceptor else [],
        options=[("grpc.max_receive_message_length", 128 * 1024), ("grpc.max_send_message_length", 32 * 1024 * 1024)])
    add_tree_services(server, coordinator, partition_extractor_from_env())
    address = f"0.0.0.0:{args.port}"
    if use_tls:
        root = os.environ.get("FEDLEARN_GRPC_ROOT_CERT")
        credentials = grpc.ssl_server_credentials([
            (Path(os.environ["FEDLEARN_GRPC_SERVER_KEY"]).read_bytes(),
             Path(os.environ["FEDLEARN_GRPC_SERVER_CERT"]).read_bytes())],
            root_certificates=Path(root).read_bytes() if root else None,
            require_client_auth=os.environ.get("FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH") == "1")
        port = server.add_secure_port(address, credentials)
    else:
        port = server.add_insecure_port(address)
    if port == 0:
        raise RuntimeError("tree server could not bind its port")
    server.start()
    print(f"XGBoost tree server listening on {address}; contract={coordinator.contract_id}", flush=True)
    history = []
    try:
        while True:
            with coordinator.condition:
                coordinator.check_deadline()
                if coordinator.state == p.TreeModel.FAILED:
                    raise RuntimeError("tree run failed before finalization")
                while coordinator.completed_rounds:
                    round_number, model, updates = coordinator.completed_rounds.pop(0)
                    record = dict(round=round_number, model_sha256=hashlib.sha256(model).hexdigest(), updates=updates,
                                  trees=load_model(model).num_boosted_rounds())
                    history.append(record)
                    print(json.dumps(record), flush=True)
                if coordinator.state == p.TreeModel.FINALIZING:
                    break
                coordinator.condition.wait(.25)
        model = load_model(coordinator.parent)
        final = bytes(model.save_raw(raw_format="ubj"))
        path = Path(args.model_path)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(final); temporary.replace(path)
        card = dict(recipe_key="XGBOOST", strategy="XgbBagging", rounds=contract.num_rounds,
                    xgboost_version="3.4.1", feature_names=list(contract.tree_training.feature_names),
                    objective="binary:logistic", cohort_size=contract.clients_per_round,
                    seed=str(contract.seed), accepted_updates=coordinator.accepted_count, history=history,
                    evaluation="No server training or evaluation data; held-out evaluation is external")
        response = requests.post(f"{backend}/api/internal/projects/{args.project_id}/artifacts",
            files={"model": (path.name, final, "application/octet-stream")},
            data={"kind": "FULL_CHECKPOINT", "recipeKey": "XGBOOST", "evalCard": json.dumps(card)},
            headers=callback_headers, timeout=120)
        response.raise_for_status()
        print("Registered tree checkpoint: " + response.text, flush=True)
        # Do not report fabricated accuracy/loss without a held-out evaluation set.
        response = requests.post(f"{backend}/api/internal/results/{args.project_id}/finished",
                                 headers=callback_headers, timeout=30)
        response.raise_for_status()
        with coordinator.condition:
            coordinator.state = p.TreeModel.COMPLETE
            coordinator.condition.notify_all()
        time.sleep(2)
        return 0
    except Exception:
        with coordinator.condition:
            coordinator.state = p.TreeModel.FAILED
            coordinator.condition.notify_all()
        raise
    finally:
        server.stop(1).wait()


def client_main():
    from fedlearn.tree.client import main
    return main()
