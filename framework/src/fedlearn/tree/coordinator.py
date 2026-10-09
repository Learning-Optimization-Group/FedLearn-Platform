"""Fixed-cohort tree rounds with parent binding and idempotent acceptance."""
import hashlib
import threading
import time

from fedlearn.communication.generated import tree_pb2 as p
from .contract import contract_id, validate_contract
from .model import _xgb, canonical, merge_trees, validate_model


class TreeCoordinator:
    def __init__(self, contract, initial_ubj, *, clock=time.time):
        self.contract = validate_contract(contract)
        self.contract_id = contract_id(contract)
        self.initial_ubj = initial_ubj
        if hashlib.sha256(initial_ubj).hexdigest() != contract.tree_training.initial_model_sha256:
            raise ValueError("initial tree artifact digest mismatch")
        booster = _xgb().Booster(params={"device": "cpu", "nthread": 1})
        booster.load_model(bytearray(initial_ubj))
        self.parent = canonical(validate_model(bytes(booster.save_raw(raw_format="json"))))
        self.server_round = 1
        self.state = p.TreeModel.WAITING
        self.clock = clock
        self.deadline = clock() + contract.round.timeout_ms / 1000
        self.condition = threading.Condition(threading.RLock())
        self.clients = {}
        self.partitions = {}
        self.pending = {}
        self.receipts = {}
        self.completed_rounds = []
        self.accepted_count = 0

    def check_deadline(self):
        with self.condition:
            if self.state in (p.TreeModel.WAITING, p.TreeModel.TRAINING) and self.clock() >= self.deadline:
                self.state = p.TreeModel.FAILED
                self.condition.notify_all()
                raise TimeoutError("fixed tree cohort did not finish before the deadline")

    def register(self, client_id, *, partition):
        with self.condition:
            self.check_deadline()
            if not client_id or len(client_id) > 128 or partition is None or type(partition) is not int or partition < 0:
                raise ValueError("invalid enrolled client identity")
            if client_id in self.clients:
                if self.clients[client_id] != partition:
                    raise ValueError("client identity changed")
                return
            if self.state != p.TreeModel.WAITING or len(self.clients) >= self.contract.clients_per_round:
                raise ValueError("tree cohort is fixed")
            if partition in self.partitions:
                raise ValueError("enrollment already bound to another client")
            self.clients[client_id] = partition
            self.partitions[partition] = client_id
            if len(self.clients) == self.contract.clients_per_round:
                self.state = p.TreeModel.TRAINING
                self.deadline = self.clock() + self.contract.round.timeout_ms / 1000
            self.condition.notify_all()

    def check_identity(self, client_id, partition):
        if client_id not in self.clients or self.clients[client_id] != partition:
            raise ValueError("unregistered or mismatched enrolled identity")

    def submit(self, request):
        with self.condition:
            if request.run_id != self.contract.run_id or request.contract_id != self.contract_id:
                raise ValueError("run or contract mismatch")
            if request.client_id not in self.clients:
                raise ValueError("unregistered client")
            key = (request.client_id, request.server_round)
            fingerprint = hashlib.sha256(request.SerializeToString(deterministic=True)).hexdigest()
            if key in self.receipts:
                if self.receipts[key] != fingerprint:
                    raise ValueError("conflicting duplicate contribution")
                return p.TreeContributionAck.ALREADY_ACCEPTED
            self.check_deadline()
            if self.state != p.TreeModel.TRAINING or request.server_round != self.server_round:
                raise ValueError("stale or future tree round")
            if request.parent_sha256 != hashlib.sha256(self.parent).hexdigest():
                raise ValueError("tree parent mismatch")
            if not 1 <= request.num_examples <= 1_000_000:
                raise ValueError("invalid local example count")
            blob = bytes(request.new_trees_json)
            if hashlib.sha256(blob).hexdigest() != request.delta_sha256:
                raise ValueError("tree delta digest mismatch")
            validate_model(blob, expected_trees=1)
            pending = {**self.pending, request.client_id: blob}
            merged = merge_trees(self.parent, list(pending.items())) if len(pending) == self.contract.clients_per_round else None
            self.pending = pending
            self.receipts[key] = fingerprint
            self.accepted_count += 1
            if merged is not None:
                self.completed_rounds.append((self.server_round, merged,
                    {client: hashlib.sha256(delta).hexdigest() for client, delta in sorted(self.pending.items())}))
                self.parent = merged
                self.pending.clear()
                if self.server_round == self.contract.num_rounds:
                    self.state = p.TreeModel.FINALIZING
                else:
                    self.server_round += 1
                    self.deadline = self.clock() + self.contract.round.timeout_ms / 1000
            self.condition.notify_all()
            return p.TreeContributionAck.ACCEPTED

    def snapshot(self):
        with self.condition:
            self.check_deadline()
            return p.TreeModel(state=self.state, server_round=self.server_round,
                parent_sha256=hashlib.sha256(self.parent).hexdigest(), model_json=self.parent,
                contract_id=self.contract_id, deadline_unix_ms=int(self.deadline * 1000),
                initial_model_ubj=self.initial_ubj)
