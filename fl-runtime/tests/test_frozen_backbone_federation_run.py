"""DA-14 Ph3.0/Ph3.3 — derivation ACTUALLY RUNS: a real, converging, multi-round federated loop.

The Ph3.0 tests prove one round composes; this proves a *derived* model (frozen backbone from a
shared BASE_REF + a trainable head) actually LEARNS over many rounds of head-only FedAvg — real
local head training (SGD + cross-entropy), aggregation through the real FedAvgAggregator, and
apply_trainable_subset every round — while the frozen backbone never rides the wire and never
changes. Self-contained + seeded (no external dataset/weights), so it is deterministic and fast.

Scope: this exercises the FL *semantics* of derivation end to end through the real aggregator. The
production client.py/fl_server.py wiring (so a spawned project federates head-only over gRPC) is a
separate, additive step (mirrors the existing LLM_LORA adapter-subset path).
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402
from fedlearn.backbone.distribution import serialize_backbone  # noqa: E402
from fedlearn.estimators.params import trainable_state  # noqa: E402
from fedlearn.server.strategy import FedAvgAggregator  # noqa: E402
from fedlearn.server.subset_federation import apply_trainable_subset, guard_client_updates  # noqa: E402

D_IN, D_HIDDEN, N_CLASSES = 16, 8, 3
HEAD_KEYS = {"head.weight", "head.bias"}


def _make_task(seed=0, n=360):
    """A shared frozen backbone (a BASE_REF blob) + a fixed ground-truth head that defines labels
    from the backbone's features — so a trainable linear head CAN fit it (the federation has a real
    target to converge to)."""
    torch.manual_seed(seed)
    base = recipes.build_frozen_backbone_model(N_CLASSES, d_in=D_IN, d_hidden=D_HIDDEN)
    blob = serialize_backbone(base)
    gt_head = nn.Linear(D_HIDDEN, N_CLASSES)  # defines the labels; never shared
    X = torch.randn(n, D_IN)
    with torch.no_grad():
        y = gt_head(torch.relu(base.backbone(X))).argmax(dim=1)
    return blob, X, y


def _derived(blob):
    return recipes.build_frozen_backbone_model(N_CLASSES, backbone_bytes=blob, d_in=D_IN, d_hidden=D_HIDDEN)


def _accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def _train_head_locally(model, X, y, epochs=5, lr=0.5):
    """Real local training — only the head has requires_grad, so only the head moves."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()


def test_derived_model_head_federation_converges_over_rounds():
    torch.manual_seed(0)
    blob, X, y = _make_task()
    Xtr, ytr, Xte, yte = X[:300], y[:300], X[300:], y[300:]
    K, ROUNDS = 3, 15
    client_data = [(Xtr[i::K], ytr[i::K]) for i in range(K)]

    server = _derived(blob)
    clients = [_derived(blob) for _ in range(K)]
    backbone_w0 = server.backbone.weight.detach().clone()
    initial_acc = _accuracy(server, Xte, yte)

    for _ in range(ROUNDS):
        global_head = trainable_state(server)
        updates = []
        for ci, (model, (cx, cy)) in enumerate(zip(clients, client_data)):
            apply_trainable_subset(model, global_head)   # start the round from the consensus head
            _train_head_locally(model, cx, cy)
            u = trainable_state(model)
            assert set(u.keys()) == HEAD_KEYS             # head-only wire, every client, every round
            updates.append((f"c{ci}", u, cx.shape[0]))
        guard_client_updates([u for _, u, _ in updates], server)
        apply_trainable_subset(server, FedAvgAggregator().aggregate(updates))

    final_acc = _accuracy(server, Xte, yte)
    # Derivation RAN and LEARNED: the federated head climbs well above its random-init accuracy.
    assert final_acc > initial_acc + 0.2
    assert final_acc > 0.7
    # The frozen backbone never rode the wire and never changed across any round.
    assert torch.equal(server.backbone.weight.detach(), backbone_w0)
    assert server.backbone.weight.requires_grad is False
