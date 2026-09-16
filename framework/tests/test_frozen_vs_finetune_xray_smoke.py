"""Smoke test for the Phase-2 `frozen_vs_finetune_xray` benchmark — the non-DP 2x2 that separates
"pretraining helped" from "freezing helped" on real frozen-backbone features.

Phase 1 established two things this harness must respect, and these tests pin both:
  * the pretrained-vs-random gap is a DATA-EFFICIENCY effect, so per-client shard size must be a
    controllable factor rather than a fixed consequence of the pool size;
  * a single shared learning rate is NOT a fair control across arms (lr=0.5 diverges on random
    features whose scale is larger), so the harness must select an LR per arm.

Assertions:
1. Dirichlet partitioning yields disjoint shards, is seed-deterministic, and honours a per-client cap.
2. Lower alpha produces more label skew (the non-IID knob actually does something).
3. The four arms expose the trainable-parameter surface they claim (frozen arms train only the head).
4. Head-only wire bytes are measured with the production codec and are far smaller than the full model.
"""
import os

import pytest

torch = pytest.importorskip("torch")

from benchmarks.frozen_vs_finetune_xray import (  # noqa: E402
    ARMS,
    _accepted_kwargs,
    _emit_run,
    should_stop_early,
    arm_spec,
    build_model,
    dirichlet_partition,
    fit_head,
    head_auc,
    auc_from_logits,
    label_skew,
    round_wire_bytes,
    run_arm,
    run_full_arm,
    select_lr,
)


def test_dirichlet_partition_is_disjoint_and_covers_every_index():
    labels = torch.tensor([0] * 50 + [1] * 50)

    parts = dirichlet_partition(labels, num_clients=5, alpha=1.0, seed=0)

    assert len(parts) == 5
    flat = [i for p in parts for i in p]
    assert len(flat) == len(set(flat)), "shards must be disjoint (no index in two clients)"
    assert set(flat) == set(range(100)), "every example must land with exactly one client"


def test_dirichlet_partition_is_seed_deterministic():
    labels = torch.tensor([0] * 40 + [1] * 40)

    a = dirichlet_partition(labels, num_clients=4, alpha=0.5, seed=7)
    b = dirichlet_partition(labels, num_clients=4, alpha=0.5, seed=7)
    c = dirichlet_partition(labels, num_clients=4, alpha=0.5, seed=8)

    assert [sorted(p) for p in a] == [sorted(p) for p in b], "same seed must reproduce the split"
    assert [sorted(p) for p in a] != [sorted(p) for p in c], "different seed must change the split"


def test_per_client_cap_bounds_shard_size():
    """Phase 1's finding: shard size is the discriminating factor, so it must be directly settable."""
    labels = torch.tensor([0] * 100 + [1] * 100)

    parts = dirichlet_partition(labels, num_clients=4, alpha=1.0, seed=0, per_client=10)

    assert all(len(p) == 10 for p in parts), f"expected 10 per client, got {[len(p) for p in parts]}"
    flat = [i for p in parts for i in p]
    assert len(flat) == len(set(flat)), "capping must not duplicate indices across clients"


def test_lower_alpha_produces_more_label_skew():
    labels = torch.tensor([0] * 200 + [1] * 200)

    iid_like = label_skew(dirichlet_partition(labels, num_clients=8, alpha=100.0, seed=0), labels)
    skewed = label_skew(dirichlet_partition(labels, num_clients=8, alpha=0.1, seed=0), labels)

    assert skewed > iid_like, f"alpha=0.1 should be more skewed than alpha=100 ({skewed} vs {iid_like})"


def test_the_2x2_defines_exactly_four_arms_over_pretraining_and_freezing():
    assert set(ARMS) == {"A", "B", "C", "D"}
    grid = {(arm_spec(a)["pretrained"], arm_spec(a)["mode"]) for a in ARMS}
    assert grid == {(False, "frozen"), (True, "frozen"), (True, "full"), (False, "full")}, (
        "the 2x2 must cover both levels of BOTH factors, else it cannot separate "
        "'pretraining helped' from 'freezing helped'"
    )


@pytest.mark.parametrize("arm", ["A", "B"])
def test_frozen_arms_train_only_the_head(arm):
    m = build_model(arm, feat_dim=16, n_classes=2, seed=0)

    trainable = {n for n, p in m.named_parameters() if p.requires_grad}

    assert trainable == {"weight", "bias"}, f"arm {arm} must federate only the head, got {trainable}"


@pytest.mark.parametrize("arm", ["C", "D"])
def test_full_finetune_arms_train_the_backbone_too(arm):
    m = build_model(arm, feat_dim=16, n_classes=2, backbone_name="resnet18", seed=0)

    trainable = {n for n, p in m.named_parameters() if p.requires_grad}

    assert len(trainable) > 20, f"arm {arm} must fine-tune the whole model, got {len(trainable)} tensors"
    assert any(n.startswith("conv1") for n in trainable), "backbone conv weights must be trainable"


def test_head_only_round_is_orders_of_magnitude_cheaper_on_the_wire():
    """Measured with the production safetensors codec, not an analytic estimate."""
    frozen = round_wire_bytes(build_model("B", feat_dim=512, n_classes=2, seed=0))
    full = round_wire_bytes(build_model("C", feat_dim=512, n_classes=2, backbone_name="resnet18", seed=0))

    assert frozen < full / 100, f"expected >100x wire saving, got {full / frozen:.1f}x"


def _separable_features(n=240, dim=32, scale=1.0, seed=0):
    """Linearly separable features at a controllable SCALE — the variable that broke arm A in Phase 1
    (random-backbone features had std 1.34 vs 0.93 for pretrained, and a shared lr=0.5 diverged).

    The class DIRECTION is fixed independently of ``seed`` so that a train set and a test set drawn
    with different seeds still share the same decision rule — otherwise the two splits encode
    unrelated tasks and a correctly-trained model scores below chance on the other.
    """
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=torch.Generator().manual_seed(12345))
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    x = torch.randn(n, dim, generator=g) * scale
    x = x + (y.float().unsqueeze(1) * 2 - 1) * direction * 0.8
    return x, y


def test_select_lr_rejects_a_diverging_learning_rate():
    x, y = _separable_features(scale=1.34, seed=0)

    chosen = select_lr(x, y, candidates=[100.0, 0.05], seed=0)

    assert chosen == 0.05, f"a diverging lr must never be selected, got {chosen}"


def test_selected_lr_keeps_the_random_scale_arm_above_chance():
    """Phase 1's actual failure: arm A returned exactly 0.500 AUC because lr was tuned for the other
    arm. With per-arm selection the same features must train."""
    x, y = _separable_features(scale=1.34, seed=1)

    lr = select_lr(x, y, candidates=[100.0, 10.0, 0.5, 0.05, 0.01], seed=1)
    auc = head_auc(fit_head(x, y, lr=lr, epochs=40, seed=1), x, y)

    assert auc > 0.75, f"selected lr={lr} still leaves the arm near chance (auc={auc:.3f})"


def test_federated_frozen_arm_converges_and_records_a_per_round_curve():
    xtr, ytr = _separable_features(n=240, scale=1.0, seed=2)
    xte, yte = _separable_features(n=120, scale=1.0, seed=3)

    out = run_arm("B", train_x=xtr, train_y=ytr, test_x=xte, test_y=yte,
                  clients=4, clients_per_round=4, alpha=1.0, rounds=6, local_epochs=2, seed=0)

    assert len(out["per_round"]) == 6, "every round must be recorded, not just the final score"
    assert out["final_auc"] > 0.75, f"federated head failed to learn (auc={out['final_auc']:.3f})"
    assert out["per_round"][-1]["auc"] >= out["per_round"][0]["auc"], "accuracy should not regress overall"


def test_federated_run_records_real_wire_bytes_and_the_selected_lr():
    xtr, ytr = _separable_features(n=160, seed=4)
    xte, yte = _separable_features(n=80, seed=5)

    out = run_arm("B", train_x=xtr, train_y=ytr, test_x=xte, test_y=yte,
                  clients=4, clients_per_round=2, alpha=1.0, rounds=3, local_epochs=1, seed=0)

    assert out["meta"]["wire_bytes_per_client_round"] > 0
    assert out["meta"]["selected_lr"] in out["meta"]["lr_candidates"]
    assert out["meta"]["clients_per_round"] == 2
    assert out["meta"]["seed"] == 0


def test_per_client_shard_size_is_recorded_because_phase1_made_it_the_key_factor():
    xtr, ytr = _separable_features(n=400, seed=6)
    xte, yte = _separable_features(n=80, seed=7)

    out = run_arm("B", train_x=xtr, train_y=ytr, test_x=xte, test_y=yte,
                  clients=4, clients_per_round=4, alpha=1.0, rounds=2, local_epochs=1,
                  per_client=25, seed=0)

    assert out["meta"]["per_client"] == 25
    assert out["meta"]["shard_sizes"] == [25, 25, 25, 25]


@pytest.mark.parametrize("arm", ["C", "D"])
def test_full_arms_refuse_to_run_from_pre_extracted_features(arm):
    """A full fine-tune must see IMAGES — pre-extracted features come from a frozen backbone and
    cannot train one. Silently accepting them would report a head-only run as a fine-tune baseline
    and invalidate the B-vs-C comparison."""
    x, y = _separable_features(n=40, seed=8)

    with pytest.raises(ValueError, match="run_full_arm"):
        run_arm(arm, train_x=x, train_y=y, test_x=x, test_y=y,
                clients=2, clients_per_round=2, alpha=1.0, rounds=1, local_epochs=1, seed=0)


def test_shard_size_is_independent_of_client_count_and_alpha():
    """The Phase-1 lever must be settable on its own. Partitioning the WHOLE pool by Dirichlet and
    then truncating couples shard size to the draw, so a thin-shard sweep at realistic client counts
    fails outright — which is exactly the configuration the experiment needs."""
    labels = torch.tensor([0] * 700 + [1] * 700)

    parts = dirichlet_partition(labels, num_clients=20, alpha=1.0, seed=0, per_client=10)

    assert [len(p) for p in parts] == [10] * 20
    flat = [i for p in parts for i in p]
    assert len(flat) == len(set(flat)), "shards must stay disjoint when sized explicitly"


def test_alpha_still_controls_label_skew_when_shard_size_is_fixed():
    labels = torch.tensor([0] * 700 + [1] * 700)

    iid_like = dirichlet_partition(labels, num_clients=20, alpha=100.0, seed=0, per_client=20)
    skewed = dirichlet_partition(labels, num_clients=20, alpha=0.05, seed=0, per_client=20)

    assert label_skew(skewed, labels) > label_skew(iid_like, labels), (
        "with shard size fixed, alpha must still control the LABEL MIX inside each shard"
    )


def _relu_like_features(n=400, dim=512, seed=0):
    """Post-ReLU frozen-backbone features: NON-NEGATIVE with a large positive mean, the shape real
    random-backbone features actually have (mean 1.24, std 1.34, 512-d). Zero-mean Gaussian
    fixtures do not reproduce the conditioning problem this causes for a linear head."""
    g = torch.Generator().manual_seed(seed)
    direction = torch.randn(dim, generator=torch.Generator().manual_seed(999))
    y = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)]).long()
    base = torch.relu(torch.randn(n, dim, generator=g) * 1.3 + 1.2)
    return base + (y.float().unsqueeze(1) * 2 - 1) * torch.relu(direction) * 0.35, y


def test_run_flags_an_arm_that_has_not_converged_within_the_round_budget():
    """Measured on real features: random-backbone features need ~1000 gradient steps to reach
    AUC 0.84 centrally, but a 20-round x 3-local-epoch federation gives ~60. An arm cut off by the
    budget must be REPORTED as budget-limited — reporting it as simply worse would manufacture a
    result that the data does not support."""
    xtr, ytr = _relu_like_features(n=400, seed=10)
    xte, yte = _relu_like_features(n=200, seed=11)

    short = run_arm("A", train_x=xtr, train_y=ytr, test_x=xte, test_y=yte,
                    clients=20, clients_per_round=10, alpha=1.0, rounds=2, local_epochs=1,
                    per_client=10, seed=0)

    assert short["meta"]["converged"] is False, "a 2-round run cannot honestly be called converged"
    assert "auc_improvement_last_half" in short["meta"]


def _tiny_imagefolder(root, *, per_class_train=12, per_class_test=6, size=32, seed=0):
    """A tiny deterministic 2-class ImageFolder with a real class signal (NORMAL darker,
    PNEUMONIA brighter), so a full fine-tune can separate it without the real dataset."""
    from PIL import Image
    g = torch.Generator().manual_seed(seed)
    for split, per in (("train", per_class_train), ("test", per_class_test)):
        for cls, base in (("NORMAL", 60), ("PNEUMONIA", 195)):
            d = os.path.join(root, split, cls)
            os.makedirs(d, exist_ok=True)
            for i in range(per):
                arr = (torch.randn(size, size, 3, generator=g) * 12 + base).clamp(0, 255)
                Image.fromarray(arr.numpy().astype("uint8")).save(os.path.join(d, f"{i}.png"))
    return root


def test_full_finetune_arm_actually_updates_the_backbone(tmp_path):
    """The defining contract of arms C/D: unlike the frozen arms, backbone weights must MOVE."""
    pytest.importorskip("torchvision")
    root = _tiny_imagefolder(str(tmp_path / "ds"))

    out = run_full_arm("C", data_dir=root, clients=2, clients_per_round=2, alpha=1.0,
                       rounds=1, local_epochs=1, img_size=32, batch_size=8, seed=0, device="cpu")

    assert out["meta"]["backbone_changed"] is True, "a fine-tune that leaves the backbone fixed is not a fine-tune"


def test_full_arm_returns_the_same_result_schema_as_the_frozen_arms(tmp_path):
    """B-vs-C is only comparable if both arms report the same fields."""
    pytest.importorskip("torchvision")
    root = _tiny_imagefolder(str(tmp_path / "ds2"), seed=1)
    xtr, ytr = _separable_features(n=80, seed=20)

    full = run_full_arm("D", data_dir=root, clients=2, clients_per_round=2, alpha=1.0,
                        rounds=1, local_epochs=1, img_size=32, batch_size=8, seed=0, device="cpu")
    frozen = run_arm("B", train_x=xtr, train_y=ytr, test_x=xtr, test_y=ytr,
                     clients=2, clients_per_round=2, alpha=1.0, rounds=1, local_epochs=1, seed=0)

    assert set(full) == set(frozen), f"schema drift: {set(full) ^ set(frozen)}"
    for k in ("clients", "rounds", "seed", "wire_bytes_per_client_round", "converged"):
        assert k in full["meta"], f"meta missing {k}"


def test_full_arm_wire_cost_dwarfs_the_head_only_arm(tmp_path):
    """The communication axis of the B-vs-C trade, measured with the production codec."""
    pytest.importorskip("torchvision")
    root = _tiny_imagefolder(str(tmp_path / "ds3"), seed=2)
    xtr, ytr = _separable_features(n=80, dim=512, seed=21)

    full = run_full_arm("D", data_dir=root, clients=2, clients_per_round=2, alpha=1.0,
                        rounds=1, local_epochs=1, img_size=32, batch_size=8, seed=0, device="cpu")
    frozen = run_arm("B", train_x=xtr, train_y=ytr, test_x=xtr, test_y=ytr,
                     clients=2, clients_per_round=2, alpha=1.0, rounds=1, local_epochs=1, seed=0)

    ratio = full["meta"]["wire_bytes_per_client_round"] / frozen["meta"]["wire_bytes_per_client_round"]
    assert ratio > 100, f"expected a >100x wire gap between full and head-only, got {ratio:.1f}x"


def test_dispatch_passes_only_kwargs_the_target_runner_accepts():
    """The CLI hands one kwarg bag to both runners, but they take different options (images-only
    args like num_workers/img_size/batch_size mean nothing to the feature-space runner). Filtering
    by the callee's real signature is what keeps a new option from crashing the sweep mid-run."""
    bag = {"clients": 4, "rounds": 2, "num_workers": 4, "img_size": 224, "batch_size": 32}

    accepted = _accepted_kwargs(run_arm, bag)

    assert "clients" in accepted and "rounds" in accepted
    assert not ({"num_workers", "img_size", "batch_size"} & set(accepted)), (
        f"image-only kwargs leaked into the feature-space runner: {accepted}"
    )
    assert set(_accepted_kwargs(run_full_arm, bag)) == set(bag), "full runner accepts them all"


def test_each_run_is_written_to_disk_as_it_completes(tmp_path):
    """A 24-cell sweep is ~2h. Writing the payload only at the end means any interruption discards
    every finished cell — which is exactly what happened on the first attempt, forcing the results
    to be reconstructed from stdout. Each cell must be durable the moment it finishes."""
    out_dir = str(tmp_path / "runs")
    run = {"arm": "B", "final_auc": 0.97, "per_round": [{"round": 1, "auc": 0.97}],
           "meta": {"per_client": 10, "seed": 0}}

    path = _emit_run(out_dir, run)

    import json
    assert os.path.exists(path), "the completed run must be on disk immediately"
    reloaded = json.load(open(path))
    assert reloaded["final_auc"] == 0.97
    assert reloaded["arm"] == "B"


def test_emitted_run_filenames_do_not_collide_across_cells(tmp_path):
    """Arm, shard size and seed all vary in the sweep; two cells must never overwrite each other."""
    out_dir = str(tmp_path / "runs")
    cells = [("B", 10, 0), ("B", 10, 1), ("B", 70, 0), ("C", 10, 0)]

    paths = {_emit_run(out_dir, {"arm": a, "final_auc": 0.5, "per_round": [],
                                 "meta": {"per_client": pc, "seed": sd}})
             for a, pc, sd in cells}

    assert len(paths) == len(cells), f"filename collision: only {len(paths)} files for {len(cells)} cells"


def test_early_stop_fires_when_the_metric_plateaus():
    """79% of the first 2x2 matrix never converged inside a fixed 20-round budget, so most cells
    reported where the budget stopped them rather than where the arm tops out. Training to a
    plateau instead of a fixed count is what makes cross-arm comparison valid."""
    plateaued = [0.90, 0.95, 0.970, 0.9701, 0.9702, 0.9703]

    assert should_stop_early(plateaued, patience=3, min_delta=0.001) is True


def test_early_stop_does_not_fire_while_the_metric_is_still_climbing():
    climbing = [0.60, 0.70, 0.78, 0.85, 0.90, 0.94]

    assert should_stop_early(climbing, patience=3, min_delta=0.001) is False


def test_early_stop_needs_enough_history_before_it_can_judge():
    """With fewer rounds than the patience window there is no evidence of a plateau yet."""
    assert should_stop_early([0.9, 0.9], patience=3, min_delta=0.001) is False


def test_early_stop_uses_best_so_far_not_last_so_a_dip_does_not_trigger_it():
    """A single bad round must not be read as convergence — the comparison is against the best
    value seen, so a transient dip keeps training rather than ending the run."""
    dipped = [0.80, 0.85, 0.90, 0.70, 0.91, 0.93]

    assert should_stop_early(dipped, patience=3, min_delta=0.001) is False


def test_early_stop_does_not_fire_on_a_noisy_but_improving_curve():
    """THE bug this test exists for: the first early-stop implementation compared each round against
    best-so-far, so an arm whose metric OSCILLATES (random-feature arms swing +/-0.15 AUC between
    rounds) triggered patience the moment it hit a lucky peak — reporting 'converged' on a run that
    was still climbing. Measured on real data: 125/125 arm-A runs stopped early with last-5 deltas
    of -0.03 to -0.16, i.e. pure oscillation. The criterion must judge the TREND, not the peak."""
    noisy_but_climbing = [0.50, 0.65, 0.52, 0.70, 0.58, 0.75, 0.62, 0.80, 0.68, 0.85]

    assert should_stop_early(noisy_but_climbing, patience=3, min_delta=0.001) is False


def test_early_stop_still_fires_on_a_noisy_curve_that_has_genuinely_flattened():
    """The complement: smoothing must not make the criterion never fire. Same oscillation
    amplitude, but no underlying trend."""
    noisy_flat = [0.80, 0.86, 0.79, 0.85, 0.81, 0.84, 0.80, 0.85, 0.81, 0.84, 0.80, 0.85]

    assert should_stop_early(noisy_flat, patience=4, min_delta=0.001) is True


def test_auc_handles_multiclass_via_macro_one_vs_rest():
    """The X-ray task is binary, which has repeatedly limited what this campaign can claim. Scaling to
    a many-class medical dataset requires AUC that is not hard-wired to 2 columns — a binary-only
    implementation would silently misreport rather than fail."""
    # 3 classes, perfectly separable by construction: argmax equals the label everywhere.
    logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0],
                           [8.0, 1.0, 0.0], [1.0, 8.0, 0.0], [0.0, 1.0, 8.0]])
    y = torch.tensor([0, 1, 2, 0, 1, 2])

    assert abs(auc_from_logits(logits, y) - 1.0) < 1e-6, "perfect separation must give AUC 1.0"


def test_multiclass_auc_is_chance_for_uninformative_logits():
    logits = torch.zeros(8, 4)
    y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    v = auc_from_logits(logits, y)
    assert abs(v - 0.5) < 1e-6, f"constant logits must give chance AUC, got {v}"


def test_binary_auc_is_unchanged_by_the_multiclass_generalisation():
    """Regression guard: every existing binary number in the campaign must remain reproducible."""
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.5, 0.0], [0.0, 1.5]])
    y = torch.tensor([0, 1, 0, 1])

    assert abs(auc_from_logits(logits, y) - 1.0) < 1e-6
