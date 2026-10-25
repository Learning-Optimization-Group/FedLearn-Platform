"""Generated-data checks for prospective pilot mechanisms; no medical inputs."""
import pytest
import torch
from torch.nn import functional as F

from capstone_pilot_mechanisms import ascent_upload, public_projection


def test_public_projection_is_orthonormal_nested_and_data_independent():
    small = public_projection(8, 2, seed=17)
    large = public_projection(8, 4, seed=17)
    assert torch.allclose(small.T @ small, torch.eye(2, dtype=torch.float64), atol=1e-12)
    assert torch.equal(small, large[:, :2])
    x = torch.arange(32, dtype=torch.float64).reshape(4, 8)
    assert torch.all(torch.linalg.vector_norm(x @ small, dim=1) <= torch.linalg.vector_norm(x, dim=1) + 1e-12)
    assert not torch.equal(small, public_projection(8, 2, seed=18))


def test_full_width_is_the_exact_unprojected_control():
    assert torch.equal(public_projection(8, 8, seed=17), torch.eye(8, dtype=torch.float64))


@pytest.mark.parametrize("dimensions", [(0, 1), (8, 0), (2, 3), (True, 1), (8, 1.5)])
def test_projection_refuses_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        public_projection(*dimensions, seed=17)


def example():
    return {
        "weight": torch.zeros((2, 1), dtype=torch.float64),
        "bias": torch.zeros(2, dtype=torch.float64),
        "owned_x": torch.tensor([[-1.], [1.]], dtype=torch.float64),
        "owned_y": torch.tensor([0, 1]),
        "initial_delta": torch.tensor([.04, -.04, 0., 0.], dtype=torch.float64),
        "released_delta": torch.tensor([-.01, .01, 0., 0.], dtype=torch.float64),
        "clip_norm": .1, "coalition_fraction": .2, "seed": 31,
        "steps": 12, "restarts": 3,
    }


def surrogate_loss(args, delta):
    parameters = torch.cat((args["weight"].reshape(-1), args["bias"]))
    candidate = parameters + (1. - args["coalition_fraction"]) * args["released_delta"] + args["coalition_fraction"] * delta
    count = args["weight"].numel()
    logits = args["owned_x"] @ candidate[:count].reshape(args["weight"].shape).T + candidate[count:]
    return float(F.cross_entropy(logits, args["owned_y"]))


def test_ascent_stays_in_ball_and_improves_the_declared_owned_data_objective():
    args = example()
    before = {key: value.clone() for key, value in args.items() if isinstance(value, torch.Tensor)}
    result = ascent_upload(**args)
    delta = result["delta"]
    assert torch.isfinite(delta).all()
    assert float(torch.linalg.vector_norm(delta)) <= .1 + 1e-12
    assert surrogate_loss(args, delta) > surrogate_loss(args, args["initial_delta"]) + 1e-3
    assert result["objective_after"] == pytest.approx(surrogate_loss(args, delta), abs=1e-12)
    assert result["loss_evaluations"] == 3 * 13
    assert result["gradient_evaluations"] == 3 * 12
    assert not delta.requires_grad
    assert all(torch.equal(args[key], value) for key, value in before.items())


def test_synthetic_positive_control_reverses_a_clipped_mean_classifier():
    args = example()
    result = ascent_upload(**args)
    # Ten clients; eight benign uploads and two coordinated constrained uploads.
    honest = args["released_delta"]
    aggregate = .8 * honest + .2 * result["delta"]
    x = torch.tensor([[-2.], [-.5], [.5], [2.]], dtype=torch.float64)
    y = torch.tensor([0, 0, 1, 1])
    clean_logits = x @ honest[:2].reshape(2, 1).T + honest[2:]
    attacked_logits = x @ aggregate[:2].reshape(2, 1).T + aggregate[2:]
    assert (clean_logits.argmax(1) == y).all()
    assert (attacked_logits.argmax(1) != y).all()
    assert float(F.cross_entropy(attacked_logits, y)) > float(F.cross_entropy(clean_logits, y))


def test_mechanisms_are_reproducible_and_leave_global_rng_untouched():
    torch.manual_seed(81)
    state = torch.random.get_rng_state().clone()
    projection = public_projection(8, 2, seed=17)
    first = ascent_upload(**example())
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.randn(20)
    assert torch.equal(public_projection(8, 2, seed=17), projection)
    assert torch.equal(ascent_upload(**example())["delta"], first["delta"])


def test_missing_history_is_exactly_the_zero_history_forecast():
    args = example()
    args.pop("released_delta")
    missing = ascent_upload(**args)
    explicit = ascent_upload(**args, released_delta=torch.zeros(4, dtype=torch.float64))
    assert torch.equal(missing["delta"], explicit["delta"])
    assert missing["objective_after"] == explicit["objective_after"]


def test_wide_head_projects_oversized_initialization_and_retains_its_objective():
    args = example()
    args["weight"] = torch.zeros((2, 512), dtype=torch.float64)
    args["owned_x"] = torch.zeros((2, 512), dtype=torch.float64)
    args["owned_x"][:, 0] = torch.tensor([-1., 1.])
    args["initial_delta"] = torch.ones(1026, dtype=torch.float64) * 3.
    args["released_delta"] = torch.zeros(1026, dtype=torch.float64)
    result = ascent_upload(**args)
    projected = args["initial_delta"] * .1 / torch.linalg.vector_norm(args["initial_delta"])
    assert result["delta"].shape == (1026,)
    assert float(torch.linalg.vector_norm(result["delta"])) <= .1 + 1e-12
    assert result["objective_before"] == pytest.approx(surrogate_loss(args, projected), abs=1e-12)
    assert result["objective_after"] >= surrogate_loss(args, projected) - 1e-12


def test_zero_gradient_remains_finite_with_the_declared_budget():
    args = example()
    args["owned_x"].zero_()
    args["initial_delta"].zero_()
    args["released_delta"].zero_()
    args["restarts"] = 1
    result = ascent_upload(**args)
    assert torch.count_nonzero(result["delta"]) == 0
    assert result["loss_evaluations"] == args["steps"] + 1
    assert result["gradient_evaluations"] == args["steps"]


@pytest.mark.parametrize("key,value", [
    ("clip_norm", 0.), ("clip_norm", float("nan")),
    ("coalition_fraction", 0.), ("coalition_fraction", 1.1),
    ("steps", 0), ("steps", True), ("restarts", 0),
    ("owned_y", torch.tensor([0.2, 1.])),
    ("owned_y", torch.tensor([0, 2])),
    ("owned_x", torch.tensor([[float("nan")], [1.]])),
    ("initial_delta", torch.zeros(3)),
])
def test_ascent_refuses_unqualified_inputs(key, value):
    args = example()
    args[key] = value
    with pytest.raises(ValueError):
        ascent_upload(**args)
