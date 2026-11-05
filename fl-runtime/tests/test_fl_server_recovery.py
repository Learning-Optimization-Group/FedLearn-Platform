"""The actual runtime entrypoint must pass recovery state to the production coordinator."""
from collections import OrderedDict
import json
import sys

import numpy as np
import pytest
import torch

import fl_server
import recipes
import recovery as recovery_module
from fedlearn.server.server import build_coordinator
from fedlearn.estimators.params import trainable_state


class ReachedServer(RuntimeError):
    pass


@pytest.fixture
def invocation(tmp_path, monkeypatch):
    model = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")
    initial = tmp_path / "initial.npz"
    np.savez(initial, **{name.replace(".", "__DOT__"): value.numpy()
                       for name, value in trainable_state(model).items()})
    monkeypatch.setenv("FEDLEARN_RUN_ID", "85456344-c974-4aba-9f4f-9ba3683b98fc")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    argv = ["fl_server.py", "--model-path", str(initial), "--project-id",
            "9b1e6d3a-2c47-4f85-a0d9-3e7c1b5a8f64", "--model-type", "TINYNET_GOLDEN",
            "--model-name", "tinynet_golden", "--strategy", "FedAvg", "--num-rounds", "3",
            "--min-clients", "2", "--checkpoint-directory", str(tmp_path / "private")]
    return argv, initial


def test_entrypoint_forwards_recovery_and_restores_original_seed_and_round(invocation, monkeypatch):
    argv, initial = invocation
    observed = []

    def start_server(**kwargs):
        config, strategy = kwargs["config"], kwargs["strategy"]
        assert config.checkpoint_directory is not None
        assert config.checkpoint_identity.run_id == "85456344-c974-4aba-9f4f-9ba3683b98fc"
        # The entrypoint must retain its process lease while running the server.
        args = fl_server.build_arg_parser().parse_args(argv[1:])
        with pytest.raises(ValueError, match="another.*process"):
            with recovery_module.runtime_recovery(args):
                pytest.fail("entrypoint released its owner lease before training")
        strategy.evaluate_fn = None
        coordinator = build_coordinator(strategy, config)
        if not config.resume_from_checkpoint:
            coordinator.set_initial_parameters(strategy.initial_parameters)
            for client, increment in (("a", .1), ("b", .2)):
                values = OrderedDict((name, value.clone() + increment)
                                     for name, value in strategy.initial_parameters.items())
                assert coordinator.submit_client_update(client, values, 2, 1)
            coordinator._aggregation_worker.join(3)
            assert not coordinator._aggregation_worker.is_alive()
        assert coordinator.current_round == 2
        assert coordinator.get_latest_metrics(1) is None
        observed.append((config.checkpoint_identity, coordinator.get_global_model_params()))
        raise ReachedServer("entrypoint forwarded recovery")

    monkeypatch.setattr(fl_server.fl.server, "start_server", start_server)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ReachedServer):
        fl_server.main()
    seed_before = json.loads((initial.parent / "private" / "session.json").read_text())["seed"]
    monkeypatch.setattr(sys, "argv", argv + ["--resume-from-checkpoint", "--port", "51235"])
    with pytest.raises(ReachedServer):
        fl_server.main()
    seed_after = json.loads((initial.parent / "private" / "session.json").read_text())["seed"]
    assert seed_before == seed_after
    assert len(observed) == 2 and observed[0][0] == observed[1][0]
    for name in observed[0][1]:
        assert torch.equal(observed[0][1][name], observed[1][1][name])


def test_changed_model_after_loading_refuses_before_network_server(invocation, monkeypatch):
    argv, initial = invocation
    real_select = fl_server.select_strategy
    reached = []

    def changed_input(*args, **kwargs):
        strategy = real_select(*args, **kwargs)
        initial.write_bytes(b"initial model changed after it was loaded")
        return strategy

    def start_server(**kwargs):
        reached.append(True)
        raise AssertionError("network server was started with changed inputs")

    monkeypatch.setattr(fl_server, "select_strategy", changed_input)
    monkeypatch.setattr(fl_server.fl.server, "start_server", start_server)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="initial model changed"):
        fl_server.main()
    assert reached == []
