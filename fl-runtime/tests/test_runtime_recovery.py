"""Runtime recovery must preserve run identity and refuse changed execution inputs."""
from collections import OrderedDict
import json

import pytest
import torch

from fl_server import build_arg_parser
from recovery import runtime_recovery, source_sha256
from fedlearn.server.checkpoint import capture_round
from fedlearn.server.strategy import FedAvg


RUN_ID = "4f571769-d190-4298-830c-34d810675868"


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = tmp_path / "initial.npz"
    model.write_bytes(b"fixed initial model bytes")
    source = tmp_path / "server.py"
    source.write_text("TRAINING_VERSION = 1\n")
    return model, [("runtime/server.py", source)]


def arguments(inputs, tmp_path, *extra):
    model, _ = inputs
    return build_arg_parser().parse_args([
        "--project-id", "f7622934-14a6-49fe-91ca-55370de610dc", "--model-type", "TINYNET_GOLDEN",
        "--model-name", "tinynet_golden", "--model-path", str(model), "--strategy", "FedAvg",
        "--checkpoint-directory", str(tmp_path / "private"), *extra,
    ])


def commit_first_round(recovery, rounds=5, completed_round=1):
    parameters = OrderedDict(w=torch.zeros(1))
    strategy = FedAvg(parameters)
    snapshot = capture_round(strategy, completed_round=completed_round, total_rounds=rounds,
                             parameters=OrderedDict(w=torch.ones(1)),
                             history=[(number, None) for number in range(1, completed_round + 1)])
    recovery.store.save(snapshot)


def test_recovery_flags_are_opt_in(inputs):
    model, _ = inputs
    args = build_arg_parser().parse_args([
        "--project-id", "p1", "--model-type", "TINYNET_GOLDEN", "--model-name", "tinynet_golden",
        "--model-path", str(model),
    ])
    assert args.checkpoint_directory is None and not args.resume_from_checkpoint
    with runtime_recovery(args, env={}) as recovery:
        assert recovery is None


def test_new_run_pins_generated_seed_and_keeps_session_private(inputs, tmp_path):
    args = arguments(inputs, tmp_path)
    with runtime_recovery(args, env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]) as recovery:
        assert type(args.seed) is int and 0 <= args.seed < 2**32
        assert recovery.identity.run_id == RUN_ID
        session_path = recovery.directory / "session.json"
        session = json.loads(session_path.read_text())
        assert session["seed"] == args.seed
        assert session["configuration_sha256"] == recovery.identity.configuration_sha256
        assert session_path.stat().st_mode & 0o077 == 0
        assert recovery.directory.stat().st_mode & 0o077 == 0
        assert recovery.server_options() == {
            "checkpoint_directory": str(recovery.directory), "checkpoint_identity": recovery.identity,
            "resume_from_checkpoint": False,
        }


def test_resume_recovers_seed_and_original_configuration_identity(inputs, tmp_path):
    args = arguments(inputs, tmp_path)
    with runtime_recovery(args, env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]) as first:
        original_identity, original_seed = first.identity, args.seed
        commit_first_round(first)
    resumed_args = arguments(inputs, tmp_path, "--resume-from-checkpoint", "--port", "51234")
    with runtime_recovery(resumed_args, env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]) as resumed:
        assert resumed.identity == original_identity
        assert resumed_args.seed == original_seed
        assert resumed.server_options()["resume_from_checkpoint"] is True
        assert resumed.store.load_latest().completed_round == 1


@pytest.mark.parametrize("change", ["seed", "rounds", "strategy", "model", "source", "run"])
def test_resume_refuses_changed_inputs_without_changing_existing_state(inputs, tmp_path, change):
    args = arguments(inputs, tmp_path, "--seed", "42")
    env = {"FEDLEARN_RUN_ID": RUN_ID}
    with runtime_recovery(args, env=env, source_files=inputs[1]) as first:
        commit_first_round(first)
    extra = {"seed": ["--seed", "43"], "rounds": ["--num-rounds", "6"],
             "strategy": ["--strategy", "FedOpt"]}.get(change, [])
    if change == "model":
        (tmp_path / "private" / "initial-model.npz").write_bytes(b"different pinned initial model")
    if change == "source":
        inputs[1][0][1].write_text("TRAINING_VERSION = 2\n")
    if change == "run":
        env = {"FEDLEARN_RUN_ID": "1e2786da-12fd-4fbe-ab95-b78db884594f"}
    original = {path.name: path.read_bytes() for path in (tmp_path / "private").iterdir()}
    resumed = arguments(inputs, tmp_path, "--seed", "42", "--resume-from-checkpoint", *extra)
    with pytest.raises(ValueError, match="identity|configuration|source|initial|seed"):
        with runtime_recovery(resumed, env=env, source_files=inputs[1]):
            pytest.fail("changed run was allowed to resume")
    assert {path.name: path.read_bytes() for path in (tmp_path / "private").iterdir()} == original


@pytest.mark.parametrize("env", [{}, {"FEDLEARN_RUN_ID": "not-a-run"}])
def test_checkpoint_mode_requires_a_canonical_run_id(inputs, tmp_path, env):
    with pytest.raises(ValueError, match="FEDLEARN_RUN_ID"):
        with runtime_recovery(arguments(inputs, tmp_path), env=env, source_files=inputs[1]):
            pytest.fail("missing identity was allowed")


def test_existing_session_requires_explicit_resume(inputs, tmp_path):
    with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                          source_files=inputs[1]) as first:
        commit_first_round(first)
    with pytest.raises(ValueError, match="explicit.*resume"):
        with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                              source_files=inputs[1]):
            pytest.fail("existing session silently reused")


def test_resume_without_committed_round_refuses_instead_of_starting_over(inputs, tmp_path):
    with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                          source_files=inputs[1]):
        pass
    with pytest.raises(ValueError, match="no.*checkpoint"):
        with runtime_recovery(arguments(inputs, tmp_path, "--resume-from-checkpoint"),
                              env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]):
            pytest.fail("empty recovery silently restarted training")


def test_two_process_owners_cannot_share_one_checkpoint_directory(inputs, tmp_path):
    with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                          source_files=inputs[1]):
        with pytest.raises(ValueError, match="another.*process|already.*owned"):
            with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                                  source_files=inputs[1]):
                pytest.fail("two runtime owners were admitted")


def test_owner_lease_is_released_after_failure(inputs, tmp_path):
    with pytest.raises(RuntimeError, match="training failed"):
        with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                              source_files=inputs[1]) as first:
            commit_first_round(first)
            raise RuntimeError("training failed")
    with runtime_recovery(arguments(inputs, tmp_path, "--resume-from-checkpoint"),
                          env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]) as resumed:
        assert resumed.store.load_latest().completed_round == 1


def test_changed_startup_inputs_are_refused_before_server_launch(inputs, tmp_path):
    with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                          source_files=inputs[1]) as recovery:
        inputs[0].write_bytes(b"model replaced during startup")
        with pytest.raises(ValueError, match="initial"):
            recovery.validate_inputs()


def test_secrets_do_not_enter_runtime_session_metadata(inputs, tmp_path):
    secret = "private-value-must-not-be-written"
    env = {"FEDLEARN_RUN_ID": RUN_ID, "FEDLEARN_FL_TOKEN_SECRET": secret,
           "FEDLEARN_INTERNAL_API_KEY": secret, "AWS_SECRET_ACCESS_KEY": secret}
    with runtime_recovery(arguments(inputs, tmp_path), env=env, source_files=inputs[1]) as recovery:
        raw = (recovery.directory / "session.json").read_text()
        assert secret not in raw
        assert str(inputs[0]) not in raw and str(inputs[1][0][1]) not in raw


@pytest.mark.parametrize("extra", [["--secure-aggregation"], ["--dp-enabled"], ["--strategy", "DeComFL"]])
def test_unqualified_recovery_modes_are_refused_before_model_loading(inputs, tmp_path, extra):
    with pytest.raises(ValueError, match="qualified|supported"):
        with runtime_recovery(arguments(inputs, tmp_path, *extra), env={"FEDLEARN_RUN_ID": RUN_ID},
                              source_files=inputs[1]):
            pytest.fail("unqualified recovery mode was admitted")


def test_source_identity_is_content_based_and_independent_of_checkout_location(tmp_path):
    first, second = tmp_path / "one.py", tmp_path / "two.py"
    first.write_text("VERSION = 1\n")
    second.write_bytes(first.read_bytes())
    assert source_sha256([("runtime/main.py", first)]) == source_sha256([("runtime/main.py", second)])
    second.write_text("VERSION = 2\n")
    assert source_sha256([("runtime/main.py", first)]) != source_sha256([("runtime/main.py", second)])


@pytest.mark.parametrize("overwritten", [b"partial final output", b"completed final output"])
def test_default_output_overwrite_does_not_destroy_initialization_needed_for_resume(inputs, tmp_path, overwritten):
    initial_bytes = inputs[0].read_bytes()
    with runtime_recovery(arguments(inputs, tmp_path), env={"FEDLEARN_RUN_ID": RUN_ID},
                          source_files=inputs[1]) as first:
        commit_first_round(first, completed_round=5)
        original_identity = first.identity
    inputs[0].write_bytes(overwritten)
    with runtime_recovery(arguments(inputs, tmp_path, "--resume-from-checkpoint"),
                          env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]) as resumed:
        assert resumed.identity == original_identity
        assert resumed.initial_model.parent == resumed.directory
        assert resumed.initial_model.read_bytes() == initial_bytes
        assert resumed.initial_model.stat().st_mode & 0o077 == 0
        assert resumed.store.load_latest().completed_round == 5


def test_explicit_initial_source_change_is_refused_even_when_a_private_copy_exists(inputs, tmp_path):
    extra = ["--init-model-path", str(inputs[0])]
    with runtime_recovery(arguments(inputs, tmp_path, *extra), env={"FEDLEARN_RUN_ID": RUN_ID},
                          source_files=inputs[1]) as first:
        commit_first_round(first)
    inputs[0].write_bytes(b"changed explicit input")
    with pytest.raises(ValueError, match="initial"):
        with runtime_recovery(arguments(inputs, tmp_path, *extra, "--resume-from-checkpoint"),
                              env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]):
            pytest.fail("changed explicit initialization was allowed")


@pytest.mark.parametrize("before,after", [("1", "0"), ("0", "1")])
def test_mtls_requirement_is_part_of_the_bound_security_policy(inputs, tmp_path, before, after):
    env = {"FEDLEARN_RUN_ID": RUN_ID, "FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH": before}
    with runtime_recovery(arguments(inputs, tmp_path), env=env, source_files=inputs[1]) as first:
        commit_first_round(first)
    env["FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH"] = after
    with pytest.raises(ValueError, match="configuration"):
        with runtime_recovery(arguments(inputs, tmp_path, "--resume-from-checkpoint"), env=env,
                              source_files=inputs[1]):
            pytest.fail("mTLS policy change was allowed")


@pytest.mark.parametrize("name", ["FEDLEARN_LLM_LORA_BASE", "FEDLEARN_LLM_LORA_SUBSET"])
def test_lora_base_and_evaluation_settings_cannot_change_on_resume(inputs, tmp_path, name):
    env = {"FEDLEARN_RUN_ID": RUN_ID, name: "original"}
    extra = ["--model-type", "LLM_LORA"]
    with runtime_recovery(arguments(inputs, tmp_path, *extra), env=env, source_files=inputs[1]) as first:
        commit_first_round(first)
    env[name] = "changed"
    with pytest.raises(ValueError, match="configuration"):
        with runtime_recovery(arguments(inputs, tmp_path, *extra, "--resume-from-checkpoint"), env=env,
                              source_files=inputs[1]):
            pytest.fail("changed model/evaluation setting was allowed")


def test_final_output_cannot_overwrite_private_recovery_state(inputs, tmp_path):
    args = arguments(inputs, tmp_path, "--model-path", str(tmp_path / "private" / "initial-model.npz"),
                     "--init-model-path", str(inputs[0]))
    with pytest.raises(ValueError, match="output.*outside|output.*private"):
        with runtime_recovery(args, env={"FEDLEARN_RUN_ID": RUN_ID}, source_files=inputs[1]):
            pytest.fail("the final model output aliases the pinned initialization")
