"""P2-2 — the CLI surface the backend shells out to.

The backend never imports the framework; it runs run_fl_server.sh -> fl_server.py. So secure
aggregation is only reachable by a deployment if these flags exist and reach ServerConfig.
"""
import pytest

from fl_server import build_arg_parser


def _parse(*extra):
    base = ["--project-id", "p1", "--model-type", "MLP", "--strategy", "DeComFL",
            "--model-path", "/tmp/m.pt", "--model-name", "m"]
    return build_arg_parser().parse_args(base + list(extra))


def test_secure_aggregation_is_off_by_default():
    """Turning it on changes the wire format for gradient submissions, so an existing
    deployment that upgrades must not suddenly refuse its own clients."""
    args = _parse()
    assert args.secure_aggregation is False


def test_the_flag_turns_it_on():
    assert _parse("--secure-aggregation").secure_aggregation is True


def test_the_threshold_is_settable_and_defaults_to_two():
    assert _parse().secure_agg_threshold == 2
    assert _parse("--secure-agg-threshold", "3").secure_agg_threshold == 3
