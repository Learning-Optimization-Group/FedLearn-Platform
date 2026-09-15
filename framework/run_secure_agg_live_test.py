#!/usr/bin/env python3
"""Live gRPC verification of secure aggregation (P2-2 / LightSecAgg).

WHY THIS EXISTS
    Every other test of the secure path drives the servicer in-process through a stub that fakes
    per-call identity with a shared field. That proves the protocol; it does not prove the system.
    This runs a REAL gRPC server on a REAL port with SE-14 client-auth enforcement on, and REAL
    client processes connecting over the channel with minted connection tokens.

    Four things only exist here:
      * the SE-15 partition extractor actually running -- the secure RPCs fail closed without a
        verified partition, so this is the first time that gate is satisfied rather than stubbed;
      * gRPC message sizes for the sealed-share relay;
      * deadlines and retries against the client's polling waits;
      * a real thread pool exercising the session locking.

WHAT IT ASSERTS
    Operational properties that only a live run can establish:
      * every round completes and the server advances;
      * secure aggregation genuinely ENGAGED -- masked submissions were accepted and not one
        submission reached the plaintext aggregation;
      * the identity gate was satisfied rather than bypassed (no FAILED_PRECONDITION);
      * every client exits cleanly.

    ...and the correctness claim itself: a secure run reaches the same model as a plaintext run
    of the same seed, to within quantisation. Two identical plaintext live runs are bit-identical,
    so the harness is deterministic enough for this to mean something.

    That comparison earned its place. On its first full run it reported 1.6e-3 against an
    expected ~1e-8, and the temptation was to explain it away as the two runs scheduling client
    catch-up differently -- the logs did show different rebuild patterns. It was not scheduling.
    A holder's summed share for round r was arriving after the server had moved to r+1 and
    completing r+1 with r's recovered scalars against r+1's seeds. The check was right and the
    story was wrong; the fix took the difference to 6e-8.

DROPOUT
    ``--dropout PHASE`` makes the last ``--dropout-count`` clients leave in round
    ``--dropout-round`` by exiting their process, so the server sees a real disconnect rather than
    a simulated one. The plaintext control drops the same clients at the equivalent point, so both
    runs aggregate over the same set and the model comparison still means something. Phases:

      before_key    leaves before publishing its key, so it never joins the round's cohort
      after_key     published its key and the cohort closed; leaves before distributing shares
      after_shares  distributed its sealed shares; leaves before its masked submission
      after_submit  its masked value was accepted and the set frozen; leaves before returning
                    a summed share

    A plaintext round has no keys or shares, so its first three phases are the same event (the
    client never submits that round) and after_submit is submit-then-leave.

    A dropout can only be survived when the cohort is larger than the minimum a round needs, so
    ``--min-clients`` sets the strategy's min_fit_clients below ``--clients``. With the two equal
    -- the shape fl_server.py produces -- any dropout leaves the round short. ``--round-timeout``
    sets the server's per-round deadline (FEDLEARN_ROUND_TIMEOUT_S), which is what resolves a round
    a client left, so a dropout run need not sit out the 120 s default.

USAGE
    PYTHONPATH=src python3 run_secure_agg_live_test.py [--rounds N] [--clients N]
        [--min-clients M] [--dropout PHASE] [--dropout-round R] [--dropout-count D]
        [--round-timeout S]

    Exit code 0 on success. Logs land in ./secure_agg_live_logs/.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import multiprocessing
import os
import socket
import sys
import time
from collections import OrderedDict

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "secure_agg_live_logs")

SEED = 42
INPUT_DIM, HIDDEN, CLASSES = 8, 6, 2
SAMPLES_PER_CLIENT = 64

DROPOUT_PHASES = ("before_key", "after_key", "after_shares", "after_submit")


class TinyNet(nn.Module):
    """Deliberately small: this test is about the transport and the protocol, not about learning.

    A small flat dimension also keeps DeComFL's model_dim handshake cheap and makes a divergence
    between the two runs easy to read off.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, CLASSES)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _build_model() -> TinyNet:
    torch.manual_seed(SEED)
    return TinyNet()


def _client_data(partition: int, num_clients: int):
    """A deterministic per-partition shard. Same data in both runs, so any model difference is
    attributable to the aggregation path rather than to the sampling."""
    g = torch.Generator().manual_seed(SEED * 1000 + partition)
    x = torch.randn(SAMPLES_PER_CLIENT, INPUT_DIM, generator=g)
    y = (x.sum(dim=1) > 0).long()
    return x, y


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _mint_token(secret_b64: str, partition: int, client_id: str, ttl_s: int = 3600) -> str:
    """A connection token the SE-14 interceptor and the SE-15 extractor will both accept."""
    import jwt
    from fedlearn.security.token_verify import DEFAULT_AUDIENCE

    return jwt.encode(
        {
            "partitionId": partition,
            "sub": client_id,
            "aud": DEFAULT_AUDIENCE,
            "exp": int(time.time()) + ttl_s,
        },
        base64.b64decode(secret_b64),
        algorithm="HS256",
    )


# ---------------------------------------------------------------------------------------------
# Server child process
# ---------------------------------------------------------------------------------------------
def _run_server(port, rounds, num_clients, min_clients, secure, threshold, secret_b64, result_q,
                tag, round_timeout_s=None):
    log_path = os.path.join(LOG_DIR, f"server_{tag}.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = sys.stderr = log_file
    logging.basicConfig(level=logging.INFO, format="[SERVER] %(asctime)s %(message)s",
                        stream=log_file, force=True)

    # Auth on for BOTH runs, not just the secure one: the secure RPCs fail closed without a
    # verified partition, so leaving it off for the control would make the two runs differ in a
    # second way and confound the comparison.
    os.environ["FEDLEARN_REQUIRE_CLIENT_AUTH"] = "1"
    os.environ["FEDLEARN_FL_TOKEN_SECRET"] = secret_b64
    if round_timeout_s is not None:
        # Read when the coordinator is constructed, inside start_server below.
        os.environ["FEDLEARN_ROUND_TIMEOUT_S"] = str(round_timeout_s)

    from fedlearn.server.decomfl_strategy import DeComFL
    from fedlearn.server.server import ServerConfig, start_server

    try:
        model = _build_model()

        # Snapshot the averaged scalars the server actually aggregated each round. Comparing
        # final models alone cannot say WHERE two runs diverged, and the per-round record is what
        # distinguishes "the aggregation differs" from "the runs were scheduled differently".
        per_round = {}

        def evaluate_fn(server_round, parameters):
            # MUST return (loss, metrics). Returning None makes DeComFL.evaluate raise while
            # unpacking, the servicer swallows it, and the round hangs until its deadline --
            # which is how this harness first found that failure.
            hist = getattr(strategy, "gradient_history", {})
            if server_round in hist:
                per_round[server_round] = [list(row) for row in hist[server_round]]
            return 0.0, {}

        strategy = DeComFL(
            initial_parameters=model.state_dict(),
            evaluate_fn=evaluate_fn,
            min_fit_clients=min_clients,
            clients_per_round=num_clients,
            num_local_steps=1,
            num_perturbations=4,
            learning_rate=0.001,
            smoothing_param=0.001,
            seed=SEED,
        )
        config = ServerConfig(
            num_rounds=rounds,
            secure_aggregation=secure,
            secure_agg_threshold=threshold,
        )
        _, final_params = start_server(f"127.0.0.1:{port}", config, strategy)
        result_q.put({
            "ok": True,
            "final_flat": strategy.global_params_flat.tolist(),
            "per_round_scalars": per_round,
            "history_rounds": sorted(strategy.gradient_history),
        })
    except Exception as exc:  # noqa: BLE001 - the harness reports, it does not handle
        logging.exception("server failed")
        result_q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------------------------
# Client child process
# ---------------------------------------------------------------------------------------------
def _install_dropout(phase, drop_round, secure, result_q, client_id):
    """Make this client process leave at ``phase`` of round ``drop_round``.

    Leaving means exiting the process, so the server sees a real disconnect. The result is queued
    and the queue flushed first, because os._exit skips the feeder thread that would deliver it.
    """
    from fedlearn.client.grpc_client import GrpcClient
    from fedlearn.client.secure_agg_client import SecureAggregationClient

    def leave():
        logging.warning("DROPOUT: leaving at %s of round %d", phase, drop_round)
        result_q.put({"client_id": client_id, "ok": True, "dropped": True,
                      "outcome": f"dropped {phase} in round {drop_round}"})
        result_q.close()
        result_q.join_thread()
        os._exit(0)

    def round_arg(position):
        def extract(args, kwargs):
            return kwargs.get("round_num", args[position] if len(args) > position else None)
        return extract

    def wrap(cls, name, round_of, leave_after):
        original = getattr(cls, name)

        def wrapper(self, *args, **kwargs):
            dropping = round_of(args, kwargs) == drop_round
            if dropping and not leave_after:
                leave()
            result = original(self, *args, **kwargs)
            if dropping:
                leave()
            return result

        setattr(cls, name, wrapper)

    if secure:
        cls, name, position = {
            "before_key": (SecureAggregationClient, "begin_round", 0),
            "after_key": (SecureAggregationClient, "distribute_shares", 0),
            "after_shares": (GrpcClient, "submit_masked_gradient_scalars", 2),
            "after_submit": (SecureAggregationClient, "collect_shares", 0),
        }[phase]
        wrap(cls, name, round_arg(position), leave_after=False)
    else:
        wrap(GrpcClient, "submit_gradient_scalars", round_arg(2),
             leave_after=(phase == "after_submit"))


def _run_client(port, partition, client_id, token, result_q, tag, secure=True, dropout=None):
    log_path = os.path.join(LOG_DIR, f"client_{tag}_{client_id}.log")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = sys.stderr = log_file
    logging.basicConfig(level=logging.INFO, format=f"[{client_id}] %(asctime)s %(message)s",
                        stream=log_file, force=True)

    # The token travels as x-connection-token on every call, added by the client interceptor.
    os.environ["FEDLEARN_CONNECTION_TOKEN"] = token

    from fedlearn.client.decomfl_client import DeComFLClient
    from fedlearn.client.decomfl_start import start_decomfl_client
    from torch.utils.data import DataLoader, TensorDataset

    if dropout is not None:
        phase, drop_round = dropout
        _install_dropout(phase, drop_round, secure, result_q, client_id)

    try:
        time.sleep(2.0)  # let the server bind
        x, y = _client_data(partition, 0)
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        client = DeComFLClient(
            model=_build_model(), train_loader=loader, smoothing_param=0.001, device="cpu",
        )
        outcome = start_decomfl_client(f"127.0.0.1:{port}", client, client_id)
        result_q.put({"client_id": client_id, "ok": True, "outcome": outcome})
    except Exception as exc:  # noqa: BLE001
        logging.exception("client failed")
        result_q.put({"client_id": client_id, "ok": False,
                      "error": f"{type(exc).__name__}: {exc}"})


# ---------------------------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------------------------
def run_once(rounds, num_clients, secure, threshold, tag, timeout_s=180, min_clients=None,
             dropout=None, round_timeout_s=None):
    """Start a server and its clients, wait, and always reap the children.

    ``dropout`` is None or ``{"phase": ..., "round": ..., "count": ...}``; the last ``count``
    partitions leave. ``min_clients`` defaults to ``num_clients``.

    The reaping is not incidental. A harness that leaves a child alive on failure contaminates
    every later run on the machine -- including any timing measurement -- and the orphan is
    invisible unless you go looking for it.
    """
    min_clients = num_clients if min_clients is None else min_clients
    first_dropped = num_clients - dropout["count"] + 1 if dropout else num_clients + 1
    port = _free_port()
    secret_b64 = base64.b64encode(os.urandom(32)).decode()
    ctx = multiprocessing.get_context("spawn")

    server_q, client_q = ctx.Queue(), ctx.Queue()
    server = ctx.Process(
        target=_run_server,
        args=(port, rounds, num_clients, min_clients, secure, threshold, secret_b64, server_q,
              tag, round_timeout_s),
    )
    clients = [
        ctx.Process(
            target=_run_client,
            args=(port, p, f"c{p}", _mint_token(secret_b64, p, f"c{p}"), client_q, tag, secure,
                  (dropout["phase"], dropout["round"]) if p >= first_dropped else None),
        )
        for p in range(1, num_clients + 1)
    ]

    server.start()
    for c in clients:
        c.start()

    try:
        deadline = time.time() + timeout_s
        server.join(timeout=max(1, deadline - time.time()))
        for c in clients:
            c.join(timeout=max(1, deadline - time.time()))

        timed_out = server.is_alive() or any(c.is_alive() for c in clients)
        server_result = server_q.get_nowait() if not server_q.empty() else None
        client_results = []
        while not client_q.empty():
            client_results.append(client_q.get_nowait())
        return {
            "timed_out": timed_out,
            "server": server_result,
            "clients": client_results,
            "port": port,
            "min_clients": min_clients,
            "dropout": dropout,
        }
    finally:
        for proc in [server, *clients]:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--min-clients", type=int, default=None,
                        help="the strategy's min_fit_clients; defaults to --clients")
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--dropout", choices=DROPOUT_PHASES, default=None)
    parser.add_argument("--dropout-round", type=int, default=2)
    parser.add_argument("--dropout-count", type=int, default=1)
    parser.add_argument("--round-timeout", type=float, default=None,
                        help="server per-round deadline in seconds (FEDLEARN_ROUND_TIMEOUT_S)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="per-run wall clock budget. A secure round has more round trips "
                             "than a plaintext one, so it needs more than the obvious amount.")
    parser.add_argument("--secure-only", action="store_true",
                        help="skip the plaintext control run")
    args = parser.parse_args()

    min_clients = args.clients if args.min_clients is None else args.min_clients
    if not 1 <= min_clients <= args.clients:
        parser.error("--min-clients must be between 1 and --clients")
    dropout = None
    if args.dropout:
        if not 1 <= args.dropout_count < args.clients:
            parser.error("--dropout-count must leave at least one client")
        if not 1 <= args.dropout_round <= args.rounds:
            parser.error("--dropout-round must be one of the run's rounds")
        dropout = {"phase": args.dropout, "round": args.dropout_round, "count": args.dropout_count}

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Logs -> {LOG_DIR}")

    shape = (f"{args.clients} clients (min {min_clients}), {args.rounds} rounds, "
             f"threshold {args.threshold}, auth ON")
    if dropout:
        shape += (f", {dropout['count']} leaving {dropout['phase']} in round {dropout['round']}")
    print(f"\n=== SECURE run: {shape} ===")
    t0 = time.time()
    secure = run_once(args.rounds, args.clients, True, args.threshold, "secure", args.timeout,
                      min_clients, dropout, args.round_timeout)
    secure_s = time.time() - t0
    _report("secure", secure, secure_s)

    # Both checks always run, so a failing run still prints its audit.
    secure_ok = _run_succeeded(secure, args.clients)
    ok = _audit_server_log("secure", expect_secure=True, rounds=args.rounds) and secure_ok

    if not args.secure_only:
        print(f"\n=== PLAINTEXT control run (same seed, auth ON, same dropout) ===")
        t0 = time.time()
        plain = run_once(args.rounds, args.clients, False, args.threshold, "plain", args.timeout,
                         min_clients, dropout, args.round_timeout)
        plain_s = time.time() - t0
        _report("plaintext", plain, plain_s)
        # Evaluated even when the secure run failed: whether the plaintext control survived the
        # same dropout is what says a failure belongs to the secure path.
        plain_ok = _run_succeeded(plain, args.clients)
        plain_ok = _audit_server_log("plain", expect_secure=False, rounds=args.rounds) and plain_ok
        ok = ok and plain_ok

        if ok:
            ok = _compare(secure, plain, args.clients)
        print(f"\nwall clock: secure {secure_s:.1f}s vs plaintext {plain_s:.1f}s")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def scan_server_log(tag):
    """Count the server-log lines that say what the server actually did, or None without a log."""
    path = os.path.join(LOG_DIR, f"server_{tag}.log")
    if not os.path.exists(path):
        return None
    text = open(path, encoding="utf-8", errors="replace").read()
    # "masked" counts accepted calls, not clients: a client re-submitting before the freeze to ask
    # whether it has happened is accepted again, so a round a client left counts several times.
    return {
        "rounds_started": text.count("[Server] Starting round"),
        "masked": text.count("Masked gradients accepted"),
        "plaintext": text.count("Receiving PLAINTEXT gradient scalars"),
        "secure_rounds": text.count("Completing secure DeComFL round"),
        "identity_refusals": text.count("Secure aggregation requires verified client identity"),
        "downgrade_refusals": text.count("Refusing plaintext submission"),
        "freezes": text.count("freezing the surviving set"),
        "force_aggregations": text.count("force-aggregating"),
        "stopped_early": text.count("Stop requested, ending training"),
    }


def _audit_server_log(tag, expect_secure, rounds) -> bool:
    """Read the server log back and prove the path we think ran actually ran.

    A secure run that silently fell back to plaintext would pass every liveness check -- the
    rounds complete, the clients exit cleanly, the model moves. The only evidence that the
    privacy mechanism engaged is in what the server did with each submission, so it is checked
    here rather than assumed from the flag.
    """
    counts = scan_server_log(tag)
    if counts is None:
        print(f"  FAILED: no server log for {tag}")
        return False

    print("  audit: " + " ".join(f"{k}={v}" for k, v in counts.items()))

    if counts["identity_refusals"]:
        print("  FAILED: the SE-15 identity gate refused a secure RPC -- client auth is not "
              "reaching the server, so the run did not exercise the secure path")
        return False
    if counts["stopped_early"]:
        print("  FAILED: the server stopped the run before its last round")
        return False

    if expect_secure:
        if counts["masked"] == 0:
            print("  FAILED: no masked submission was accepted; secure aggregation did not engage")
            return False
        if counts["plaintext"]:
            print(f"  FAILED: {counts['plaintext']} plaintext submission(s) reached the aggregation "
                  f"on a secure server")
            return False
        if counts["secure_rounds"] < rounds:
            print(f"  FAILED: {counts['secure_rounds']} secure round(s) completed, expected {rounds}")
            return False
    else:
        if counts["masked"]:
            print(f"  FAILED: {counts['masked']} masked submission(s) on a plaintext server")
            return False
        if counts["plaintext"] == 0:
            print("  FAILED: no plaintext submission reached the aggregation")
            return False
    return True


def _run_succeeded(run, num_clients) -> bool:
    if run["timed_out"]:
        print("  FAILED: timed out")
        return False
    if not run["server"] or not run["server"].get("ok"):
        print(f"  FAILED: server {run['server']}")
        return False
    bad = [c for c in run["clients"] if not c.get("ok")]
    if bad:
        print(f"  FAILED: {len(bad)} client(s) errored: {bad}")
        return False
    if len(run["clients"]) != num_clients:
        print(f"  FAILED: {len(run['clients'])}/{num_clients} clients reported")
        return False
    return True


def _report(label, run, seconds):
    server = run["server"]
    print(f"  port {run['port']}  {seconds:.1f}s  timed_out={run['timed_out']}")
    if server and server.get("ok"):
        print(f"  server ok, history rounds {server['history_rounds']}")
    elif server:
        print(f"  server error: {server.get('error')}")
    for c in run["clients"]:
        print(f"    {c['client_id']}: {'ok ' + str(c.get('outcome')) if c.get('ok') else c.get('error')}")


# Observed 5.96e-08 on a 3-client, 3-round run; the stale-round bug this check caught showed
# 1.64e-03. The bound sits well above the former and far below the latter, so it tolerates
# quantisation and platform float noise while still catching a defect of that class.
MODEL_AGREEMENT_BOUND = 1e-5


def _compare(secure, plain, num_clients) -> bool:
    """Masking must change what the server SEES, not what it COMPUTES.

    Not bit-identical: the secure path quantises into a fixed-point field and back, so it is
    exact in the field and approximate after dequantisation.
    """
    a = torch.tensor(secure["server"]["final_flat"])
    b = torch.tensor(plain["server"]["final_flat"])
    if a.shape != b.shape:
        print(f"  FAILED: shape mismatch {a.shape} vs {b.shape}")
        return False
    diff = (a - b).abs().max().item()
    print(f"\n  max |secure - plaintext| = {diff:.3e}  (bound {MODEL_AGREEMENT_BOUND:.1e}, "
          f"model magnitude {a.abs().max().item():.3e})")
    if diff > MODEL_AGREEMENT_BOUND:
        print("  FAILED: the secure path did not reproduce the plaintext model. Check the "
              "per-round scalars in the run result -- a divergence that appears at one round "
              "and persists points at the aggregation, not at quantisation.")
        return False
    print("  secure and plaintext agree within quantisation")
    return True


if __name__ == "__main__":
    sys.exit(main())
