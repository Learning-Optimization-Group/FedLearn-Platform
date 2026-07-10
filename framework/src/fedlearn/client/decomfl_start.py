# src/fedlearn/client/decomfl_start.py
"""
Start function for DeComFL clients.
"""

import grpc
import logging
import time
import traceback
from .grpc_client import GrpcClient
from .decomfl_client import DeComFLClient

log = logging.getLogger(__name__)

# Terminal outcomes of a client session. Returned by start_decomfl_client so the
# caller (and tests) can distinguish a normal end-of-run from a real disconnect.
OUTCOME_COMPLETED = "completed"      # run finished normally -> exit 0
OUTCOME_DISCONNECTED = "disconnected"  # server unreachable/gone mid-run
OUTCOME_ERROR = "error"              # registration/unexpected failure

# Bounded rejoin budget for transient, non-terminal errors before giving up.
_MAX_CONSECUTIVE_FAILURES = 3
_RETRY_DELAY_SECONDS = 10


def start_decomfl_client(server_address: str, client: DeComFLClient, client_id: str) -> str:
    """
    Start a DeComFL client that connects to the server.

    Args:
        server_address: gRPC server address (e.g., "localhost:50051")
        client: DeComFLClient instance
        client_id: Unique identifier for this client

    Returns:
        One of OUTCOME_COMPLETED / OUTCOME_DISCONNECTED / OUTCOME_ERROR.
    """
    comm_client = GrpcClient(client_id=client_id, server_address=server_address)
    last_completed_round = -1
    consecutive_failures = 0
    outcome = OUTCOME_ERROR

    # Register with server
    if not comm_client.register():
        log.error("[%s] Could not register with server; exiting", client_id)
        return OUTCOME_ERROR

    log.info("[%s] Registered with server; starting heartbeat", client_id)
    comm_client.start_heartbeat()

    if hasattr(client, 'set_grpc_client'):
        client.set_grpc_client(comm_client)

    # FR-1: adopt the server's global model so every party shares the same initial model x_0 —
    # DeComFL's core invariant. Without this the client trains from its own random init and its
    # gradient scalars are directional derivatives of a DIFFERENT function than the server's
    # global model, so the aggregate is meaningless and the model cannot converge. This is the
    # one-shot O(d) initial download the paper assumes; per-round communication stays O(1).
    try:
        global_params, _, _ = comm_client.get_global_model()
    except grpc.RpcError as e:
        log.error(
            "[%s] Could not download the global model (%s); exiting rather than training from "
            "an unsynced init", client_id, e.details(),
        )
        comm_client.stop_heartbeat()
        comm_client.close()
        return

    if not global_params:
        log.error(
            "[%s] Server returned no global model (server not ready or stopping); exiting to "
            "avoid an unsynced x_0", client_id,
        )
        comm_client.stop_heartbeat()
        comm_client.close()
        return

    client.load_global_model(global_params)
    log.info("[%s] Synced local model to the server's global model before training", client_id)

    try:
        while True:
            # FR-10: server-driven stop (delivered via the heartbeat response). Exit before
            # starting another round once the server has asked this client to halt.
            if comm_client.should_stop_training():
                log.info("[%s] Server requested stop via heartbeat; shutting down", client_id)
                break
            try:
                # 1. Get DeComFL configuration (seeds + rebuild history)
                log.debug("[%s] Fetching DeComFL config", client_id)
                comm_client.update_status("fetching_config", 0, 0)

                server_round, seeds, rebuild_history, config = comm_client.get_decomfl_config()

                # A successful poll clears the transient-failure counter.
                consecutive_failures = 0

                if server_round == -1:
                    log.info("[%s] Server signalled run complete; shutting down cleanly", client_id)
                    outcome = OUTCOME_COMPLETED
                    break

                if server_round > last_completed_round:
                    log.info("[%s] Starting DeComFL training for round %d", client_id, server_round)
                    comm_client.current_round = server_round

                    # 2. Rebuild model if needed (for missed rounds)
                    if rebuild_history:
                        log.info(
                            "[%s] Rebuilding model from %d missed rounds",
                            client_id, len(rebuild_history),
                        )
                        comm_client.update_status("rebuilding", 0, 0)
                        learning_rate = float(config.get('learning_rate', 0.001))
                        client.rebuild_model(rebuild_history, learning_rate)

                    # 3. Perform local ZO training
                    comm_client.update_status("training", 0, 1)
                    training_config = dict(config)
                    training_config['seeds'] = seeds

                    gradient_scalars, num_examples = client.fit(None, training_config)

                    # FR-10: if the server stopped mid-fit, fit() broke out of the K-step loop
                    # early and the scalars are a partial (non-KxP) grid the server would
                    # reject anyway — do not submit; shut down.
                    if comm_client.should_stop_training():
                        log.info("[%s] Round %d aborted by server stop; discarding partial "
                                 "scalars and shutting down", client_id, server_round)
                        break

                    # 4. Submit gradient scalars
                    log.debug("[%s] Submitting gradient scalars for round %d",
                              client_id, server_round)
                    comm_client.update_status("submitting_update", 0, 0)

                    if comm_client.submit_gradient_scalars(gradient_scalars, num_examples, server_round):
                        log.info("[%s] Submitted gradient scalars for round %d",
                                 client_id, server_round)
                        last_completed_round = server_round
                        comm_client.update_status("idle", 0, 0)
                    else:
                        log.error("[%s] Failed to submit gradient scalars for round %d",
                                  client_id, server_round)
                        comm_client.update_status("error", 0, 0)
                else:
                    log.debug("[%s] Server still on round %d; waiting", client_id, server_round)
                    comm_client.update_status("waiting", 0, 0)
                    time.sleep(5)

            except grpc.RpcError as e:
                code = e.code() if hasattr(e, "code") else None

                # A completed run tears the server's RPCs down, which surfaces as
                # CANCELLED (in-flight call cancelled) or UNAVAILABLE (socket
                # closed). Before treating that as a failure, ask the server
                # whether the run is simply over — if so, this is a NORMAL
                # terminal condition, so exit 0 instead of retry-looping.
                if comm_client.server_reports_complete():
                    log.info("[%s] Server reports run complete; shutting down cleanly", client_id)
                    outcome = OUTCOME_COMPLETED
                    break

                # No completion signal. CANCELLED/UNAVAILABLE means the server
                # went away without finishing the run -> genuine disconnect.
                # (We do NOT blanket-swallow CANCELLED: a real disconnect is
                # reported as such, not as a clean success.)
                if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED):
                    log.warning(
                        "[%s] Server unreachable mid-run (code=%s); shutting down", client_id, code,
                    )
                    outcome = OUTCOME_DISCONNECTED
                    break

                # Other transient error: bounded rejoin attempts before giving up.
                consecutive_failures += 1
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    log.warning(
                        "[%s] gRPC error (code=%s) persisted for %d attempts; shutting down",
                        client_id, code, consecutive_failures,
                    )
                    outcome = OUTCOME_DISCONNECTED
                    break

                log.warning(
                    "[%s] gRPC error (code=%s): %s. Rejoining (%d/%d) in %ds",
                    client_id, code, e.details(),
                    consecutive_failures, _MAX_CONSECUTIVE_FAILURES, _RETRY_DELAY_SECONDS,
                )
                comm_client.update_status("error", 0, 0)
                time.sleep(_RETRY_DELAY_SECONDS)
            except Exception:
                log.exception("[%s] Unexpected DeComFL client error; shutting down", client_id)
                comm_client.update_status("error", 0, 0)
                outcome = OUTCOME_ERROR
                break

    finally:
        log.info("[%s] Shutting down (%s)", client_id, outcome)
        comm_client.stop_heartbeat()
        comm_client.close()
        log.info("[%s] Shutdown complete", client_id)

    return outcome