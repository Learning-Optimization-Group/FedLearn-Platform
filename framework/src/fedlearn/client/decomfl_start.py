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


def start_decomfl_client(server_address: str, client: DeComFLClient, client_id: str):
    """
    Start a DeComFL client that connects to the server.

    Args:
        server_address: gRPC server address (e.g., "localhost:50051")
        client: DeComFLClient instance
        client_id: Unique identifier for this client
    """
    comm_client = GrpcClient(client_id=client_id, server_address=server_address)
    last_completed_round = -1

    # Register with server
    if not comm_client.register():
        log.error("[%s] Could not register with server; exiting", client_id)
        return

    log.info("[%s] Registered with server; starting heartbeat", client_id)
    comm_client.start_heartbeat()

    if hasattr(client, 'set_grpc_client'):
        client.set_grpc_client(comm_client)

    try:
        while True:
            try:
                # 1. Get DeComFL configuration (seeds + rebuild history)
                log.debug("[%s] Fetching DeComFL config", client_id)
                comm_client.update_status("fetching_config", 0, 0)

                server_round, seeds, rebuild_history, config = comm_client.get_decomfl_config()

                if server_round == -1:
                    log.info("[%s] Server signalled training complete; shutting down", client_id)
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
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    log.warning("[%s] Server unavailable; shutting down", client_id)
                    break
                log.warning(
                    "[%s] gRPC error (code=%s): %s. Retrying in 10s",
                    client_id, e.code(), e.details(),
                )
                comm_client.update_status("error", 0, 0)
                time.sleep(10)
            except Exception:
                log.exception("[%s] Unexpected DeComFL client error; shutting down", client_id)
                comm_client.update_status("error", 0, 0)
                break

    finally:
        log.info("[%s] Shutting down", client_id)
        comm_client.stop_heartbeat()
        comm_client.close()
        log.info("[%s] Shutdown complete", client_id)