# src/fedlearn/client/client.py
from abc import ABC, abstractmethod
from collections import OrderedDict
# import pika
import pickle
import grpc
import torch
import time
from typing import Tuple
from .grpc_client import GrpcClient
import logging
import os
# import pika
import traceback

log = logging.getLogger(__name__)

class Client(ABC):
    """Abstract base class for federated learning clients."""

    @abstractmethod
    def get_parameters(self) -> OrderedDict[str, torch.Tensor]:
        """Return the current local model parameters."""
        pass

    @abstractmethod
    def fit(self, parameters: OrderedDict[str, torch.Tensor], config: dict) -> Tuple[
        OrderedDict[str, torch.Tensor], int]:
        """Train the local model using the provided parameters."""
        pass


def start_client(server_address: str, client: Client, client_id: str):
    """
    Starts a client that connects to a server with heartbeat support.

    Args:
        server_address: gRPC server address (e.g., "localhost:50051")
        client: The Client instance that implements fit() and get_parameters()
        client_id: Unique identifier for this client
    """
    comm_client = GrpcClient(client_id=client_id, server_address=server_address)
    last_completed_round = -1  # Start at -1 to accept round 0 or 1 initially

    # Register with the server
    if not comm_client.register():
        log.error("[%s] Could not register with server; exiting", client_id)
        return

    log.info("[%s] Registered with server; starting heartbeat", client_id)
    comm_client.start_heartbeat()

    # Pass comm_client to the client for progress updates
    if hasattr(client, 'set_grpc_client'):
        client.set_grpc_client(comm_client)

    try:
        while True:
            try:
                # 1. Get model from server
                log.debug("[%s] Fetching global model", client_id)
                comm_client.update_status("fetching_model", 0, 0)

                parameters, server_round, config = comm_client.get_global_model()

                if server_round == -1:  # Server signalled completion
                    log.info("[%s] Server signalled training complete; shutting down", client_id)
                    break

                # Only proceed if the server has advanced to a new round
                if server_round > last_completed_round:
                    log.info("[%s] Starting local training for round %d", client_id, server_round)

                    # Update current round in grpc_client for heartbeat
                    comm_client.current_round = server_round
                    comm_client.update_status("training", 0, 1)  # Will be updated by training loop

                    # 2. Train the model (fit). client.fit() should call
                    # comm_client.update_status() during training to drive heartbeats.
                    new_parameters, num_examples = client.fit(parameters, config)

                    # 3. Submit the update
                    log.debug("[%s] Submitting update for round %d", client_id, server_round)
                    comm_client.update_status("submitting_update", 0, 0)

                    if comm_client.submit_update(new_parameters, num_examples, server_round):
                        log.info("[%s] Submitted update for round %d", client_id, server_round)
                        last_completed_round = server_round
                        comm_client.update_status("idle", 0, 0)
                    else:
                        log.error("[%s] Failed to submit update for round %d", client_id, server_round)
                        comm_client.update_status("error", 0, 0)
                else:
                    # The server is still in the same round, waiting for other clients.
                    # TODO: Replace polling with a WaitForNextRound server-streaming RPC
                    # that blocks until the Coordinator fires _round_complete_event,
                    # instantly pushing new round metadata to connected clients.
                    log.debug("[%s] Server still on round %d; waiting", client_id, server_round)
                    comm_client.update_status("waiting", 0, 0)
                    time.sleep(2)

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    log.warning("[%s] Server unavailable; shutting down", client_id)
                    break  # Server has shut down, exit the loop
                log.warning(
                    "[%s] gRPC error (code=%s): %s. Retrying in 10s",
                    client_id, e.code(), e.details(),
                )
                comm_client.update_status("error", 0, 0)
                time.sleep(10)
            except Exception:
                # Unknown failure path — log full traceback and exit so that
                # the orchestrator can restart the client cleanly. Returning
                # control to a misbehaving inner loop tends to mask root causes.
                log.exception("[%s] Unexpected client error; shutting down", client_id)
                comm_client.update_status("error", 0, 0)
                break

    finally:
        log.info("[%s] Shutting down", client_id)
        comm_client.stop_heartbeat()
        comm_client.close()
        log.info("[%s] Shutdown complete", client_id)

