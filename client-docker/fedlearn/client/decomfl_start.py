# src/fedlearn/client/decomfl_start.py
from __future__ import annotations

"""
Start function for DeComFL clients.
"""

import grpc
import time
import traceback
from .grpc_client import GrpcClient
from .decomfl_client import DeComFLClient


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
        print(f"[{client_id}] Could not register with server. Exiting.")
        return

    # Start heartbeat
    print(f"[{client_id}] Starting heartbeat...")
    comm_client.start_heartbeat()

    # Pass comm_client to client for progress updates
    if hasattr(client, 'set_grpc_client'):
        client.set_grpc_client(comm_client)

    try:
        while True:
            try:
                # 1. Get DeComFL configuration (seeds + rebuild history)
                print(f"[{client_id}] Fetching DeComFL config...")
                comm_client.update_status("fetching_config", 0, 0)

                server_round, seeds, rebuild_history, config = comm_client.get_decomfl_config()

                if server_round == -1:
                    print(f"[{client_id}] Server has finished training. Shutting down.")
                    break

                # Only proceed if server has advanced to new round
                if server_round > last_completed_round:
                    print(f"[{client_id}] Starting DeComFL training for round {server_round}...")

                    # Update current round for heartbeat
                    comm_client.current_round = server_round

                    # 2. Rebuild model if needed (for missed rounds)
                    if rebuild_history:
                        print(f"[{client_id}] Rebuilding model from {len(rebuild_history)} missed rounds...")
                        comm_client.update_status("rebuilding", 0, 0)
                        learning_rate = float(config.get('learning_rate', 0.001))
                        client.rebuild_model(rebuild_history, learning_rate)

                    # 3. Perform local ZO training
                    comm_client.update_status("training", 0, 1)

                    # Add seeds to config
                    training_config = dict(config)
                    training_config['seeds'] = seeds

                    # Train (returns gradient scalars, not parameters)
                    gradient_scalars, num_examples = client.fit(None, training_config)

                    # 4. Submit gradient scalars
                    print(f"[{client_id}] Submitting gradient scalars for round {server_round}...")
                    comm_client.update_status("submitting_update", 0, 0)

                    if comm_client.submit_gradient_scalars(gradient_scalars, num_examples, server_round):
                        print(f"[{client_id}] Successfully submitted gradient scalars for round {server_round}.")
                        last_completed_round = server_round
                        comm_client.update_status("idle", 0, 0)
                    else:
                        print(f"[{client_id}] Failed to submit gradient scalars for round {server_round}.")
                        comm_client.update_status("error", 0, 0)
                else:
                    # Server still in same round
                    print(f"[{client_id}] Server still in round {server_round}. Waiting...")
                    comm_client.update_status("waiting", 0, 0)
                    time.sleep(5)

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print(f"[{client_id}] Server unavailable. Shutting down.")
                    break
                else:
                    print(f"[{client_id}] RPC error: {e.details()}. Retrying in 10 seconds...")
                    traceback.print_exc()
                    comm_client.update_status("error", 0, 0)
                    time.sleep(10)
            except Exception as e:
                print(f"[{client_id}] Unexpected error: {e}. Shutting down.")
                traceback.print_exc()
                comm_client.update_status("error", 0, 0)
                break

    finally:
        print(f"[{client_id}] Shutting down...")
        comm_client.stop_heartbeat()
        comm_client.close()
        print(f"[{client_id}] Shutdown complete.")