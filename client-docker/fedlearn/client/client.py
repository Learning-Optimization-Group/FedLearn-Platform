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

def get_rabbitmq_parameters():
    user = os.environ.get("RABBITMQ_USER", "admin")
    password = os.environ.get("RABBITMQ_PASS", "admin")
    host = os.environ.get("RABBITMQ_HOST", "localhost")
    port = int(os.environ.get("RABBITMQ_PORT", 5672))
    vhost = "/"  # explicitly use default vhost

    print("=== RabbitMQ Connection Parameters ===")
    print(f"USER: {user}")
    print(f"PASS: {password}")
    print(f"HOST: {host}")
    print(f"PORT: {port}")
    print(f"VHOST: {vhost}")
    print("====================================")

    credentials = pika.PlainCredentials(user, password)
    parameters = pika.ConnectionParameters(
        host=host,
        port=port,
        virtual_host=vhost,
        credentials=credentials,
        heartbeat=600,
        blocked_connection_timeout=300
    )
    return parameters

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
        print(f"[{client_id}] Could not register with the server. Exiting.")
        return

    # Start heartbeat thread
    print(f"[{client_id}] Starting heartbeat...")
    comm_client.start_heartbeat()

    # Pass comm_client to the client for progress updates
    if hasattr(client, 'set_grpc_client'):
        client.set_grpc_client(comm_client)

    try:
        while True:
            try:
                # 1. Get model from server
                print(f"[{client_id}] Fetching global model...")
                comm_client.update_status("fetching_model", 0, 0)

                parameters, server_round, config = comm_client.get_global_model()

                if server_round == -1:  # Server signaling to stop
                    print(f"[{client_id}] Server has finished training. Shutting down.")
                    break

                # Only proceed if the server has advanced to a new round
                if server_round > last_completed_round:
                    print(f"[{client_id}] Starting local training for round {server_round}...")

                    # Update current round in grpc_client for heartbeat
                    comm_client.current_round = server_round
                    comm_client.update_status("training", 0, 1)  # Will be updated by training loop

                    # 2. Train the model (fit)
                    # The client.fit() method should call comm_client.update_status() during training
                    new_parameters, num_examples = client.fit(parameters, config)

                    # 3. Submit the update
                    print(f"[{client_id}] Submitting update for round {server_round}...")
                    comm_client.update_status("submitting_update", 0, 0)

                    if comm_client.submit_update(new_parameters, num_examples, server_round):
                        print(f"[{client_id}] Successfully submitted update for round {server_round}.")
                        last_completed_round = server_round  # Update our state
                        comm_client.update_status("idle", 0, 0)
                    else:
                        print(f"[{client_id}] Failed to submit update for round {server_round}.")
                        comm_client.update_status("error", 0, 0)
                else:
                    # The server is still in the same round, waiting for other clients.
                    # We should wait before polling again.
                    print(f"[{client_id}] Server is still in round {server_round}. Waiting...")
                    comm_client.update_status("waiting", 0, 0)
                    time.sleep(5)  # Wait for 5 seconds before checking again

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    print(f"[{client_id}] Server is unavailable. Shutting down.")
                    break  # Server has shut down, so we exit the loop
                else:
                    print(f"[{client_id}] An RPC error occurred: {e.details()}. Retrying in 10 seconds...")
                    print(f"[{client_id}] Error code: {e.code()}")
                    traceback.print_exc()
                    comm_client.update_status("error", 0, 0)
                    time.sleep(10)
            except Exception as e:
                print(f"[{client_id}] An unexpected error occurred: {e}. Shutting down.")
                print(f"[{client_id}] Full traceback:")
                traceback.print_exc()
                comm_client.update_status("error", 0, 0)
                break

    finally:
        # Clean shutdown - stop heartbeat and close connection
        print(f"[{client_id}] Shutting down...")
        comm_client.stop_heartbeat()
        comm_client.close()
        print(f"[{client_id}] Shutdown complete.")


def start_mq_client(mq_host: str, client: Client, client_id: str, project_id: str):
    """Connects to RabbitMQ and enters a loop to consume tasks and produce results."""

    connection = pika.BlockingConnection(get_rabbitmq_parameters())
    channel = connection.channel()
    tasks_queue_name = f'tasks_queue_{project_id}'
    results_queue_name = f'results_queue_{project_id}'


    channel.queue_declare(queue=tasks_queue_name, durable=True)
    channel.queue_declare(queue=results_queue_name, durable=True)
    print(f"[{client_id}] tasks_queue_name", tasks_queue_name)
    print(f"[{client_id}] results_queue_name", results_queue_name)


    print(f"[{client_id}] Waiting for tasks. To exit press CTRL+C")

    def callback(ch, method, properties, body):
        print(f"\n[{client_id}] Task received.")
        try:
            task_message = pickle.loads(body)
            params = task_message['params']
            server_round = task_message['round']
            config = task_message['config']

            print(f"[{client_id}] Starting local training for round {server_round}...")
            new_params, num_examples = client.fit(params, config)

            # Move parameters to CPU before sending
            new_params_cpu = OrderedDict({k: v.cpu() for k, v in new_params.items()})

            print(f"[{client_id}] Training complete. Publishing results for round {server_round}...")
            result_message = {
                'client_id': client_id,
                'params': new_params_cpu,
                'num_examples': num_examples,
                'round': server_round
            }
            channel.basic_publish(
                exchange='',
                routing_key=results_queue_name,
                body=pickle.dumps(result_message),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            print(f"[{client_id}] Results published. Waiting for next task.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"[{client_id}] ERROR processing task: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=tasks_queue_name, on_message_callback=callback)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print(f"\n[{client_id}] Shutting down.")
        connection.close()