from concurrent import futures
import grpc
import time
from dataclasses import dataclass

from .strategy import Strategy
from .coordinator import FLCoordinator
# import pika
# import pickle
# from .async_coordinator import FLCoordinator, ResultConsumer
from .strategy import FedAvgAggregator
from .grpc_servicer import FederatedLearningServiceServicer
from ..communication.generated import fedlearn_pb2_grpc
import logging
import sys
import os
# import pika

# def get_rabbitmq_parameters():
#     user = os.environ.get("RABBITMQ_USER", "admin")
#     password = os.environ.get("RABBITMQ_PASS", "admin")
#     host = os.environ.get("RABBITMQ_HOST", "localhost")
#     port = int(os.environ.get("RABBITMQ_PORT", 5672))
#     vhost = "/"  # explicitly use default vhost
#
#     print("=== RabbitMQ Connection Parameters ===")
#     print(f"USER: {user}")
#     print(f"PASS: {password}")
#     print(f"HOST: {host}")
#     print(f"PORT: {port}")
#     print(f"VHOST: {vhost}")
#     print("====================================")
#
#     credentials = pika.PlainCredentials(user, password)
#     parameters = pika.ConnectionParameters(
#         host=host,
#         port=port,
#         virtual_host=vhost,
#         credentials=credentials,
#         heartbeat=600,
#         blocked_connection_timeout=300
#     )
#     return parameters


@dataclass
class ServerConfig:
    num_rounds: int = 3


def start_server(
        server_address: str,
        config: ServerConfig,
        strategy: Strategy
) -> tuple[list, dict]:
    """
    Start a gRPC Federated Learning server with heartbeat support.

    Args:
        server_address: Address to bind server (e.g., "0.0.0.0:50051")
        config: Server configuration
        strategy: Aggregation strategy
        project_id: Project identifier

    Returns:
        Tuple of (history, final_parameters)
    """
    logging.info(f"Starting FedLearn server on {server_address}")

    # Create coordinator
    coordinator = FLCoordinator(
        strategy=strategy,
        min_clients_for_aggregation=strategy.min_fit_clients,
        clients_per_round=strategy.clients_per_round,
    )

    coordinator.set_initial_parameters(strategy.initial_parameters)
    # Create gRPC server with proper options
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            # Keepalive settings for long-running clients
            ('grpc.keepalive_time_ms', 120000),  # 120 seconds
            ('grpc.keepalive_timeout_ms', 60000),  # 30 seconds
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 120000),
            ('grpc.http2.min_ping_interval_without_data_ms', 120000),
            ('grpc.http2.bdp_probe', False),
            ('grpc.http2.max_ping_strikes', 0),

            # Connection limits
            ('grpc.max_connection_idle_ms', 7200000),  # 2 hours
            ('grpc.max_connection_age_ms', 14400000),  # 4 hours
            ('grpc.max_connection_age_grace_ms', 600000),  # 10 min

            # Message size limits
            ('grpc.max_send_message_length', 1024 * 1024 * 1024),  # 1 GB
            ('grpc.max_receive_message_length', 1024 * 1024 * 1024),  # 1 GB
        ]
    )

    # Add servicer
    fedlearn_pb2_grpc.add_FederatedLearningServiceServicer_to_server(
        FederatedLearningServiceServicer(coordinator),
        grpc_server
    )

    # Bind address
    grpc_server.add_insecure_port(server_address)

    # Start server
    grpc_server.start()
    logging.info(f"gRPC server started and listening on {server_address}")

    try:
        # Run federated learning training loop
        history = []

        for round_num in range(1,config.num_rounds+1):

            coordinator.start_round()
            logging.info(f"[Server] Starting round {round_num}/{config.num_rounds}")
            logging.info(f"[Server] Waiting for {coordinator.min_clients} clients to submit updates...")

            # Wait for round to complete (blocks until min_clients submit updates)
            coordinator.wait_for_round_to_complete()

            if coordinator.stop_requested:
                logging.info("[Server] Stop requested, ending training.")
                break

            # Get metrics from completed round
            metrics = coordinator.get_latest_metrics()
            if metrics:
                history.append((round_num, metrics))
                logging.info(f"[Server] Round {round_num} complete. Metrics: {metrics}")
            else:
                logging.warning(f"[Server] Round {round_num} completed but no metrics available.")

            # Advance to next round
            coordinator.current_round += 1


        # Get final parameters
        final_parameters = coordinator.get_global_model_params()

        logging.info("Federated learning complete. Stopping server...")

        return history, final_parameters

    except KeyboardInterrupt:
        logging.info("Server interrupted by user")
        return [], {}
    except Exception as e:
        logging.error(f"Error during federated learning: {e}", exc_info=True)
        coordinator.signal_stop()
        return [], {}
    finally:
        # Graceful shutdown
        grpc_server.stop(grace=5)
        logging.info("gRPC server stopped")


# def start_server_mq(config: ServerConfig, strategy: Strategy, project_id: str, mq_host: str = '127.0.0.1', mq_port: int = 5672, max_retries: int = 5, retry_delay: float = 3.0):
#     """
#     Starts an asynchronous federated learning server using a message queue with retry mechanism.
#     """
#     logging.info("--- Starting ASYNCHRONOUS FedLearn Server ---")
#
#     coordinator = FLCoordinator(min_clients_per_round=strategy.min_fit_clients)
#     result_consumer = ResultConsumer(coordinator, project_id, mq_host)
#     result_consumer.daemon = True
#     result_consumer.start()
#
#     # Retry loop for RabbitMQ connection
#     attempt = 0
#     connection = None
#     while attempt < max_retries:
#         try:
#             logging.info(f"Attempting to connect to RabbitMQ at {mq_host}:{mq_port} (Attempt {attempt+1}/{max_retries})")
#
#             connection_params = pika.ConnectionParameters(host=mq_host, port=mq_port, virtual_host='/',heartbeat=600, blocked_connection_timeout=300)
#             connection = pika.BlockingConnection(get_rabbitmq_parameters())
#             logging.info("Connected to RabbitMQ successfully.")
#             break
#         except pika.exceptions.AMQPConnectionError as e:
#             attempt += 1
#             logging.warning(f"Connection attempt {attempt} failed: {e}")
#             if attempt < max_retries:
#                 logging.info(f"Retrying in {retry_delay} seconds...")
#                 time.sleep(retry_delay)
#             else:
#                 logging.critical(f"Could not connect to RabbitMQ after {max_retries} attempts. Exiting.")
#                 raise
#
#     channel = connection.channel()
#     tasks_queue_name = f'tasks_queue_{project_id}'
#     channel.queue_declare(queue=tasks_queue_name, durable=True)
#     logging.info(f"start_server_mq tasks_queue_name-{tasks_queue_name}")
#
#     history = []
#     parameters = strategy.initialize_parameters()
#
#     for r in range(1, config.num_rounds + 1):
#         logging.info(f"\n======== Broadcasting Task for Round {r}/{config.num_rounds} ========")
#
#         task_message = {'params': parameters, 'round': r, 'config': {'local_epochs': 1}}
#         logging.info(f"Task message type: {type(task_message)}")
#         logging.info(f"Approx size: {sys.getsizeof(task_message)} bytes")
#         channel.basic_publish(
#             exchange='',
#             routing_key=tasks_queue_name,
#             body=pickle.dumps(task_message),
#             properties=pika.BasicProperties(delivery_mode=2)
#         )
#
#         coordinator.wait_for_round_to_complete(r)
#         results = coordinator.get_and_clear_updates_for_round(r)
#
#         aggregator = FedAvgAggregator()
#         aggregated_parameters = aggregator.aggregate(results)
#
#         if aggregated_parameters:
#             parameters = aggregated_parameters
#             loss, metrics = strategy.evaluate(r, parameters)
#             history.append((r, {"loss": loss, **metrics}))
#         else:
#             logging.warning(f"Aggregation for round {r} failed. Skipping model update.")
#
#     logging.info("--- Federated Learning session complete. ---")
#     connection.close()
#     time.sleep(5)
#
#     return history, parameters
