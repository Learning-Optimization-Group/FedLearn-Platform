from concurrent import futures
import grpc
import time
from dataclasses import dataclass

from .strategy import Strategy
from .coordinator import FLCoordinator
from .grpc_servicer import FederatedLearningServiceServicer
from ..communication.generated import fedlearn_pb2_grpc
from ..security.interceptor import interceptor_from_env
from ..security.tls import check_server_tls_policy
import logging
import sys
import os
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage()
        }
        if record.exc_info:
            log_obj["stackTrace"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def configure_logging() -> None:
    """Configure root logging as JSON-on-stdout for the FL-server process.

    Called explicitly by start_server (the entrypoint) — NOT at import time — so importing the
    framework as a library does not hijack the host application's root logger (FR-9). The FL server
    runs as its own process spawned by the backend, which parses this JSON stdout, so owning the
    root logger is appropriate for that entrypoint.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]


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
    configure_logging()   # FR-9: set up JSON root logging at the entrypoint, not at import time
    logging.info(f"Starting FedLearn server on {server_address}")

    # Create coordinator
    coordinator = FLCoordinator(
        strategy=strategy,
        min_clients_for_aggregation=strategy.min_fit_clients,
        clients_per_round=strategy.clients_per_round,
    )

    coordinator.set_initial_parameters(strategy.initial_parameters)
    # Create gRPC server with proper options
    max_expected_clients = int(os.environ.get('MAX_CLIENTS', 50))
    optimal_workers = (max_expected_clients * 2) + 10
    # SE-1: gate the FL boundary on a valid connection token when FEDLEARN_REQUIRE_CLIENT_AUTH=1.
    # Absent (local/dev) -> no interceptor -> fail-open. Enforce-on but no secret -> raises here.
    auth_interceptor = interceptor_from_env()
    logging.info("FL-boundary client-token auth %s",
                 "ENABLED" if auth_interceptor is not None else "disabled (dev fail-open)")
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=optimal_workers),
        interceptors=[auth_interceptor] if auth_interceptor is not None else [],
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

    # Bind address. Uses TLS when FEDLEARN_GRPC_USE_TLS=1. SE-2: fail closed rather than serve a
    # deployed profile (FEDLEARN_REQUIRE_TLS=1) in plaintext, and require the certs when TLS is on.
    use_tls = check_server_tls_policy()
    if use_tls:
        server_key_path = os.environ["FEDLEARN_GRPC_SERVER_KEY"]
        server_cert_path = os.environ["FEDLEARN_GRPC_SERVER_CERT"]
        root_cert_path = os.environ.get("FEDLEARN_GRPC_ROOT_CERT")
        require_client_auth = os.environ.get("FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH", "0") == "1"

        with open(server_key_path, "rb") as f:
            server_key = f.read()
        with open(server_cert_path, "rb") as f:
            server_cert = f.read()
        root_cert = None
        if root_cert_path:
            with open(root_cert_path, "rb") as f:
                root_cert = f.read()

        server_credentials = grpc.ssl_server_credentials(
            [(server_key, server_cert)],
            root_certificates=root_cert,
            require_client_auth=require_client_auth,
        )
        grpc_server.add_secure_port(server_address, server_credentials)
        logging.info("gRPC TLS enabled (require_client_auth=%s)", require_client_auth)
    else:
        grpc_server.add_insecure_port(server_address)
        logging.warning("gRPC server running without TLS. Set FEDLEARN_GRPC_USE_TLS=1 for production.")

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

            # Note: Round advancement is handled by coordinator._trigger_aggregation_and_evaluation()
            # Do NOT increment here — it was causing double-increment (rounds 1→3→5).


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

