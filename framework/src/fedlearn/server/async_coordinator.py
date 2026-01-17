import logging
import threading
import pickle
# import pika
from collections import OrderedDict
from typing import Dict, Optional
import time

import os
# import pika

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

class FLCoordinator:
    """
    A thread-safe class that groups client results by round number.
    This correctly handles asynchronous and out-of-order client submissions.
    """

    def __init__(self, min_clients_per_round: int):
        self.min_clients = min_clients_per_round
        self._lock = threading.Lock()
        self._round_events: Dict[int, threading.Event] = {}
        self._round_updates: Dict[int, list] = {}

    def submit_client_update(self, client_id: str, params: OrderedDict, num_examples: int, trained_on_round: int):
        with self._lock:
            logging.info(f"[Coordinator] Received update from '{client_id}' for round {trained_on_round}.")
            if trained_on_round not in self._round_updates:
                self._round_updates[trained_on_round] = []

            if any(client_id == cid for cid, _, _ in self._round_updates[trained_on_round]):
                logging.warning(f"Duplicate update from {client_id} for round {trained_on_round} ignored.")
                return
            self._round_updates[trained_on_round].append((client_id, params, num_examples))
            self._round_updates[trained_on_round].append((params, num_examples))

            if len(self._round_updates[trained_on_round]) >= self.min_clients:
                logging.info(
                    f"[Coordinator] Round {trained_on_round} has received enough updates. Signaling completion.")
                if trained_on_round in self._round_events:
                    self._round_events[trained_on_round].set()

    def wait_for_round_to_complete(self, round_number: int):
        with self._lock:
            if round_number not in self._round_events:
                self._round_events[round_number] = threading.Event()
        logging.info(f"[Server Main] Now waiting for round {round_number} to complete...")
        self._round_events[round_number].wait()

    def get_and_clear_updates_for_round(self, round_number: int) -> list:
        with self._lock:
            updates = self._round_updates.pop(round_number, [])
            self._round_events.pop(round_number, None)
            return updates


class ResultConsumer(threading.Thread):
    """
    Consumes results from clients via RabbitMQ and passes them to the FLCoordinator.
    Includes automatic reconnection and retry handling.
    """

    def __init__(self, coordinator, project_id: str, mq_host: str, mq_port: int = 5672,
                 max_retries: int = 5, retry_delay: float = 3.0):
        super().__init__()
        self.coordinator = coordinator
        self.project_id = project_id
        self.mq_host = mq_host
        self.mq_port = mq_port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._stop_event = threading.Event()

    def run(self):
        attempt = 0
        results_queue_name = f"results_queue_{self.project_id}"
        print(f"[ResultConsumer] results_queue_name - {results_queue_name}")

        while not self._stop_event.is_set():
            try:
                logging.info(f"[ResultConsumer] Connecting to RabbitMQ at {self.mq_host}:{self.mq_port}...")
                connection = pika.BlockingConnection(
                    get_rabbitmq_parameters()
                )
                channel = connection.channel()
                channel.queue_declare(queue=results_queue_name, durable=True)
                logging.info(f"[ResultConsumer] Listening on queue '{results_queue_name}'")

                def callback(ch, method, properties, body):
                    try:
                        result = pickle.loads(body)
                        self.coordinator.submit_client_update(
                            client_id=result['client_id'],
                            params=result['params'],
                            num_examples=result['num_examples'],
                            trained_on_round=result['round']
                        )
                        ch.basic_ack(delivery_tag=method.delivery_tag)
                    except Exception as e:
                        logging.error(f"[ResultConsumer] Error handling message: {e}", exc_info=True)
                        ch.basic_nack(delivery_tag=method.delivery_tag)

                channel.basic_qos(prefetch_count=1)
                channel.basic_consume(queue=results_queue_name, on_message_callback=callback)
                channel.start_consuming()

            except pika.exceptions.AMQPConnectionError as e:
                attempt += 1
                logging.warning(f"[ResultConsumer] Connection lost (attempt {attempt}/{self.max_retries}): {e}")
                if attempt < self.max_retries:
                    logging.info(f"[ResultConsumer] Retrying connection in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logging.critical("[ResultConsumer] Maximum retry attempts reached. Exiting consumer thread.")
                    break

            except Exception as e:
                logging.error(f"[ResultConsumer] Unexpected error: {e}", exc_info=True)
                time.sleep(self.retry_delay)

            else:
                # Reset retry counter after successful reconnection
                attempt = 0

            finally:
                try:
                    if 'connection' in locals() and connection.is_open:
                        connection.close()
                except Exception:
                    pass

    def stop(self):
        """Gracefully stops the consumer thread."""
        self._stop_event.set()
