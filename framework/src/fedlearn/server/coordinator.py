import threading
from collections import OrderedDict
import torch
from typing import Optional, List, Tuple, Dict
import time
from threading import Lock
from .strategy import Strategy


class FLCoordinator:
    """
    A class that owns the concept of rounds and signals the main loop when a round is complete.
    """

    def __init__(self, strategy: Strategy, min_clients_for_aggregation: int,clients_per_round:int):
        self.strategy = strategy
        self.min_clients = min_clients_for_aggregation
        self.clients_per_round = clients_per_round

        self._lock = threading.Lock()
        self._round_complete_event = threading.Event()

        self._global_model_params: Optional[OrderedDict[str, torch.Tensor]] = None
        self._client_updates_received: List[Tuple[OrderedDict[str, torch.Tensor], int]] = []
        self._registered_clients: set[str] = set()
        self.current_round = 1  # Start at round 1
        self.stop_requested = False
        self.latest_metrics: Optional[dict] = None
        self.client_heartbeats: Dict[str, dict] = {}
        self.heartbeat_lock = Lock()
        self.heartbeat_timeout = 300

    def start_round(self):
        """Called by the main loop to begin a new round."""
        self._round_complete_event.clear()

    def wait_for_round_to_complete(self):
        """Called by the main loop. Blocks until the current round finishes."""
        while not self._round_complete_event.wait(timeout=1.0):
            if self.stop_requested:
                break

    def get_global_model_for_client(self) -> Tuple[Optional[OrderedDict[str, torch.Tensor]], int, dict]:
        with self._lock:
            if self.stop_requested:
                return None, -1, {}
            return self._global_model_params, self.current_round, {}

    def submit_client_update(self, client_id: str, params: OrderedDict[str, torch.Tensor], num_examples: int,
                             trained_on_round: int):
        with self._lock:
            if trained_on_round < self.current_round:
                return  # Ignore stale updates

            if trained_on_round > self.current_round:
                # Client is ahead, something is wrong. Ignore.
                return

            print(f"[Coordinator] Received update from '{client_id}' for round {self.current_round}.")
            self._client_updates_received.append((params, num_examples))

            # if len(self._client_updates_received) >= self.min_clients:
            if len(self._client_updates_received) == self.clients_per_round:
                self._trigger_aggregation_and_evaluation()

    def _trigger_aggregation_and_evaluation(self):
        """This is the core logic for advancing a round."""
        print(
            f"[Coordinator] Aggregating {len(self._client_updates_received)} updates for round {self.current_round}...")

        results = list(self._client_updates_received)
        self._client_updates_received.clear()

        aggregated_parameters = self.strategy.aggregate_fit(self.current_round, results)

        if aggregated_parameters is not None:
            self._global_model_params = aggregated_parameters
            loss, metrics = self.strategy.evaluate(self.current_round, self._global_model_params)
            self.latest_metrics = {"loss": loss, **metrics}
        else:
            print(f"WARNING: Aggregation for round {self.current_round} failed.")
            self.latest_metrics = None

        # Advance to the next round and signal completion
        # self.current_round += 1
        self._round_complete_event.set()

    def set_initial_parameters(self, params: Optional[OrderedDict[str, torch.Tensor]]):
        self._global_model_params = params

    def get_latest_metrics(self) -> Optional[dict]:
        """Returns the metrics from the last completed round."""
        return self.latest_metrics

    def signal_stop(self):
        self.stop_requested = True
        self._round_complete_event.set()  # Release any waiting threads

    def register_client(self, client_id: str) -> bool:
        with self._lock:
            self._registered_clients.add(client_id)
            return True

    def get_global_model_params(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        """Safely returns the final global model parameters."""
        with self._lock:
            return self._global_model_params


    def update_client_heartbeat(self, client_id:str, status:str, current_step:int, total_steps:int, current_round:int)->tuple[bool,bool,str]:
        """
        Update the last  heartbeat time for a client
        """

        with self.heartbeat_lock:
            self.client_heartbeats[client_id] = {
                'status': status,
                'current_step': current_step,
                'total_steps': total_steps,
                'current_round': current_round,
                'last_seen': time.time()
            }

        if current_step % 10 == 0 or current_step == total_steps:
            progress = (current_step / total_steps * 100) if total_steps > 0 else 0
            print(f"[Heartbeat] {client_id}: {status} - Round {current_round}, "
                  f"Step {current_step}/{total_steps} ({progress:.1f}%)")

        should_stop = False

        return True, should_stop, f"Heartbeat received for {client_id}"

    def get_active_clients(self)->list[str]:
        """

        Get list of clients that have sent heartbeat recently
        :return:
        """

        current_time = time.time()
        active_clients = []

        with self.heartbeat_lock:
            for client_id, heartbeat_data in self.client_heartbeats.items():
                if current_time - heartbeat_data['last_seen'] < self.heartbeat_timeout:
                    active_clients.append(client_id)

        return active_clients

    def get_client_status(self, client_id: str) -> dict:
        """Get the current status of a specific client."""
        with self.heartbeat_lock:
            return self.client_heartbeats.get(client_id, {})

    def is_client_alive(self, client_id: str) -> bool:
        """Check if a client is still alive based on heartbeat."""
        with self.heartbeat_lock:
            if client_id not in self.client_heartbeats:
                return False

            last_seen = self.client_heartbeats[client_id]['last_seen']
            return (time.time() - last_seen) < self.heartbeat_timeout

    def get_server_status(self) -> dict:
        """Get current server status."""
        with self._lock:
            return {
                "current_round": self.current_round,
                "required_clients_for_round": self.min_clients,
                "received_updates_this_round": len(self._client_updates_received)
            }
