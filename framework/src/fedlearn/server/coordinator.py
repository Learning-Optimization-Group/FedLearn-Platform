import logging
import math
import os
import threading
from collections import OrderedDict
import torch
from typing import Optional, List, Tuple, Dict
import time
from threading import Lock
from .strategy import Strategy

log = logging.getLogger(__name__)

# Default per-round client-dropout deadline (seconds). A synchronous FedAvg
# round otherwise blocks forever if a selected client never reports.
DEFAULT_ROUND_TIMEOUT_S = 120.0


def _round_timeout_from_env(default: float) -> float:
    """Read FEDLEARN_ROUND_TIMEOUT_S with a safe int/float parse + fallback."""
    raw = os.environ.get("FEDLEARN_ROUND_TIMEOUT_S")
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        log.warning(
            "Invalid FEDLEARN_ROUND_TIMEOUT_S=%r; falling back to %.1fs",
            raw, default,
        )
        return default
    if value <= 0:
        log.warning(
            "Non-positive FEDLEARN_ROUND_TIMEOUT_S=%r; falling back to %.1fs",
            raw, default,
        )
        return default
    return value


class MalformedDeComFLSubmission(ValueError):
    """A DeComFL gradient-scalar submission whose shape is not the round's expected K x P grid.

    Raised at coordinator ingress so the gRPC servicer can surface it as INVALID_ARGUMENT — a
    malformed grid would otherwise reach aggregate_fit (which indexes grad_scalars[k][p]) and crash
    the aggregation thread after the client was already acknowledged.
    """


class FLCoordinator:
    """
    A class that owns the concept of rounds and signals the main loop when a round is complete.
    """

    def __init__(self, strategy: Strategy, min_clients_for_aggregation: int, clients_per_round: int,
                 round_timeout_s: Optional[float] = None,
                 grad_clip_threshold: Optional[float] = 1000.0):
        self.strategy = strategy
        self.min_clients = min_clients_for_aggregation
        self.clients_per_round = clients_per_round

        # SE-3 poisoning defense (layer 2): clamp each DeComFL gradient scalar into
        # [-grad_clip_threshold, +grad_clip_threshold] at ingress. The honest zeroth-order scalar
        # envelope is ~O(10) (O(100) at init), so the 1e3 default sits >=10x above honest support:
        # the clamp is the identity map on honest values, preserving DeComFL's bounded-gradient
        # convergence assumption with zero trajectory bias, while capping a 1e9-scale hijack to a
        # bounded, recoverable per-round step. Set to None to disable. NOT covered here (documented
        # scope): within-bound stealth bias and rail collusion need client identity (SE-1) +
        # reputation; robust aggregation (trimmed-mean/median) is a large-cohort feature deferred
        # behind a min-cohort gate, being a no-op at the 1-3 client cohorts this platform runs.
        self.grad_clip_threshold = grad_clip_threshold

        # Per-round dropout deadline. Precedence: explicit constructor arg >
        # FEDLEARN_ROUND_TIMEOUT_S env var > module default.
        if round_timeout_s is not None:
            self.round_timeout_s = float(round_timeout_s)
        else:
            self.round_timeout_s = _round_timeout_from_env(DEFAULT_ROUND_TIMEOUT_S)

        self._lock = threading.Lock()
        self._round_complete_event = threading.Event()

        self._global_model_params: Optional[OrderedDict[str, torch.Tensor]] = None
        self._client_updates_received: List[Tuple[OrderedDict[str, torch.Tensor], int]] = []
        self._registered_clients: set[str] = set()
        self.current_round = 1  # Start at round 1
        self.stop_requested = False
        self.latest_metrics: Optional[dict] = None
        # Client-reported training telemetry (loss/accuracy/compute), fed by ReportClientMetrics (v2).
        self.client_metrics_log: List[dict] = []
        self.client_heartbeats: Dict[str, dict] = {}
        self.heartbeat_lock = Lock()
        self.heartbeat_timeout = 300

        # Failure state surfaced when a round is force-aggregated or aborted on timeout.
        self.last_round_failed = False
        self.last_round_message: Optional[str] = None

        # Monotonic timestamp marking when the current round began. Used to
        # enforce round_timeout_s independently of wall-clock adjustments.
        self._round_started_at = time.monotonic()

    def start_round(self):
        """Called by the main loop to begin a new round."""
        with self._lock:
            self._client_updates_received.clear()  # Prevent stale state leakage across rounds
            self._round_started_at = time.monotonic()  # Reset the dropout deadline for this round
        self._round_complete_event.clear()

    def wait_for_round_to_complete(self):
        """Called by the main loop. Blocks until the current round finishes.

        If the configured per-round dropout timeout elapses before all
        ``clients_per_round`` report, the round is resolved instead of hanging
        forever: force-aggregated with whatever arrived if at least
        ``min_clients`` (>=1) reported, otherwise the server is signalled to stop.
        """
        while not self._round_complete_event.wait(timeout=1.0):
            if self.stop_requested:
                break
            if (time.monotonic() - self._round_started_at) >= self.round_timeout_s:
                self._handle_round_timeout()
                break

    def _handle_round_timeout(self):
        """Resolve a round that blew its dropout deadline.

        Mirrors the locking discipline of submit_client_update: re-check the
        received-count and invoke the aggregation trigger while holding
        self._lock so we don't race a client update that completes the round
        at the same instant.
        """
        with self._lock:
            # A client may have completed the round between the wait() timeout
            # and acquiring the lock; if so, the trigger already fired.
            if self._round_complete_event.is_set():
                return

            received = len(self._client_updates_received)
            total = self.clients_per_round
            # The strategy aggregates from min_clients; require at least 1.
            required = max(1, self.min_clients)

            if received >= required:
                log.warning(
                    "Round %d timed out after %.1fs; force-aggregating %d of %d clients "
                    "that reported (min required=%d)",
                    self.current_round, self.round_timeout_s, received, total, required,
                )
                self.last_round_failed = True
                self.last_round_message = (
                    f"Round {self.current_round} timed out after {self.round_timeout_s:.1f}s; "
                    f"force-aggregated {received}/{total} clients (min required={required})."
                )
                # FR-4: dispatch to the strategy-appropriate trigger. The submit paths are protocol-
                # specific (FedAvg->submit_client_update, DeComFL->submit_decomfl_update) so they call
                # their trigger directly, but this timeout path is strategy-agnostic and must not
                # hardcode the FedAvg trigger — that would skip DeComFL's gradient_history write.
                self._trigger_round_completion()
            else:
                log.error(
                    "Round %d timed out after %.1fs with only %d of %d clients reported "
                    "(min required=%d); stopping server",
                    self.current_round, self.round_timeout_s, received, total, required,
                )
                self.last_round_failed = True
                self.last_round_message = (
                    f"Round {self.current_round} timed out after {self.round_timeout_s:.1f}s "
                    f"with only {received}/{total} clients reported (min required={required}); "
                    f"server stopped."
                )
                self.stop_requested = True
                self._round_complete_event.set()  # Release the main loop

    def get_global_model_for_client(self) -> Tuple[Optional[OrderedDict[str, torch.Tensor]], int, dict]:
        with self._lock:
            if self.stop_requested:
                return None, -1, {}
            return self._global_model_params, self.current_round, {}

    # Maximum allowed num_examples to prevent model poisoning via inflated dataset sizes
    MAX_NUM_EXAMPLES = 100_000

    def submit_client_update(self, client_id: str, params: OrderedDict[str, torch.Tensor], num_examples: int,
                             trained_on_round: int):
        with self._lock:
            if trained_on_round < self.current_round:
                return  # Ignore stale updates

            if trained_on_round > self.current_round:
                # Client is ahead, something is wrong. Ignore.
                return

            # Sanitize num_examples to prevent model poisoning
            if num_examples <= 0:
                # Suspicious payload — keep at WARNING so it shows up without DEBUG noise.
                log.warning(
                    "Invalid num_examples (%s) from client %s; skipping update",
                    num_examples, client_id,
                )
                return
            num_examples = min(num_examples, self.MAX_NUM_EXAMPLES)

            log.debug("Received update from %s for round %d", client_id, self.current_round)
            self._client_updates_received.append((params, num_examples))

            if len(self._client_updates_received) == self.clients_per_round:
                log.info(
                    "All %d clients reported for round %d; aggregating",
                    self.clients_per_round, self.current_round,
                )
                self._trigger_aggregation_and_evaluation()

    def _trigger_round_completion(self):
        """Dispatch a completed/force-resolved round to the strategy-appropriate aggregation path.

        DeComFL needs its own trigger: it records gradient_history[round] (clients replay it via
        get_rebuild_history to rebuild locally) and guards a None evaluate. The default FedAvg trigger
        does neither, so routing a DeComFL round through it silently desyncs every client and can
        crash on a run with no evaluate_fn. Uses the same DeComFL detection as the DeComFL trigger.
        Called while self._lock is held.
        """
        if 'DeComFL' in str(type(self.strategy)) and hasattr(self.strategy, 'gradient_history'):
            self._trigger_decomfl_aggregation_and_evaluation()
        else:
            self._trigger_aggregation_and_evaluation()

    def _trigger_aggregation_and_evaluation(self):
        """Aggregate client updates and advance the round counter.

        THREADING CONTRACT: This method MUST only be called while self._lock
        is held (by submit_client_update). The round counter mutation and
        event signal are therefore atomic with respect to concurrent RPC
        threads calling submit_client_update or get_global_model_for_client.
        """
        log.debug(
            "Aggregating %d updates for round %d",
            len(self._client_updates_received), self.current_round,
        )

        results = list(self._client_updates_received)
        self._client_updates_received.clear()

        aggregated_parameters = self.strategy.aggregate_fit(self.current_round, results)

        if aggregated_parameters is not None:
            self._global_model_params = aggregated_parameters
            loss, metrics = self.strategy.evaluate(self.current_round, self._global_model_params)
            self.latest_metrics = {"loss": loss, **metrics}
        else:
            # Aggregation returning None is a hard failure for the round, but
            # the server can continue — log at WARNING so operators see it
            # without breaking out of the federated loop.
            log.warning("Aggregation for round %d failed", self.current_round)
            self.latest_metrics = None

        # Advance round and signal LAST — state is consistent before any
        # waiting thread wakes up, because we are still inside _lock.
        self.current_round += 1
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
            # Per-step heartbeats are very chatty; keep at DEBUG.
            log.debug(
                "heartbeat client=%s status=%s round=%d step=%d/%d (%.1f%%)",
                client_id, status, current_round, current_step, total_steps, progress,
            )

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

    def record_client_metrics(self, metrics: dict) -> None:
        """Store a client's per-round training telemetry (ReportClientMetrics, v2 §6.4)."""
        with self.heartbeat_lock:
            self.client_metrics_log.append(metrics)
        self.latest_metrics = metrics

    def get_server_status(self) -> dict:
        """Get current server status."""
        with self._lock:
            return {
                "current_round": self.current_round,
                "required_clients_for_round": self.min_clients,
                "received_updates_this_round": len(self._client_updates_received)
            }

    # Add to FLCoordinator class in coordinator.py

    def submit_decomfl_update(
            self,
            client_id: str,
            gradient_scalars: List[List[float]],
            num_examples: int,
            trained_on_round: int
    ):
        """
        Handle DeComFL gradient scalar submission.

        Args:
            client_id: Client identifier
            gradient_scalars: Nested list [local_step][perturbation] of gradient scalars
            num_examples: Number of training examples
            trained_on_round: Round number client trained on
        """
        with self._lock:
            if trained_on_round < self.current_round:
                # Stale submission from a slow client; expected during dropout/rejoin.
                log.debug(
                    "Ignoring stale DeComFL update from %s (trained=%d, current=%d)",
                    client_id, trained_on_round, self.current_round,
                )
                return

            if trained_on_round > self.current_round:
                # Client claims to be ahead of the server — protocol violation.
                log.warning(
                    "Client %s is ahead of server (trained=%d, current=%d); ignoring",
                    client_id, trained_on_round, self.current_round,
                )
                return

            log.debug(
                "Received DeComFL update from %s for round %d (%d/%d)",
                client_id, self.current_round,
                len(self._client_updates_received) + 1, self.clients_per_round,
            )

            # FR-5: validate the K x P grid shape against the strategy's configuration BEFORE the
            # scalars can reach aggregate_fit (which does grad_scalars[k][p] and would otherwise
            # crash the aggregation thread on a wrong shape, long after the client was acknowledged).
            # Raised — not silently dropped — so the servicer maps it to a client-visible
            # INVALID_ARGUMENT. Validated on a copy of the shape only; content checks follow below.
            expected_k = getattr(self.strategy, "K", None)
            expected_p = getattr(self.strategy, "P", None)
            if expected_k is not None and expected_p is not None:
                if len(gradient_scalars) != expected_k or any(len(row) != expected_p for row in gradient_scalars):
                    raise MalformedDeComFLSubmission(
                        f"expected {expected_k}x{expected_p} gradient scalars from {client_id}, "
                        f"got {len(gradient_scalars)} step(s) with widths {[len(r) for r in gradient_scalars]}"
                    )

            # FR-5: dedup. A client that already submitted (and was accepted) this round must not be
            # appended again — a second submission would be double-counted in the averaged update,
            # inflating that one client's weight. Keep the first accepted update, ignore the rest.
            if any(cid == client_id for cid, _, _ in self._client_updates_received):
                log.warning(
                    "Ignoring duplicate DeComFL update from %s in round %d",
                    client_id, self.current_round,
                )
                return

            # Reject non-finite gradient scalars before they reach aggregation (SE-3 poisoning
            # defense): a single NaN/Inf would corrupt the averaged update for every honest client
            # in the round, an unattributable denial-of-integrity attack over a plaintext channel.
            if not all(math.isfinite(g) for row in gradient_scalars for g in row):
                log.warning(
                    "Rejecting DeComFL update from %s: non-finite gradient scalars (poisoning defense)",
                    client_id,
                )
                return

            # Layer 2 (SE-3): clamp finite-but-large scalars to a bounded magnitude (see __init__).
            # A client sending g=1e9 would otherwise dominate g_sum and hijack the averaged step for
            # every honest client. We CLAMP (not reject) to preserve liveness, and do it here at
            # ingress — before storage — so both consumers of the stored scalars stay in lockstep:
            # aggregate_fit (steps the real global model) and _calculate_average_gradients (feeds
            # gradient_history, which clients replay to rebuild locally) read identical values.
            tau = self.grad_clip_threshold
            if tau is not None:
                if any(abs(g) > tau for row in gradient_scalars for g in row):
                    log.warning(
                        "Clamping out-of-range gradient scalars from %s to +/-%g (SE-3 poisoning defense)",
                        client_id, tau,
                    )
                gradient_scalars = [[max(-tau, min(tau, g)) for g in row] for row in gradient_scalars]

            # Store as tuple: (client_id, gradient_scalars, num_examples)
            self._client_updates_received.append((client_id, gradient_scalars, num_examples))

            if len(self._client_updates_received) >= self.clients_per_round:
                log.info(
                    "All %d DeComFL updates received for round %d; aggregating",
                    self.clients_per_round, self.current_round,
                )
                self._trigger_decomfl_aggregation_and_evaluation()

    def _trigger_decomfl_aggregation_and_evaluation(self):
        """Aggregate DeComFL gradient scalar submissions and advance the round.

        THREADING CONTRACT: This method MUST only be called while self._lock
        is held (by submit_decomfl_update). See _trigger_aggregation_and_evaluation
        for the full rationale.
        """
        log.debug(
            "Aggregating %d DeComFL updates for round %d",
            len(self._client_updates_received), self.current_round,
        )

        results = list(self._client_updates_received)
        self._client_updates_received.clear()

        # Aggregate gradient scalars
        aggregated_parameters = self.strategy.aggregate_fit(self.current_round, results)

        if aggregated_parameters is not None:
            self._global_model_params = aggregated_parameters

            # Calculate average gradients and store in strategy history
            avg_gradients = self._calculate_average_gradients(results)

            # Check if strategy is DeComFL and has gradient_history
            if 'DeComFL' in str(type(self.strategy)) and hasattr(self.strategy, 'gradient_history'):
                # Keyed by round (dict) so it aligns with seed_history + get_rebuild_history (audit #29)
                self.strategy.gradient_history[self.current_round] = avg_gradients
                log.debug("Stored gradient history for round %d", self.current_round)

            # Evaluate. evaluate() returns None when the server has no evaluate_fn (e.g. a bare
            # scalar-aggregation MVP); guard so a round completes instead of crashing on unpack.
            eval_result = self.strategy.evaluate(self.current_round, self._global_model_params)
            if eval_result is not None:
                loss, metrics = eval_result
                self.latest_metrics = {"loss": loss, **metrics}
                log.info(
                    "Round %d complete (loss=%.4f, metrics=%s)",
                    self.current_round, loss, metrics,
                )
            else:
                log.info("Round %d complete (no evaluate_fn configured; eval skipped)", self.current_round)
        else:
            log.warning("DeComFL aggregation for round %d failed", self.current_round)
            self.latest_metrics = None

        # Advance round and signal LAST — see _trigger_aggregation_and_evaluation.
        self.current_round += 1
        self._round_complete_event.set()

    def _calculate_average_gradients(
            self,
            results: List[Tuple[str, List[List[float]], int]]
    ) -> List[List[float]]:
        """
        Calculate average gradient scalars across clients.

        Returns:
            avg_gradients[k][p] = average gradient scalar for local step k, perturbation p
        """
        if not results:
            return []

        # Get dimensions from first result
        _, first_grads, _ = results[0]
        K = len(first_grads)
        P = len(first_grads[0]) if K > 0 else 0

        # Initialize averages
        avg_gradients = [[0.0 for _ in range(P)] for _ in range(K)]

        # Sum gradients from all clients
        num_clients = len(results)
        for client_id, grad_scalars, num_examples in results:
            for k in range(K):
                for p in range(P):
                    avg_gradients[k][p] += grad_scalars[k][p]

        # Average
        for k in range(K):
            for p in range(P):
                avg_gradients[k][p] /= num_clients

        return avg_gradients