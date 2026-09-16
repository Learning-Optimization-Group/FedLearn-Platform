"""In-process federated simulation over the production coordinator (P0-1).

``SimulatedFederation`` runs a federation of arbitrarily many clients inside one process. It
drives the *production* :class:`~fedlearn.server.coordinator.FLCoordinator` and the *production*
:class:`~fedlearn.server.strategy.Strategy` implementations by direct method call — no gRPC
channel, no TCP port, no subprocess. A simulated result therefore exercises the same
aggregation, the same poisoning defenses and the same round bookkeeping that a deployed run
does; only the transport is elided, and even that can be put back (see ``wire_in_the_loop``).

Why this shape
--------------
The deployed path reserves one TCP port per FL server from ``50000-50010``, capping the
platform at 11 concurrent federations and making a 1000-client experiment inexpressible. That
is a deployment constraint leaking into the science: every FL result worth comparing against is
quoted at client counts far above it.

The coordinator turned out to already be transport-free — ``register_client``,
``get_global_model_for_client``, ``submit_client_update`` and ``start_round`` are ordinary
methods, and gRPC lives entirely in the servicer that wraps them. So this module is a driver
loop, not a reimplementation. Nothing here re-derives FedAvg.

Three properties this class is responsible for
----------------------------------------------
1. **Determinism.** Client selection, dropout, wire routing and every client's local training
   derive from ``(seed, client_id, round)`` alone (see :mod:`fedlearn.simulation.rng`), so a
   run is reproducible from its ``meta`` block and a client's trajectory does not depend on
   how many peers exist. Torch's global RNG is scoped and restored around every client, so
   running client A cannot perturb client B.

2. **No wall-clock dependence.** A full round resolves inline: the submit that completes the
   cohort fires aggregation directly. A round with modelled dropout is resolved immediately
   via ``FLCoordinator.resolve_round_incomplete`` rather than by sleeping out the 120-second
   deadline the deployed server uses (P0-1c). Simulated time is never real time.

3. **The wire stays testable.** ``wire_in_the_loop`` routes a fraction of client updates
   through the real deterministic safetensors encode/decode. Running experiments with it off
   is only defensible because the test suite asserts that off and on agree bit-for-bit; if
   that ever fails, the codec has a precision or ordering bug and wire-free results are suspect.

Memory
------
Clients are constructed per round and released, rather than held for the lifetime of the run,
so peak memory scales with ``clients_per_round`` and not with ``num_clients``. That is what
makes 1000 clients viable on a laptop. The corollary is that ``client_factory`` is called once
per participation, so it should close over already-partitioned indices (cheap) rather than
re-reading a dataset from disk. Clients are assumed stateless between rounds, which is true for
the FedAvg/FedProx/FedOpt family — each round begins by loading the global parameters.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.strategy import Strategy

from .rng import ClientRng, RunRng, torch_rng_scope

log = logging.getLogger(__name__)

__all__ = ["SimulatedFederation", "SimulationResult", "RoundRecord"]

ClientFactory = Callable[[int, ClientRng], object]


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

@dataclass
class RoundRecord:
    """One federated round, in the form it lands in ``research/results/``.

    ``forced`` distinguishes a round that completed normally from one force-aggregated with a
    partial cohort. Silently merging the two would let a dropout study read as a clean run.
    """

    round: int
    selected: List[int]
    reported: List[int]
    dropped: List[int]
    forced: bool
    num_examples: int
    loss: Optional[float]
    metrics: Dict[str, float]
    wire_clients: int
    wire_bytes: int
    wall_seconds: float

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class SimulationResult:
    """A complete run: per-round curves, final parameters, and enough provenance to redo it."""

    rounds: List[RoundRecord]
    final_params: "OrderedDict[str, torch.Tensor]"
    final_digest: str
    meta: Dict = field(default_factory=dict)
    stopped_early: bool = False
    stop_reason: Optional[str] = None

    def to_json(self) -> dict:
        """The record format: ``meta`` + ``per_round``, matching the existing result files."""
        return {
            "meta": {
                **self.meta,
                "final_digest": self.final_digest,
                "stopped_early": self.stopped_early,
                "stop_reason": self.stop_reason,
            },
            "per_round": [r.to_json() for r in self.rounds],
        }


# --------------------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------------------

class SimulatedFederation:
    """Drive a federation of ``num_clients`` in-process against a real coordinator.

    Args:
        strategy: a production :class:`~fedlearn.server.strategy.Strategy`.
        client_factory: ``(client_id, client_rng) -> Client``. Called once per participation;
            see the module docstring on memory. The returned object must implement
            ``fit(parameters, config) -> (state_dict, num_examples)``.
        num_clients: size of the client population.
        clients_per_round: cohort sampled each round. Must not exceed ``num_clients``.
        seed: run seed. The only source of randomness in the whole run.
        initial_parameters: the starting global model.
        client_config: base config handed to every ``fit`` call (e.g. ``learning_rate``,
            ``local_epochs``). Strategy-supplied keys (e.g. FedProx's ``proximal_mu``) are
            merged over it, since the strategy is authoritative for its own knobs.
        min_clients_for_aggregation: floor below which an incomplete round stops the run
            instead of aggregating.
        wire_in_the_loop: fraction of client updates routed through the real safetensors
            encode/decode. 0.0 is fastest; 1.0 exercises the codec on every update.
        dropout_rate: fraction of each sampled cohort that fails to report, modelled rather
            than waited for.
        round_timeout_s: passed to the coordinator. A correct simulation never reaches it;
            it is settable so a test can prove that.
        device: device string handed to the factory's discretion (not applied here).
    """

    def __init__(
        self,
        strategy: Strategy,
        client_factory: ClientFactory,
        num_clients: int,
        clients_per_round: int,
        seed: int,
        initial_parameters: Optional["OrderedDict[str, torch.Tensor]"] = None,
        client_config: Optional[Dict] = None,
        min_clients_for_aggregation: int = 1,
        wire_in_the_loop: float = 0.0,
        dropout_rate: float = 0.0,
        round_timeout_s: Optional[float] = None,
        device: str = "cpu",
    ):
        if num_clients < 1:
            raise ValueError(f"num_clients must be >= 1, got {num_clients}")
        if clients_per_round < 1:
            raise ValueError(f"clients_per_round must be >= 1, got {clients_per_round}")
        if clients_per_round > num_clients:
            raise ValueError(
                f"clients_per_round={clients_per_round} exceeds num_clients={num_clients}"
            )
        if not 0.0 <= wire_in_the_loop <= 1.0:
            raise ValueError(f"wire_in_the_loop must be in [0, 1], got {wire_in_the_loop}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

        self.strategy = strategy
        self.client_factory = client_factory
        self.num_clients = int(num_clients)
        self.clients_per_round = int(clients_per_round)
        self.seed = int(seed)
        self.client_config = dict(client_config or {})
        self.wire_in_the_loop = float(wire_in_the_loop)
        self.dropout_rate = float(dropout_rate)
        self.device = device

        self.run_rng = RunRng(seed)
        self.coordinator = FLCoordinator(
            strategy=strategy,
            min_clients_for_aggregation=min_clients_for_aggregation,
            clients_per_round=clients_per_round,
            round_timeout_s=round_timeout_s,
        )
        params = initial_parameters
        if params is None:
            params = strategy.initialize_parameters()
        self.coordinator.set_initial_parameters(params)

    # -- round-scoped server randomness -------------------------------------------------

    def _server_rng(self, round_num: int) -> np.random.Generator:
        """Server-side stream for one round.

        Round-scoped rather than a single running stream so that round 5's cohort is
        reproducible without replaying rounds 1-4 — the same reasoning as
        :meth:`ClientRng.for_round`, and what makes a single anomalous round re-examinable.
        """
        return self.run_rng.server_rng(round_num)

    def _select_cohort(self, rng: np.random.Generator) -> List[int]:
        return sorted(
            int(c)
            for c in rng.choice(self.num_clients, size=self.clients_per_round, replace=False)
        )

    def _plan_round(self, round_num: int) -> Tuple[List[int], List[int], set]:
        """Decide the cohort, who drops, and who goes over the wire — all from the seed.

        Drawn in a fixed order from one round-scoped stream so the plan is a pure function of
        ``(seed, round)``, independent of what any client subsequently does.
        """
        rng = self._server_rng(round_num)
        selected = self._select_cohort(rng)

        n_drop = int(round(self.dropout_rate * len(selected)))
        dropped = (
            sorted(int(c) for c in rng.choice(selected, size=n_drop, replace=False))
            if n_drop > 0
            else []
        )
        participants = [c for c in selected if c not in set(dropped)]

        n_wire = int(round(self.wire_in_the_loop * len(participants)))
        wire = (
            set(int(c) for c in rng.choice(participants, size=n_wire, replace=False))
            if n_wire > 0
            else set()
        )
        return selected, dropped, wire

    # -- the wire -----------------------------------------------------------------------

    @staticmethod
    def _wire_roundtrip(
        params: "OrderedDict[str, torch.Tensor]",
    ) -> Tuple["OrderedDict[str, torch.Tensor]", int]:
        """Encode/decode through the real deterministic safetensors codec.

        Only floating-point tensors traverse the codec, which is float32-only by design (that
        constraint is what lets the libtorch-free mobile client decode it). Integer buffers —
        e.g. BatchNorm's ``num_batches_tracked`` — are passed through rather than silently
        coerced to float32 and back, which would corrupt them. Returns the decoded parameters
        and the encoded byte count, so wire volume is accounted per round.
        """
        floats = [
            (k, v.detach().cpu().numpy())
            for k, v in params.items()
            if v.is_floating_point()
        ]
        if not floats:
            return params, 0

        blob = save_safetensors(floats)
        decoded, _meta = load_safetensors(blob)
        # load_safetensors hands back views into the blob; copy so the tensors own their memory
        # and torch does not warn about a non-writable array.
        by_name = {name: np.array(arr, copy=True) for name, arr in decoded}

        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for k, v in params.items():
            if k in by_name:
                out[k] = torch.from_numpy(by_name[k]).reshape(v.shape)
            else:
                out[k] = v
        return out, len(blob)

    @staticmethod
    def _digest(params: "OrderedDict[str, torch.Tensor]") -> str:
        """sha256 over the canonical safetensors encoding.

        Deliberately the same function the wire uses, so the digest is comparable against a
        C++/mobile encoding of the same state — not a Python-only hash that proves nothing
        cross-language.
        """
        floats = [
            (k, v.detach().cpu().numpy())
            for k, v in params.items()
            if v.is_floating_point()
        ]
        return hashlib.sha256(save_safetensors(floats)).hexdigest()

    # -- the loop -----------------------------------------------------------------------

    def run(self, num_rounds: int) -> SimulationResult:
        """Run ``num_rounds`` federated rounds and return the full record.

        Stops early — recording why — if the coordinator gives up on a round (too few clients
        reported to aggregate). That is the coordinator's own policy, honoured rather than
        overridden, so simulated dropout behaves as deployed dropout would.

        The entire run is wrapped in a single :func:`torch_rng_scope`, which makes it
        *hermetic* in both directions. Scoping only the client callbacks is not enough: the
        strategy's ``evaluate_fn`` is user code that runs on the server side, and a perfectly
        ordinary one that constructs a model to load parameters into draws from torch's global
        RNG. That leaks the run's randomness into the caller's stream — the exact coupling
        :mod:`fedlearn.simulation.rng` exists to prevent — and it also makes the run's own
        server-side draws depend on whatever global state the caller happened to leave behind.
        The run-level scope closes both.
        """
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")

        started = time.monotonic()
        records: List[RoundRecord] = []
        stopped_early = False
        stop_reason: Optional[str] = None

        with torch_rng_scope(self.run_rng.server_torch_seed()):
            for _ in range(num_rounds):
                if self.coordinator.stop_requested:
                    stopped_early = True
                    stop_reason = self.coordinator.last_round_message
                    break

                record = self._run_one_round()
                records.append(record)

                if self.coordinator.stop_requested:
                    stopped_early = True
                    stop_reason = self.coordinator.last_round_message
                    break

        final_params = self.coordinator._global_model_params or OrderedDict()
        wall = time.monotonic() - started

        meta = {
            "seed": self.seed,
            "num_clients": self.num_clients,
            "clients_per_round": self.clients_per_round,
            "num_rounds": len(records),
            "requested_rounds": num_rounds,
            "strategy": type(self.strategy).__name__,
            "client_config": self.client_config,
            "wire_in_the_loop": self.wire_in_the_loop,
            "dropout_rate": self.dropout_rate,
            "device": self.device,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "wall_seconds": wall,
            "total_wire_bytes": sum(r.wire_bytes for r in records),
        }

        return SimulationResult(
            rounds=records,
            final_params=final_params,
            final_digest=self._digest(final_params),
            meta=meta,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )

    def _run_one_round(self) -> RoundRecord:
        t0 = time.monotonic()
        coord = self.coordinator
        coord.start_round()
        round_num = coord.current_round

        selected, dropped, wire = self._plan_round(round_num)
        dropped_set = set(dropped)

        global_params, _rnd, strategy_cfg = coord.get_global_model_for_client()
        # The strategy is authoritative for its own knobs (FedProx's mu), so it wins the merge.
        config = {**self.client_config, **(strategy_cfg or {})}

        reported: List[int] = []
        total_examples = 0
        wire_bytes = 0
        wire_clients = 0

        for client_id in selected:
            if client_id in dropped_set:
                continue

            client_rng = self.run_rng.client(client_id)

            # Construction is round-independent so a client's initial weights do not shift
            # from round to round; training is round-scoped so each round's stochasticity is
            # distinct. Both are scoped, so neither leaks into the next client or the caller.
            with torch_rng_scope(client_rng.torch_seed()):
                client = self.client_factory(client_id, client_rng)

            with torch_rng_scope(client_rng.torch_seed(round_num)):
                params, num_examples = client.fit(global_params, config)

            if client_id in wire:
                params, nbytes = self._wire_roundtrip(params)
                wire_bytes += nbytes
                wire_clients += 1

            coord.submit_client_update(
                client_id=str(client_id),
                params=params,
                num_examples=num_examples,
                trained_on_round=round_num,
            )
            reported.append(client_id)
            total_examples += num_examples

            del client  # release before the next client is built — peak RSS is per-cohort

        # A full cohort already aggregated inline on the last submit (coordinator.py). Only an
        # incomplete cohort needs resolving, and it is resolved immediately rather than by
        # waiting out the wall-clock deadline (P0-1c).
        forced = False
        if len(reported) < self.clients_per_round:
            forced = True
            coord.resolve_round_incomplete(
                f"simulated dropout ({len(dropped)} of {len(selected)} clients)"
            )

        metrics = coord.get_latest_metrics() or {}
        loss = metrics.get("loss")

        return RoundRecord(
            round=round_num,
            selected=selected,
            reported=reported,
            dropped=dropped,
            forced=forced,
            num_examples=total_examples,
            loss=float(loss) if loss is not None else None,
            metrics={k: float(v) for k, v in metrics.items() if k != "loss"},
            wire_clients=wire_clients,
            wire_bytes=wire_bytes,
            wall_seconds=time.monotonic() - t0,
        )
