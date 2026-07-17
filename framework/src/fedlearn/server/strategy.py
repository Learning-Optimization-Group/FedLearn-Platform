import logging
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple
from collections import OrderedDict
import torch

from fedlearn.server._update_normalize import normalize_update, normalize_updates

log = logging.getLogger(__name__)

class Strategy(ABC):
    """Abstract base class for learning strategies."""
    @abstractmethod
    def initialize_parameters(self)->Optional[OrderedDict[str, torch.Tensor]]:
        """Initialize the global"""
        pass


    @abstractmethod
    def aggregate_fit(
            self,
            server_round:int,
            results:list[Tuple[OrderedDict[str, torch.Tensor],int]],
    )-> Optional[OrderedDict[str, torch.Tensor]]:
        """Aggregate training results from clients"""
        pass

    @abstractmethod
    def evaluate(
            self, server_round: int, parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        """Evaluate the global model."""
        pass


class FedAvg(Strategy):
    """The default strategy for FedAvg."""

    def __init__(
            self,
            initial_parameters: OrderedDict[str, torch.Tensor],
            evaluate_fn:Optional[Callable]=None,
            min_fit_clients:int=1,
            clients_per_round:int=None
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        self.aggregator = FedAvgAggregator()

    def initialize_parameters(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        return self.initial_parameters

    def aggregate_fit(
            self,
            server_round: int,
            results: list[Tuple[OrderedDict[str, torch.Tensor], int]],
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        if not results:
            return None

        # Aggregate using the same logic as before
        return self.aggregator.aggregate(results)

    def evaluate(
            self, server_round: int, parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        if self.evaluate_fn is None:
            return None

        # Call the user-provided evaluation function
        loss, metrics = self.evaluate_fn(server_round, parameters)
        log.info(
            "FedAvg eval round=%d loss=%.4f metrics=%s",
            server_round, loss, metrics,
        )
        return loss, metrics


class FedAvgAggregator:
    MAX_SAMPLES = 100_000  # Cap to prevent model poisoning via inflated num_examples

    def aggregate(self, updates):
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Coerce the accepted wire shapes (2-/3-tuples, JSON-encoded params) into a uniform list of
        # (client_id, state_dict, num_examples) via the shared normalizer.
        updates = normalize_updates(updates)

        # Sanitize num_examples: cap and reject invalid values
        sanitized_updates = [
            (cid, p, min(n, self.MAX_SAMPLES))
            for cid, p, n in updates if n > 0
        ]
        if not sanitized_updates:
            raise ValueError("No valid updates after sanitization.")

        # FR-18 / fedavg-4: template the aggregate on the UNION of client keys, and total examples
        # PER KEY. Templating on updates[0] alone silently drops any key only later clients carry;
        # weighting every key by num_examples/total_examples-over-ALL-clients but summing only the
        # clients that HAVE the key scaled a subset-held key by a weight share < 1, decaying it
        # toward zero each round (and letting an inflated-num_examples client bypass the per-client
        # L2 clip on the keys it omitted). Renormalizing each key over the clients that actually
        # provided it fixes both: a client missing a key now contributes nothing to it (correct),
        # and when every client holds every key this reduces exactly to the previous weighted mean.
        aggregated_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        key_totals: dict[str, int] = {}
        for _cid, params, num_examples in sanitized_updates:
            for key, tensor in params.items():
                if key not in aggregated_params:
                    aggregated_params[key] = torch.zeros_like(tensor.to(device), dtype=torch.float32)
                key_totals[key] = key_totals.get(key, 0) + num_examples

        for client_id, params, num_examples in sanitized_updates:
            for key in params:
                if key in aggregated_params:
                    weight = num_examples / key_totals[key]
                    torch.add(
                        aggregated_params[key],
                        params[key].to(device).float(),
                        alpha=weight,
                        out=aggregated_params[key],
                    )

            # Aggressively free client memory buffer
            params.clear()

        return aggregated_params


class FedLoRA(Strategy):
    """Federated LoRA. Clients communicate ONLY adapter params (FFA: B+head; FedIT: A+B+head).

    Reuses FedAvgAggregator (num-examples-weighted average over whatever keys are present).
    Under FFA_LORA, A is frozen+shared, so it is NOT aggregated — the strategy re-attaches the
    frozen A (captured from initial_parameters) to every aggregated global so it is redistributed
    unchanged and stays identical across clients (which makes avg(B)@A == avg(B@A) exact).
    """

    def __init__(self, initial_parameters, evaluate_fn=None, min_fit_clients=1,
                 clients_per_round=None, aggregation="FFA_LORA",
                 dp_enabled=False, dp_clip_norm=None, dp_noise_multiplier=None, dp_seed=None,
                 dp_target_epsilon=None, dp_delta=None, dp_num_clients=None, dp_rounds=None):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        self.aggregation = aggregation
        self.aggregator = FedAvgAggregator()
        self._frozen_a = (
            OrderedDict((k, v.clone()) for k, v in initial_parameters.items() if "lora_A" in k)
            if aggregation == "FFA_LORA" else OrderedDict()
        )
        if aggregation == "FFA_LORA" and not self._frozen_a:
            raise ValueError(
                "FFA_LORA requires lora_A keys in initial_parameters (the global adapter must carry "
                "the shared frozen A); none found — ensure the initial adapter is the FULL adapter (A+B+head)."
            )

        # ---- Differential privacy (FR-13). Default OFF: when disabled the aggregate_fit path below
        # is byte-for-byte the original num_examples-weighted average + frozen-A re-attach. When
        # enabled, aggregate_fit takes the central-DP path (clip each client's adapter delta to S,
        # UNIFORM-average, add Gaussian noise z*S/N on the aggregatable keys only) implemented in
        # fedlearn.privacy.dp_mechanism, then re-attaches the frozen A bit-identical (FFA invariant).
        self.dp_enabled = bool(dp_enabled)
        self.dp_clip_norm = None if dp_clip_norm is None else float(dp_clip_norm)
        self.dp_noise_multiplier = None if dp_noise_multiplier is None else float(dp_noise_multiplier)
        self.dp_seed = None if dp_seed is None else int(dp_seed)
        self.dp_target_epsilon = None if dp_target_epsilon is None else float(dp_target_epsilon)
        self.dp_delta = None if dp_delta is None else float(dp_delta)
        self.dp_num_clients = None if dp_num_clients is None else int(dp_num_clients)
        self.dp_rounds = None if dp_rounds is None else int(dp_rounds)
        # Accounted (ε, δ) trace for the chosen noise multiplier (eval-card / SE-11). None when DP is
        # off or the accounting params (δ + round count) weren't supplied.
        self.dp_accounted_epsilon = None
        self.dp_q = None
        if self.dp_enabled:
            if self.dp_clip_norm is None or self.dp_clip_norm <= 0:
                raise ValueError("FedLoRA dp_enabled requires dp_clip_norm (S) > 0.")
            # Subsampling rate for the accountant: cohort / enrolled population (q=1 if the population
            # is unknown — the conservative no-amplification assumption).
            self.dp_q = (self.clients_per_round / self.dp_num_clients) if self.dp_num_clients else 1.0

            # Calibrate the noise multiplier z: supplied directly, OR solved from a target-ε budget
            # via the RDP accountant. Exactly one of the two must be given.
            from fedlearn.privacy.dp_accountant import (
                compute_rdp, get_epsilon, required_noise_multiplier,
            )
            if self.dp_target_epsilon is not None:
                if self.dp_noise_multiplier is not None:
                    raise ValueError(
                        "FedLoRA DP: give either dp_noise_multiplier OR dp_target_epsilon, not both.")
                if self.dp_delta is None or self.dp_rounds is None:
                    raise ValueError(
                        "FedLoRA dp_target_epsilon requires dp_delta and dp_rounds to solve z.")
                self.dp_noise_multiplier = float(required_noise_multiplier(
                    self.dp_target_epsilon, self.dp_q, self.dp_rounds, self.dp_delta))
            elif self.dp_noise_multiplier is None:
                raise ValueError(
                    "FedLoRA dp_enabled requires exactly one of dp_noise_multiplier (z) or "
                    "dp_target_epsilon.")
            if self.dp_noise_multiplier < 0:
                raise ValueError("FedLoRA dp_noise_multiplier (z) must be >= 0.")

            # Best-effort accounted-ε trace: computable whenever δ and the round count are known.
            if self.dp_delta is not None and self.dp_rounds is not None:
                self.dp_accounted_epsilon = float(get_epsilon(
                    compute_rdp(self.dp_q, self.dp_noise_multiplier, self.dp_rounds),
                    self.dp_delta)[0])

        # Running global reference for the DP delta (mirrors RobustAggregator._global). Kept float32
        # to match the aggregator's output dtype so a delta subtraction never silently upcasts. This
        # is a NEW side-channel: it does not alter the non-DP return value (see aggregate_fit).
        self._global = OrderedDict(
            (k, v.detach().clone().to(torch.float32)) for k, v in initial_parameters.items()
        )
        # The DP noise generator is ALWAYS a DEDICATED torch.Generator (never the global default),
        # persisted across rounds so advancing it never reuses identical noise. With an explicit
        # dp_seed it is reproducible (tests/audits); in production (dp_seed=None) it is seeded from
        # FRESH OS entropy, INDEPENDENT of any global torch seed.
        #
        # This isolation is load-bearing for the (epsilon, delta) guarantee: fl_server.resolve_run_seed
        # calls torch.manual_seed(S) for data/model-init reproducibility and DISCLOSES S on the eval
        # card + logs. If the DP noise were drawn from the global default generator (the old
        # dp_seed=None path passed generator=None), it would become a deterministic function of that
        # disclosed seed — an adversary holding the card could replay the run and STRIP the noise,
        # recovering the un-noised client-level aggregate and voiding DP (DA-3 x FR-13 interaction).
        self._dp_generator = None
        if self.dp_enabled:
            self._dp_generator = torch.Generator()
            if self.dp_seed is not None:
                self._dp_generator.manual_seed(self.dp_seed)
            else:
                self._dp_generator.seed()  # non-deterministic OS-entropy seed, independent of global RNG

    def initialize_parameters(self):
        return self.initial_parameters

    def aggregate_fit(self, server_round, results):
        if not results:
            return None
        self._assert_client_keys_allowed(results)
        self._assert_homogeneous(results)
        if self.dp_enabled:
            aggregated = self._aggregate_fit_dp(results)
        else:
            # --- Non-DP path: byte-for-byte the original weighted-average + frozen-A re-attach. ---
            aggregated = self.aggregator.aggregate(results)
            if self.aggregation == "FFA_LORA":
                for k, v in self._frozen_a.items():
                    aggregated[k] = v.clone()
        # Update the running reference for next round's DP delta. Side-effect only: it CLONES
        # `aggregated`, so the object returned to the caller is unchanged (the non-DP return stays
        # byte-identical to the pre-DP behaviour).
        self._global = OrderedDict(
            (k, v.detach().clone().to(torch.float32)) for k, v in aggregated.items()
        )
        return aggregated

    def _aggregate_fit_dp(self, results):
        """Central-DP aggregation path (FR-13). Clip each client's adapter delta to S, uniform-
        average, add Gaussian noise z*S/N on the aggregatable keys ONLY (every client key that is
        NOT a frozen lora_A key), then re-attach the frozen A bit-identical (zero noise on A keeps
        the FFA invariant avg(B)@A == avg(B@A) exact)."""
        from fedlearn.privacy.dp_mechanism import dp_aggregate  # lazy import avoids an import cycle

        aggregatable_keys = [k for k in self._client_keys(results) if k not in self._frozen_a]
        aggregated = dp_aggregate(
            results,
            global_params=self._global,
            aggregatable_keys=aggregatable_keys,
            clip_norm=self.dp_clip_norm,
            noise_multiplier=self.dp_noise_multiplier,
            generator=self._dp_generator,
        )
        # Re-attach the frozen A exactly as the non-DP FFA path does (bit-identical, never noised).
        if self.aggregation == "FFA_LORA":
            for k, v in self._frozen_a.items():
                aggregated[k] = v.clone()
        return aggregated

    @staticmethod
    def _client_keys(results):
        """The parameter key list of the first client (homogeneity already asserted upstream),
        decoding a JSON-string payload if necessary via the shared update normalizer."""
        _cid, params, _n = normalize_update(results[0])
        return list(params.keys())

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        loss, metrics = self.evaluate_fn(server_round, parameters)
        log.info("FedLoRA eval round=%d loss=%.4f metrics=%s", server_round, loss, metrics)
        return loss, metrics

    def _assert_client_keys_allowed(self, results):
        """FR-23: enforce a server-side allowlist — every client key must be in the server's known
        adapter surface (``initial_parameters``).

        ``_assert_homogeneous`` only checks clients against EACH OTHER, so a min-clients=1 client (or
        a colluding full cohort) could append keys outside the adapter — e.g. poisoned base-model
        weights under their peft state-dict names — which would then be averaged into the global,
        broadcast to every peer, and packaged into the LORA_ADAPTER registry bundle. Reject any key
        the server did not initialise. Clients may still send a subset (FFA re-attaches the frozen
        A), but never a superset: smuggled tensors have no home in the server's adapter surface.
        """
        expected = set(self.initial_parameters.keys())
        for entry in results:
            _cid, params, _n = normalize_update(entry)
            extra = set(params.keys()) - expected
            if extra:
                raise ValueError(
                    f"FedLoRA client sent keys outside the server's adapter surface: "
                    f"{sorted(extra)} — refusing to aggregate smuggled tensors "
                    f"(server-side adapter allowlist, FR-23)."
                )

    @staticmethod
    def _assert_homogeneous(results):
        """Raise ValueError if clients disagree on adapter key set or per-key shape (homogeneous rank)."""
        def params_of(entry):
            return entry[1] if len(entry) == 3 else entry[0]
        ref = params_of(results[0])
        ref_shapes = {k: tuple(v.shape) for k, v in ref.items()}
        for entry in results[1:]:
            shapes = {k: tuple(v.shape) for k, v in params_of(entry).items()}
            if shapes != ref_shapes:
                raise ValueError(
                    "Heterogeneous LoRA adapters across clients (key/shape mismatch). "
                    "FedLoRA requires homogeneous rank/config."
                )


class FedProx(Strategy):
    """FedProx (Li et al. 2020, "Federated Optimization in Heterogeneous Networks",
    https://arxiv.org/abs/1812.06127).

    Server aggregation is IDENTICAL to FedAvg: the num-examples-weighted mean of the client
    models (reuses FedAvgAggregator — no reimplementation). FedProx's entire difference from
    FedAvg is CLIENT-SIDE: each client minimises its local objective plus a proximal penalty

        min_w  F_i(w) + (mu/2) * || w - w_global ||^2

    which keeps the local solution near the round's starting global model (mitigates client
    drift under heterogeneity + partial work). Because that term lives in the client's loss,
    `mu` never touches server aggregation — so ``mu = 0`` makes aggregation bitwise-identical
    to FedAvg. ``mu`` is plumbed to clients via :meth:`get_client_config` -> config[
    "proximal_mu"], read by :meth:`fedlearn.client.local_trainer.LocalTrainer.fit` exactly the
    way DeComFLClient.fit reads config["learning_rate"].
    """

    def __init__(
            self,
            initial_parameters: OrderedDict[str, torch.Tensor],
            evaluate_fn: Optional[Callable] = None,
            min_fit_clients: int = 1,
            clients_per_round: int = None,
            proximal_mu: float = 0.0,
            learning_rate: float = 0.01,
            local_epochs: int = 1,
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients
        # Client-side hyperparameters shipped via get_client_config (see class docstring).
        self.mu = float(proximal_mu)
        self.learning_rate = float(learning_rate)
        self.local_epochs = int(local_epochs)
        self.aggregator = FedAvgAggregator()

    def initialize_parameters(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        return self.initial_parameters

    def aggregate_fit(
            self,
            server_round: int,
            results: list[Tuple[OrderedDict[str, torch.Tensor], int]],
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        if not results:
            return None
        # Pure FedAvg aggregation — mu is applied in the client objective, never here.
        return self.aggregator.aggregate(results)

    def get_client_config(self) -> dict:
        """Per-round hyperparameters delivered to clients (proto config is map<string,string>).

        Values are stringified so they can flow through the string-keyed/valued protobuf config
        map unchanged; the client coerces them back (float(config["proximal_mu"])).
        """
        return {
            "proximal_mu": str(self.mu),
            "learning_rate": str(self.learning_rate),
            "local_epochs": str(self.local_epochs),
        }

    def evaluate(
            self, server_round: int, parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        if self.evaluate_fn is None:
            return None
        loss, metrics = self.evaluate_fn(server_round, parameters)
        log.info("FedProx eval round=%d loss=%.4f metrics=%s", server_round, loss, metrics)
        return loss, metrics


class FedOpt(Strategy):
    """Server-side adaptive federated optimisation — FedAdam / FedYogi
    (Reddi et al. 2021, "Adaptive Federated Optimization", https://arxiv.org/abs/2003.00295).

    Clients do ordinary local SGD (e.g. LocalTrainer with mu=0). The server aggregates the
    returned client MODELS into ``x_bar`` with the usual num-examples-weighted mean
    (FedAvgAggregator), forms a pseudo-gradient, and applies an Adam-style adaptive step while
    persisting the moment state ``(m, v)`` ACROSS rounds.

    Pseudo-gradient (FR-11 spec): ``g_t = w_global(old) - x_bar``. This is ``-Delta_t`` in the
    paper (they define ``Delta_t = x_bar - w_global`` and ASCEND with +Delta); using ``g_t`` and
    DESCENDING is algebraically identical. Per-coordinate update:

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        FedAdam:  v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        FedYogi:  v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - g_t^2) * g_t^2
        w_global(new) = w_global(old) - eta * m_t / (sqrt(v_t) + tau)

    ``eta`` is the SERVER learning rate; ``tau`` is the adaptivity/degeneracy constant. Moments
    initialise to zero and accumulate, so a round's step depends on the whole history — a later
    round's update differs from an earlier one on the same aggregated input.
    """

    def __init__(
            self,
            initial_parameters: OrderedDict[str, torch.Tensor],
            evaluate_fn: Optional[Callable] = None,
            min_fit_clients: int = 1,
            clients_per_round: int = None,
            server_learning_rate: float = 1.0,
            beta1: float = 0.9,
            beta2: float = 0.99,
            tau: float = 1e-3,
            variant: str = "adam",
            learning_rate: float = 0.01,
            local_epochs: int = 1,
    ):
        variant = str(variant).lower()
        if variant not in ("adam", "yogi"):
            raise ValueError(f"FedOpt variant must be 'adam' or 'yogi', got {variant!r}")

        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.min_fit_clients = min_fit_clients
        self.clients_per_round = clients_per_round if clients_per_round is not None else min_fit_clients

        # Server optimiser hyperparameters.
        self.eta = float(server_learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.tau = float(tau)
        self.variant = variant

        # Client-side SGD hyperparameters (FedOpt clients train plainly; proximal_mu=0).
        self.learning_rate = float(learning_rate)
        self.local_epochs = int(local_epochs)

        self.aggregator = FedAvgAggregator()

        # The server owns the authoritative global model; it needs the PREVIOUS value each round
        # to form g_t = old - x_bar. Kept as float32 to match the aggregator's output dtype.
        self._global = OrderedDict(
            (k, v.detach().clone().to(torch.float32)) for k, v in initial_parameters.items()
        )
        # Persistent Adam moments (m, v), lazily allocated on the first aggregate_fit so their
        # shapes/dtypes match the aggregated tensors exactly. None => "no round has run yet".
        self._m: Optional[OrderedDict[str, torch.Tensor]] = None
        self._v: Optional[OrderedDict[str, torch.Tensor]] = None

        log.info(
            "FedOpt initialised: variant=%s eta=%g beta1=%g beta2=%g tau=%g",
            self.variant, self.eta, self.beta1, self.beta2, self.tau,
        )

    def initialize_parameters(self) -> Optional[OrderedDict[str, torch.Tensor]]:
        return self.initial_parameters

    def aggregate_fit(
            self,
            server_round: int,
            results: list[Tuple[OrderedDict[str, torch.Tensor], int]],
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        if not results:
            return None

        # x_bar: num-examples-weighted mean of the client models (reuse FedAvg aggregation).
        aggregated = self.aggregator.aggregate(results)

        if self._m is None:
            self._m = OrderedDict((k, torch.zeros_like(v)) for k, v in self._global.items())
            self._v = OrderedDict((k, torch.zeros_like(v)) for k, v in self._global.items())

        new_global = OrderedDict()
        for key, old in self._global.items():
            x_bar = aggregated[key].to(old.dtype)
            g = old - x_bar                                    # pseudo-gradient (== -Delta_t)

            m = self.beta1 * self._m[key] + (1.0 - self.beta1) * g
            g2 = g * g
            if self.variant == "adam":
                v = self.beta2 * self._v[key] + (1.0 - self.beta2) * g2
            else:  # yogi
                v = self._v[key] - (1.0 - self.beta2) * torch.sign(self._v[key] - g2) * g2

            new = old - self.eta * m / (torch.sqrt(v) + self.tau)

            self._m[key] = m
            self._v[key] = v
            new_global[key] = new

        self._global = new_global
        # Return a copy so a downstream mutation of the served model can't corrupt server state.
        return OrderedDict((k, v.clone()) for k, v in new_global.items())

    def get_client_config(self) -> dict:
        """Client-side SGD hyperparameters (proto config is map<string,string>)."""
        return {
            "learning_rate": str(self.learning_rate),
            "local_epochs": str(self.local_epochs),
            "proximal_mu": "0.0",
        }

    def evaluate(
            self, server_round: int, parameters: OrderedDict[str, torch.Tensor]
    ) -> Optional[Tuple[float, dict]]:
        if self.evaluate_fn is None:
            return None
        loss, metrics = self.evaluate_fn(server_round, parameters)
        log.info("FedOpt eval round=%d loss=%.4f metrics=%s", server_round, loss, metrics)
        return loss, metrics

