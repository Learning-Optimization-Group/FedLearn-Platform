# 03 — Server Internals

## Table of Contents
- [Overview](#overview)
- [Entry Point: start_server()](#entry-point-start_server)
- [Server Configuration](#server-configuration)
- [Security Wiring in start_server](#security-wiring-in-start_server)
- [JSON Structured Logging](#json-structured-logging)
- [The FLCoordinator](#the-flcoordinator)
  - [State Machine](#state-machine)
  - [Thread Safety Model](#thread-safety-model)
  - [Round Lifecycle](#round-lifecycle)
  - [The Per-Round Dropout Deadline](#the-per-round-dropout-deadline)
  - [Update Submission and Aggregation Trigger](#update-submission-and-aggregation-trigger)
  - [Ingress Defenses on submit_client_update](#ingress-defenses-on-submit_client_update)
  - [Stale Update Rejection](#stale-update-rejection)
  - [Client Registration and Identity Binding](#client-registration-and-identity-binding)
  - [Shipping Client-Side Hyperparameters](#shipping-client-side-hyperparameters)
- [Heartbeat Management](#heartbeat-management)
- [DeComFL Path in the Coordinator](#decomfl-path-in-the-coordinator)
- [The gRPC Servicer](#the-grpc-servicer)
- [Server Thread Pool Sizing](#server-thread-pool-sizing)
- [Graceful Shutdown](#graceful-shutdown)
- [Environment Variables Reference](#environment-variables-reference)

---

## Overview

The server-side of the framework comprises three cooperating objects:

```
start_server()
    │
    ├── creates FLCoordinator       (business logic / state)
    ├── creates grpc.Server          (network I/O)
    └── creates FederatedLearningServiceServicer (RPC dispatcher)
                │
                └── delegates all calls to FLCoordinator
```

The key architectural principle is **strict separation of concerns**:
- **`FLCoordinator`** never touches networking.
- **`FederatedLearningServiceServicer`** never touches model parameters directly — it only calls coordinator methods.
- **`start_server()`** only sets up infrastructure and runs the outer training loop.

---

## Entry Point: start_server()

```python
# server.py
def start_server(
    server_address: str,
    config: ServerConfig,
    strategy: Strategy
) -> tuple[list, dict]:
    """
    Args:
        server_address: e.g. "0.0.0.0:50051"
        config: ServerConfig(num_rounds=10)
        strategy: An instance of FedAvg, DeComFL, or a custom Strategy subclass

    Returns:
        history: List of (round_num, metrics_dict) for each completed round
        final_parameters: The global model state_dict after all rounds
    """
```

Call sequence:

1. **Configure logging** — `configure_logging()` installs the JSON formatter on the **root** logger.
   Called here at the entry point, deliberately **not** at import time (FR-9), so `import fedlearn`
   from a host application does not hijack that application's root logger.
2. **Create coordinator** — `FLCoordinator(strategy, min_clients_for_aggregation=strategy.min_fit_clients, clients_per_round=strategy.clients_per_round)`
3. **Set initial parameters** — `coordinator.set_initial_parameters(strategy.initial_parameters)`
   (the attribute directly; `initialize_parameters()` returns the same object)
4. **Build the auth interceptor** — `interceptor_from_env()`; `None` in dev (fail-open), and it
   **raises** if enforcement is on with no secret (fail-closed on misconfiguration)
5. **Create gRPC server** — with a ThreadPoolExecutor sized for the expected client count and the
   interceptor list
6. **Register servicer** — with the SE-15 `partition_extractor` wired from the same env gate
7. **Bind address** — `check_server_tls_policy()` decides; `add_secure_port` or `add_insecure_port`
8. **Start server** — `grpc_server.start()` begins accepting connections
9. **Run training loop** — iterates rounds, blocking on `coordinator.wait_for_round_to_complete()`
10. **Signal completion + drain** — `coordinator.mark_training_complete()` then sleep
    `FEDLEARN_COMPLETION_DRAIN_SECONDS` (default 3) so connected clients can observe the terminal
    state and exit 0 instead of retry-looping on the `CANCELLED`/`UNAVAILABLE` a hard teardown
    would surface
11. **Return** history and final parameters
12. **Cleanup** — `grpc_server.stop(grace=5)` in the `finally` block

> **`history` only records rounds that produced metrics.** The loop appends `(round_num, metrics)`
> only when `coordinator.get_latest_metrics()` is truthy; a round with no `evaluate_fn`, or one
> whose aggregation returned `None`, logs a warning and is **absent from `history`**. So
> `len(history)` is not necessarily `config.num_rounds`.

---

## Server Configuration

```python
@dataclass
class ServerConfig:
    num_rounds: int = 3
```

Currently a minimal dataclass. Extended configuration (e.g., client selection fraction, convergence criteria) is expected to be passed through the `Strategy` object's constructor arguments rather than `ServerConfig`.

Typical usage:

```python
import fedlearn as fl

model = MyCNN()
strategy = fl.FedAvg(
    initial_parameters=model.state_dict(),
    evaluate_fn=evaluate_global_model,
    min_fit_clients=2,
    clients_per_round=3,
)

history, final_params = fl.server.start_server(
    server_address="0.0.0.0:50051",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

---

## Security Wiring in start_server

Three independent mechanisms are wired at startup, all env-gated and all fail-closed on
misconfiguration rather than silently degrading. `security/` owns the policy; `server.py` only
applies it.

| Mechanism | Gate | Off (default) | On |
|---|---|---|---|
| **TLS (SE-2)** | `FEDLEARN_GRPC_USE_TLS=1`, required by `FEDLEARN_REQUIRE_TLS=1` | `add_insecure_port` + a WARNING log | `ssl_server_credentials` + `add_secure_port`; optional mTLS via `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH=1` |
| **Connection token (SE-1)** | `FEDLEARN_REQUIRE_CLIENT_AUTH=1` | no interceptor — dev **fail-open** | `ConnectionTokenInterceptor` requires a valid `x-connection-token` on all 10 FL RPCs |
| **Identity binding (SE-15)** | the *same* `FEDLEARN_REQUIRE_CLIENT_AUTH=1` | `partition_extractor=None` — binding disabled | one token's server-assigned `partitionId` is pinned 1:1 to one wire `client_id` |

- `check_server_tls_policy()` **raises `TlsPolicyError`** if `FEDLEARN_REQUIRE_TLS=1` while TLS is
  not enabled (never silently serve a deployed profile in plaintext), and again if TLS is on but the
  key or cert path is missing.
- `interceptor_from_env()` **raises `RuntimeError`** if enforcement is on and neither
  `FEDLEARN_FL_TOKEN_SECRET` nor the `APP_JWT_SECRET` fallback is set — the server refuses to start
  rather than run the gate open.
- The interceptor matches by **method name**, not full path, so it is package-agnostic and
  auto-exempts the health check and server reflection. It also enforces **FR-7**: when
  `FEDLEARN_RUN_ID` is set, a token whose `runId` claim names a different run is
  `PERMISSION_DENIED` (identity proven, but not for *this* federation) rather than
  `UNAUTHENTICATED`.
- Verification is PyJWT with an `algorithms` allowlist of `HS256/384/512` only — that allowlist is
  the `alg=none` / alg-confusion defense. The whole HMAC family is accepted because the Java signer
  uses `.signWith(key)` with no explicit algorithm and JJWT infers the alg from the key's bit length;
  hardcoding HS256 would reject a longer secret's tokens. The key is the **base64-decoded** secret,
  and a decoded key shorter than 32 bytes is rejected independently of Java's own check.

### Why SE-15 exists

The wire `client_id` is a self-chosen handle — the proto itself annotates it *"NOT trusted for
authz"*. Without binding, one valid token can be replayed under many `client_id` values, and each
fake id becomes its own averaged update: a single enrolled participant impersonates the whole cohort
and dominates FedAvg/DeComFL aggregation. `FLCoordinator.bind_or_check_identity(partition_id,
client_id)` pins the pair on first use and thereafter rejects both directions of conflict (a
partition presenting a second `client_id`, or a `client_id` claimed by a second partition). The
whole check-then-bind runs under the coordinator lock, so two concurrent first-use calls for the
same partition cannot both win.

The servicer calls `_enforce_client_identity()` **before** each RPC's broad `try/except` — a
`context.abort` raises, and that must reach gRPC rather than be swallowed and remapped to
`INTERNAL`. For `SubmitModelUpdateStream`, whose `client_id` lives inside the chunk stream, the
first chunk is pulled, checked, then chained back into the loop with `itertools.chain`.

---

## JSON Structured Logging

The server configures Python's root logger to emit JSON on startup:

```python
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
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]          # note: REPLACES the root handlers
```

**Why JSON logging?** The Spring Boot backend captures the Python process's stdout and parses it line-by-line to extract structured log objects, which are then broadcast to the React frontend via STOMP WebSocket. Plain text logs would require fragile regex parsing.

> **FR-9 — it is called from `start_server()`, not at import time.** `root.handlers = [handler]`
> *replaces* whatever the host application configured. That is appropriate for the FL server, which
> runs as its own process whose stdout the backend parses — and wrong for anyone doing
> `import fedlearn` inside a larger program. Doing it at import time hijacked the host's root
> logger; doing it at the entry point does not. `tests/test_logging_hygiene.py` guards this.

Example output:
```json
{"timestamp": "2024-01-15T10:23:45.123Z", "level": "INFO", "message": "Starting round 1/10"}
{"timestamp": "2024-01-15T10:23:46.456Z", "level": "INFO", "message": "All 3 clients reported for round 1; aggregating"}
{"timestamp": "2024-01-15T10:23:46.789Z", "level": "INFO", "message": "[Server] Round 1 complete. Metrics: {'loss': 0.43, 'accuracy': 0.87}"}
```

---

## The FLCoordinator

`FLCoordinator` is the stateful core of the server. It owns:
- The current global model parameters
- The current round number (**1-based** — it starts at 1, not 0)
- The list of received updates for the current round
- Client registration, heartbeat records, and the SE-15 partition ↔ client_id bijection
- Client-reported telemetry (`client_metrics_log`, fed by `ReportClientMetrics`)
- Two distinct terminal flags: `stop_requested` (stop **or** error teardown) and
  `training_complete` (all configured rounds finished successfully)
- Two round-failure signals: `last_round_failed` / `last_round_message`

### State Machine

```
              start_round()  ← clears updates AND resets the dropout deadline
                   │
    ┌──────────────▼─────────────────────────────┐
    │              COLLECTING                    │ ← waiting for client updates
    │                                            │
    │  submit_client_update()  ──────────────────┤
    │  (called per client, on its own gRPC thread)│
    └──────┬──────────────────────────────┬──────┘
           │ len(updates)==clients_per_round│ round_timeout_s elapsed
           │  (inline, on the Nth submit)   │  (on the MAIN thread)
           ▼                                ▼
    ┌──────────────────────┐   ┌────────────────────────────────────┐
    │  _trigger_*_and_     │   │   resolve_round_incomplete()       │
    │  evaluation()        │   │   received >= max(1, min_clients)? │
    │                      │   │     yes → force-aggregate (+ flag) │
    │  strategy.aggregate_ │◄──┤     no  → stop_requested = True    │
    │    fit() / evaluate()│   └────────────────────────────────────┘
    │  current_round += 1  │
    │  _round_complete_    │
    │    event.set()       │
    └──────────┬───────────┘
               │
    ┌──────────▼───────────────────┐
    │  wait_for_round_to_complete  │
    │  unblocks in server.py loop  │
    └──────────────────────────────┘
```

> Both paths converge on the same trigger, and `resolve_round_incomplete` is **idempotent**: it
> re-checks `_round_complete_event.is_set()` under the lock and returns if a client completed the
> round in the meantime, so a redundant call cannot double-aggregate.

### Thread Safety Model

The coordinator is accessed concurrently by:
- **The main server thread** — calling `start_round()`, `wait_for_round_to_complete()`,
  `resolve_round_incomplete()` (via the timeout path), `get_latest_metrics()`
- **One or more gRPC threads per active client** — calling `submit_client_update()` /
  `submit_decomfl_update()`, `get_global_model_for_client()`, `register_client()`,
  `bind_or_check_identity()`, `update_client_heartbeat()`, `record_client_metrics()`

Two separate locks protect different parts of the state:

```python
class FLCoordinator:
    def __init__(self, ...):
        self._lock = threading.Lock()        # protects round state + model params + identity maps
        self._round_complete_event = threading.Event()  # signals round completion

        self.heartbeat_lock = Lock()         # protects the heartbeat dict + client_metrics_log
        self.client_heartbeats: Dict[str, dict] = {}
```

> **Critical threading contract:** every aggregation trigger —
> `_trigger_aggregation_and_evaluation()`, `_trigger_decomfl_aggregation_and_evaluation()` and the
> dispatcher `_trigger_round_completion()` — is called **while `_lock` is held**, from
> `submit_client_update` / `submit_decomfl_update` / `resolve_round_incomplete`. The round counter
> increment and `_round_complete_event.set()` are therefore atomic — no waiting thread can observe
> an inconsistent state.
>
> The corollary is that `strategy.aggregate_fit()` and `evaluate_fn` both execute inside that
> critical section. Keep them non-blocking.

`DeComFL` carries a third lock of its own, `_seed_lock`, guarding `get_or_create_seeds` against
concurrent client RPCs racing to generate the same round's seeds (see
[06](06_decomfl.md#seed-generation-and-sharing)).

### Round Lifecycle

```python
def start_round(self):
    """Called by main loop to begin a new round."""
    with self._lock:
        self._client_updates_received.clear()   # prevent stale state leaking across rounds
        self._round_started_at = time.monotonic()  # reset the dropout deadline for this round
    self._round_complete_event.clear()           # reset completion signal

def wait_for_round_to_complete(self):
    """Blocking wait on the main-loop thread. Wakes every second to check the stop flag
    AND the per-round dropout deadline."""
    while not self._round_complete_event.wait(timeout=1.0):
        if self.stop_requested:
            break
        if (time.monotonic() - self._round_started_at) >= self.round_timeout_s:
            self._handle_round_timeout()
            break
```

The 1-second timeout in `wait()` is what lets the server respond to `signal_stop()` and to the
dropout deadline without hanging indefinitely. The deadline uses `time.monotonic()`, so a wall-clock
adjustment cannot lengthen or shorten a round.

### The Per-Round Dropout Deadline

A synchronous FedAvg round blocks forever if a selected client never reports. `round_timeout_s`
bounds it. Precedence: **explicit constructor arg > `FEDLEARN_ROUND_TIMEOUT_S` > the module default
`DEFAULT_ROUND_TIMEOUT_S = 120.0`.** A malformed or non-positive env value logs a warning and falls
back to the default rather than being honoured.

The **policy** (when the deadline has passed) and the **mechanism** (what to do about it) are
deliberately split — `_handle_round_timeout()` decides, `resolve_round_incomplete(reason)` acts:

```python
def resolve_round_incomplete(self, reason: str):
    with self._lock:
        if self._round_complete_event.is_set():
            return                                   # already resolved inline — idempotent
        received = len(self._client_updates_received)
        required = max(1, self.min_clients)          # the strategy aggregates from min_clients
        if received >= required:
            self.last_round_failed = True            # force-aggregated, NOT a clean round
            self.last_round_message = f"Round … {reason}; force-aggregated {received}/{total}…"
            self._trigger_round_completion()         # FR-4: strategy-appropriate dispatch
        else:
            self.last_round_failed = True
            self.stop_requested = True               # never aggregate an EMPTY cohort
            self._round_complete_event.set()
```

Two things this buys, both load-bearing:

- **An empty cohort never aggregates.** FedAvg over zero updates produces a zero-key aggregate that
  would silently *wipe* the global model while the round advanced as a false success. Below the
  floor the run stops instead, with the reason recorded in `last_round_message`.
- **FR-4: the dispatch is strategy-aware.** `_trigger_round_completion()` checks
  `isinstance(self.strategy, DeComFL)` and routes to the DeComFL trigger, because that path also
  writes `gradient_history[round]` (which clients replay to rebuild locally) and guards a `None`
  evaluate. Hardcoding the FedAvg trigger here would silently desync every DeComFL client.

The split also exists so the **in-process simulator** can model dropout deterministically: it calls
the mechanism directly and resolves the round immediately, instead of sleeping out a 120-second
deadline per dropped round (P0-1c). Deployed behaviour is unchanged — `_handle_round_timeout` is
still the only caller on the server path, and it still phrases the failure as a timeout.

### Update Submission and Aggregation Trigger

```python
def submit_client_update(self, client_id, params, num_examples, trained_on_round):
    with self._lock:
        if trained_on_round != self.current_round:
            return                                  # stale or ahead — see below

        # FR-5: dedup. A retried submit must be counted ONCE.
        if any(cid == client_id for cid, _p, _n in self._client_updates_received):
            log.warning("Ignoring duplicate update from %s in round %d", client_id, self.current_round)
            return

        if num_examples <= 0:
            log.warning("Invalid num_examples (%s) from client %s; skipping update", num_examples, client_id)
            return
        num_examples = min(num_examples, self.MAX_NUM_EXAMPLES)   # cap at 100,000

        # ... the ingress defenses below (empty / non-finite / shape / optional clip) ...

        # Tagged with the client identity so a poisoned update is ATTRIBUTABLE.
        self._client_updates_received.append((client_id, params, num_examples))

        if len(self._client_updates_received) == self.clients_per_round:
            self._trigger_aggregation_and_evaluation()
            # Note: this is still inside _lock — intentional

def _trigger_aggregation_and_evaluation(self):
    """MUST be called while self._lock is held."""
    results = list(self._client_updates_received)
    self._client_updates_received.clear()  # free memory immediately

    aggregated_parameters = self.strategy.aggregate_fit(self.current_round, results)

    if aggregated_parameters is not None:
        self._global_model_params = aggregated_parameters
        # FR-22: evaluate() returns None when no evaluate_fn is configured — the constructor
        # default for FedAvg/FedProx/FedOpt/FedLoRA/Robust. Unpacking None as a 2-tuple raised
        # a TypeError INSIDE the lock, after updates were cleared, wedging the round.
        eval_result = self.strategy.evaluate(self.current_round, self._global_model_params)
        if eval_result is not None:
            loss, metrics = eval_result
            self.latest_metrics = {"loss": loss, **metrics}
        else:
            self.latest_metrics = None
    else:
        log.warning("Aggregation for round %d failed", self.current_round)
        self.latest_metrics = None

    # Advance round counter LAST, then signal
    self.current_round += 1
    self._round_complete_event.set()
```

> **The stored update is a 3-tuple `(client_id, params, num_examples)`**, not a 2-tuple. Every
> aggregator accepts both shapes via `server/_update_normalize.normalize_updates`, but the
> coordinator always tags the client id so a rejected or anomalous update is attributable.

> **Memory note:** `self._client_updates_received.clear()` is called immediately after copying to `results`. This frees all client parameter tensors as soon as aggregation completes, which is critical when training large models across many clients.

### Ingress Defenses on submit_client_update

Every one of these runs **before** the update is stored, under the lock. All of them `raise
ValueError`, which the servicer maps to a client-visible `INVALID_ARGUMENT` — the point is that a
bad client is told it is bad, rather than the round dying later and blaming whoever submitted last.

| Guard | Rejects | Why it must be at ingress |
|---|---|---|
| **FR-5 dedup** | a `client_id` already counted this round | `ABORTED`/`UNAVAILABLE`/`DEADLINE_EXCEEDED` are client-retryable, so the server *will* see the same update twice. A second append both inflates that client's weight and can trip the `clients_per_round` trigger with fewer than N distinct clients. First accepted update wins. |
| **Empty update** | `params == {}` | With `clients_per_round == 1` (or an all-empty cohort) FedAvg produces a zero-key aggregate that silently **wipes** the global model while the round advances as a false success. Nothing downstream catches it: the finiteness check is `all([]) == True` and the shape loop is a no-op on zero keys. |
| **SE-3 non-finite** | any NaN/Inf tensor | One NaN corrupts the average for every honest client in the round. Checked in the tensor's own dtype **and** in float32 — a value finite only in a wider dtype (`1e300` as float64) overflows to `inf` when downcast to the aggregation precision, or becomes NaN inside the delta clip. |
| **FR-17 shape mismatch** | a key whose shape differs from the global's | FedAvg does an in-place `torch.add`, so a mismatch would raise deep inside `aggregate_fit` — *after* the round's updates were cleared and the client was ACKed, wedging the round, discarding every honest update, misattributing the error to the last (often honest) submitter, and then tripping the timeout path that stops the whole server. Only keys shared with the global are checked; missing/extra keys are a separate concern (FR-18). |
| **SE-3 delta clip** (opt-in) | nothing — it **bounds** | `client_update_l2_clip` clips `params - global` to a joint L2 budget and returns `global + clipped_delta`. **Off by default**, unlike the DeComFL scalar clamp: a too-tight bound on dense FedAvg deltas biases honest convergence, so it is a deliberate knob. Only float tensors with a matching global key/shape are clipped; integer buffers pass through. |

### Shipping Client-Side Hyperparameters

`get_global_model_for_client()` returns `(params, current_round, config)` where `config` is
**not** empty — it is `self._strategy_client_config()`:

```python
def _strategy_client_config(self) -> dict:
    get_cfg = getattr(self.strategy, "get_client_config", None)
    return {} if get_cfg is None else get_cfg()
```

This is the params-path analogue of DeComFL's `GetDeComFLConfig`. A strategy that needs to push
client-side knobs — FedProx's `proximal_mu`, or the shared `learning_rate` / `local_epochs` —
exposes `get_client_config()` returning a `str -> str` dict (the proto `config` is
`map<string,string>`). `FedProx` and `FedOpt` implement it; `FedAvg`, `FedLoRA`, `DeComFL` and
`RobustAggregator` do not, so they yield `{}` and behave exactly as before. `LocalTrainer.fit()`
reads the values back and coerces them.

### Stale Update Rejection

The coordinator rejects updates that don't match the current round:

```python
if trained_on_round < self.current_round:
    return  # Stale: client trained on an old model, ignore
if trained_on_round > self.current_round:
    return  # Illegal: client is claiming to be ahead of the server
```

**Why stale updates happen:** If a client is slow and submits its update after the round has already advanced (because other clients submitted faster), the update is stale. Accepting it would corrupt the next round's aggregation with parameters from a different global model version.

**Why "ahead" updates are rejected:** This shouldn't happen in correct protocol execution. If it does, it indicates a bug or a malicious client. Rejecting silently is safe.

### Client Registration and Identity Binding

```python
def register_client(self, client_id: str) -> bool:
    with self._lock:
        self._registered_clients.add(client_id)
        return True  # currently always accepts
```

The coordinator's own registration is lightweight — it just records the client ID and always
returns `True`. **Rejection happens one layer up**, in the servicer: `RegisterClient` rejects a
set-but-mismatched `protocol_version` before ever reaching the coordinator, and
`_enforce_client_identity` can abort `PERMISSION_DENIED` before that. The response also carries
`assigned_round = coordinator.current_round`, which is what a **late joiner** should start on.

The anti-Sybil binding lives in a separate method (SE-15), called from the servicer:

```python
def bind_or_check_identity(self, partition_id: int, client_id: str) -> bool:
    with self._lock:
        bound_client = self._partition_to_client.get(partition_id)
        if bound_client is not None:
            return bound_client == client_id                 # this token is already pinned
        if self._client_to_partition.get(client_id, partition_id) != partition_id:
            return False   # this client_id already belongs to a DIFFERENT token
        self._partition_to_client[partition_id] = client_id
        self._client_to_partition[client_id] = partition_id
        return True
```

Trust-on-first-use, a strict bijection in both directions, and the whole check-then-bind under the
lock so two concurrent first-use calls for one partition cannot both win. No capacity limit on
`_registered_clients` exists yet.

---

## Heartbeat Management

The coordinator maintains a `client_heartbeats` dictionary keyed by `client_id`:

```python
{
    "client_0": {
        "status": "training",
        "current_step": 42,
        "total_steps": 100,
        "current_round": 3,
        "last_seen": 1705312345.678  # unix timestamp
    }
}
```

Updated by `update_client_heartbeat()` on every heartbeat RPC (every 5 seconds per client):

```python
def update_client_heartbeat(self, client_id, status, current_step, total_steps, current_round):
    # Record liveness FIRST, so bookkeeping stays accurate even during teardown.
    with self.heartbeat_lock:
        self.client_heartbeats[client_id] = {
            'status': status,
            'current_step': current_step,
            'total_steps': total_steps,
            'current_round': current_round,
            'last_seen': time.time()
        }
    # ... DEBUG progress logging (per-step heartbeats are chatty) ...

    # FR-10: should_stop reflects the coordinator's REAL stop state.
    if self.stop_requested:
        return True, True, f"Server stop requested; {client_id} should abort training"
    return True, False, f"Heartbeat received for {client_id}"
```

> **`should_stop` is WIRED (FR-10) — it is no longer a hardcoded `False`.** It mirrors
> `stop_requested`, which is set by `signal_stop()`, by `mark_training_complete()`, and by the
> quorum-lost branch of `resolve_round_incomplete()`. A globally-stopped coordinator therefore asks
> every heart-beating client to halt its fit loop. On the client side, `GrpcClient.send_heartbeat`
> latches it into a `threading.Event` (`_stop_training`) that the training thread polls between
> local steps — the heartbeat stub is the *only* channel that can reach a client whose training stub
> is blocked inside `fit()`. See
> [02 — Dual-Channel Heartbeat](02_grpc_communication.md#the-heartbeat-channel-is-also-the-stop-signal-fr-10)
> and [04 — Client Internals](04_client_internals.md#server-driven-stop-fr-10).

Active client detection uses a 300-second timeout:

```python
def get_active_clients(self) -> list[str]:
    current_time = time.time()
    with self.heartbeat_lock:
        return [
            client_id
            for client_id, data in self.client_heartbeats.items()
            if current_time - data['last_seen'] < self.heartbeat_timeout  # 300 seconds
        ]
```

---

## DeComFL Path in the Coordinator

When `DeComFL` is the active strategy, the coordinator has a separate update path with its own
validation and its own trigger:

```python
def submit_decomfl_update(self, client_id, gradient_scalars, num_examples, trained_on_round):
    with self._lock:
        # ... same stale / ahead rejection as the FedAvg path ...

        # FR-5: validate the K x P grid against the STRATEGY's configuration, before the
        # scalars can reach aggregate_fit (which indexes grad_scalars[k][p]).
        # Raised, not dropped -> the servicer maps it to INVALID_ARGUMENT.
        if len(gradient_scalars) != strategy.K or any(len(r) != strategy.P for r in gradient_scalars):
            raise MalformedDeComFLSubmission(...)

        # FR-5: dedup (same rationale as the FedAvg path).
        if any(cid == client_id for cid, _, _ in self._client_updates_received):
            return

        # SE-3 layer 1: reject non-finite scalars outright.
        if not all(math.isfinite(g) for row in gradient_scalars for g in row):
            return

        # SE-3 layer 2: CLAMP finite-but-huge scalars into [-tau, +tau].
        tau = self.grad_clip_threshold                       # default 1000.0; None disables
        gradient_scalars = [[max(-tau, min(tau, g)) for g in row] for row in gradient_scalars]

        # Store as 3-tuple: (client_id, gradient_scalars, num_examples)
        self._client_updates_received.append((client_id, gradient_scalars, num_examples))

        if len(self._client_updates_received) >= self.clients_per_round:
            self._trigger_decomfl_aggregation_and_evaluation()

def _trigger_decomfl_aggregation_and_evaluation(self):
    results = list(self._client_updates_received)
    self._client_updates_received.clear()

    aggregated_parameters = self.strategy.aggregate_fit(self.current_round, results)

    if aggregated_parameters is not None:
        self._global_model_params = aggregated_parameters

        avg_gradients = self._calculate_average_gradients(results)
        # FR-6: TYPED dispatch — isinstance guarantees gradient_history exists on a DeComFL.
        if isinstance(self.strategy, DeComFL):
            # A DICT keyed by ROUND, aligned with seed_history + get_rebuild_history.
            self.strategy.gradient_history[self.current_round] = avg_gradients

        eval_result = self.strategy.evaluate(self.current_round, self._global_model_params)
        if eval_result is not None:
            loss, metrics = eval_result
            self.latest_metrics = {"loss": loss, **metrics}

    self.current_round += 1
    self._round_complete_event.set()
```

Three things here are easy to get wrong:

- **`gradient_history` is a `Dict[int, List[List[float]]]` keyed by round number**, assigned to —
  not a list appended to. Same for `seed_history`. The old list-append produced N entries per round
  and off-by-one indexing, and handed each client a *different* perturbation direction, breaking
  DeComFL's shared-seed invariant.
- **The strategy check is `isinstance(self.strategy, DeComFL)`, not `hasattr(...)`** (FR-6).
  `decomfl_strategy` does not import `coordinator`, so the typed import is safe.
- **The scalar guard is two-layered on purpose.** Non-finite is *rejected*; finite-but-huge is
  *clamped*, to preserve liveness. The clamp happens at ingress, before storage, so both consumers
  of the stored scalars read identical values: `aggregate_fit` (which steps the real global model)
  and `_calculate_average_gradients` (which feeds `gradient_history`, which clients replay to
  rebuild). Clamping later would desync them. The honest zeroth-order scalar envelope is ~O(10)
  (O(100) at init), so the `1e3` default is ≥10× above honest support — the identity map on honest
  values, with zero trajectory bias, while capping a 1e9-scale hijack to a bounded, recoverable
  step.

**What the clamp does NOT cover, stated plainly:** within-bound stealth bias and rail collusion. Those
need client identity (SE-1/SE-15) plus reputation. Robust aggregation (median / trimmed-mean) is a
*large-cohort* defense and is close to a no-op at the 1–3 client cohorts this platform often runs —
which is why it is opt-in per project rather than the default. See
[05 — Byzantine-Robust Aggregation](05_strategies.md#byzantine-robust-aggregation-fr-12).

The `_calculate_average_gradients()` computes element-wise average gradient scalars across all clients:

```python
def _calculate_average_gradients(self, results):
    # results = [(client_id, grad_scalars[K][P], num_examples), ...]
    _, first_grads, _ = results[0]
    K = len(first_grads)   # local steps
    P = len(first_grads[0]) # perturbations

    avg = [[0.0] * P for _ in range(K)]
    for _, grads, _ in results:
        for k in range(K):
            for p in range(P):
                avg[k][p] += grads[k][p]

    num_clients = len(results)
    for k in range(K):
        for p in range(P):
            avg[k][p] /= num_clients

    return avg
```

These averaged gradients are stored in `strategy.gradient_history[round]` so that late-joining or
straggling clients can replay missed rounds and rebuild their local model to match the current global
state. Note the average is **unweighted** — `num_examples` is collected but not used, which the proto
comment states explicitly. See
[06 — Model Rebuild for Missed Rounds](06_decomfl.md#model-rebuild-for-missed-rounds) for the
consumer side, including the `DeComFLRebuildGap` fail-loud guard and history pruning.

---

## The gRPC Servicer

`FederatedLearningServiceServicer` is a thin I/O adapter. Every method:
1. Enforces the SE-15 identity binding (**before** any `try/except` — `context.abort` raises, and
   that must reach gRPC rather than be swallowed and remapped)
2. Deserialises the protobuf request
3. Calls the appropriate coordinator method
4. Serialises the protobuf response, mapping exceptions to the right status code

The only logic that genuinely lives *in* the servicer rather than the coordinator is the part that
is about the transport itself: protocol-version negotiation, the SE-18 streamed-upload caps, the
safetensors encode/chunking for `GetGlobalModelStream`, and the exception → status-code map. Example:

```python
class FederatedLearningServiceServicer(fedlearn_pb2_grpc.FederatedLearningServiceServicer):

    def __init__(self, coordinator: FLCoordinator):
        self.coordinator = coordinator

    def RegisterClient(self, request, context):
        self._enforce_client_identity(request.client_id, context)   # SE-15, before any try:
        if request.protocol_version and request.protocol_version != SERVER_PROTOCOL_VERSION:
            return RegisterClientResponse(status=REJECTED, message="Protocol version mismatch: …",
                                          protocol_version=SERVER_PROTOCOL_VERSION)
        success = self.coordinator.register_client(request.client_id)
        return RegisterClientResponse(
            status=ACCEPTED if success else REJECTED,
            message="...",
            assigned_round=self.coordinator.current_round,   # late joiners start on the live round
            protocol_version=SERVER_PROTOCOL_VERSION,
        )

    def SubmitModelUpdate(self, request, context):
        self._enforce_client_identity(request.client_id, context)   # SE-15, before any try:
        try:
            params, num_examples = proto_to_parameters(request.parameters)
            self.coordinator.submit_client_update(
                request.client_id, params, num_examples, request.trained_on_round
            )
            return SubmitModelUpdateResponse(received=True)
        except ValueError as e:
            # A client's fault -> INVALID_ARGUMENT, not the generic INTERNAL that hides it.
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid model update: {e}")
        except Exception:
            context.abort(grpc.StatusCode.INTERNAL, "An internal server error occurred.")
```

The `GetGlobalModelStream` and `SubmitModelUpdateStream` RPCs use Python generators — `yield` for
server streaming, iterator consumption for client streaming. `SERVER_PROTOCOL_VERSION = 2` and must
equal the mobile client's `kProtocolVersion`.

> **Two RPCs report failure without aborting.** `GetDeComFLConfig` and `SubmitGradientScalars` use
> `context.set_code()` / `set_details()` and then `return` a default-valued response of their own
> type, rather than `context.abort()`. So a DeComFL client sees a status code *and* an empty
> message. (`SubmitGradientScalars` returning the wrong response type was FR-6's bug; it now returns
> `SubmitGradientScalarsResponse(received=False)`.) `Heartbeat` likewise swallows an internal error
> into `HeartbeatResponse(acknowledged=False, should_stop=False)` — a telemetry failure never fails
> the round, and neither does `ReportClientMetrics`.

---

## Server Thread Pool Sizing

The thread pool must be large enough to handle all concurrent RPCs:

```python
max_expected_clients = int(os.environ.get('MAX_CLIENTS', 50))
optimal_workers = (max_expected_clients * 2) + 10
# 50 clients → 110 workers
# 10 clients → 30 workers
```

The `* 2` factor accounts for the fact that each client typically has **two concurrent RPCs** at any point:
1. The primary training RPC (e.g., a long-running `GetGlobalModelStream`)
2. The heartbeat RPC (fires every 5 seconds)

The `+ 10` provides overhead for registration RPCs, status polls, etc.

---

## Graceful Shutdown

The server shuts down in a controlled sequence:

```python
# server.py
try:
    history = []
    for round_num in range(1, config.num_rounds + 1):
        coordinator.start_round()
        coordinator.wait_for_round_to_complete()

        if coordinator.stop_requested:
            break

        metrics = coordinator.get_latest_metrics()
        if metrics:
            history.append((round_num, metrics))   # rounds with no metrics are NOT appended
        else:
            logging.warning("Round %d completed but no metrics available.", round_num)

    final_parameters = coordinator.get_global_model_params()

    # Tell still-connected clients the run is OVER, then give them a window to see it.
    coordinator.mark_training_complete()
    drain = float(os.environ.get("FEDLEARN_COMPLETION_DRAIN_SECONDS", "3"))
    if drain > 0:
        time.sleep(drain)

    return history, final_parameters

except KeyboardInterrupt:
    return [], {}
except Exception as e:
    coordinator.signal_stop()   # releases any waiting threads
    return [], {}
finally:
    grpc_server.stop(grace=5)   # waits up to 5s for active RPCs to complete
```

### Two terminal flags, deliberately distinct

```python
def signal_stop(self):                 # stop OR error teardown — NOT a clean completion
    self.stop_requested = True
    self._round_complete_event.set()

def mark_training_complete(self):      # all configured rounds finished successfully
    with self._lock:
        self.training_complete = True
        self.stop_requested = True     # so the -1 sentinel also fires
    self._round_complete_event.set()
```

`GetServerStatus` reports `TRAINING_COMPLETE` when **either** is set, so a client can always exit
cleanly; but `training_complete` is the precise "all rounds done" signal, and only
`mark_training_complete()` sets it.

### How a client learns the run is over — three routes

1. **The `-1` sentinel.** `get_global_model_for_client()` returns `(None, -1, {})` once
   `stop_requested` is set; `GetDeComFLConfig` returns `current_round = -1`. `start_client` /
   `start_decomfl_client` break on it.
2. **`should_stop` on the heartbeat** (FR-10) — reaches a client even while its training stub is
   blocked inside `fit()`.
3. **`GetServerStatus` → `TRAINING_COMPLETE`.** This is the one that survives a hard teardown: a
   completed run tears the RPCs down, so an in-flight call surfaces as `CANCELLED`/`UNAVAILABLE`.
   `GrpcClient.server_reports_complete()` probes on the *heartbeat* channel (the training stub may
   be in a bad state after a cancelled call) and, if the server is still draining, reports the run
   finished — which is exactly what the drain window is for. A genuinely crashed server returns
   `None` from the probe, so the client keeps its disconnect handling and reports
   `OUTCOME_DISCONNECTED` rather than a false success.

```python
# client.py
parameters, server_round, config = comm_client.get_global_model()
if server_round == -1:
    log.info("Server signalled training complete; shutting down")
    break
```

---

## Environment Variables Reference

Everything the server process reads. TLS/auth variables are also listed in
[02 — TLS Configuration](02_grpc_communication.md#environment-variable-reference).

| Variable | Default | Read by | Description |
|----------|---------|---------|-------------|
| `MAX_CLIENTS` | `50` | `server.py` | Sizes the gRPC thread pool: `(MAX_CLIENTS * 2) + 10` workers |
| `FEDLEARN_ROUND_TIMEOUT_S` | `120.0` | `coordinator.py` | Per-round dropout deadline. Invalid or ≤ 0 → warning + fall back to the default |
| `FEDLEARN_COMPLETION_DRAIN_SECONDS` | `"3"` | `server.py` | Post-run window so clients can observe `TRAINING_COMPLETE`; ≤ 0 skips it |
| `FEDLEARN_GRPC_USE_TLS` | `"0"` | `security/tls.py` | `"1"` to enable TLS |
| `FEDLEARN_REQUIRE_TLS` | unset | `security/tls.py` | `"1"` to **require** TLS — refuses to serve plaintext (SE-2) |
| `FEDLEARN_GRPC_SERVER_KEY` | — | `server.py` | Path to server private key (PEM); required when TLS is on |
| `FEDLEARN_GRPC_SERVER_CERT` | — | `server.py` | Path to server certificate (PEM); required when TLS is on |
| `FEDLEARN_GRPC_ROOT_CERT` | — | `server.py` | CA cert for verifying clients (mTLS) |
| `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH` | `"0"` | `server.py` | `"1"` to require client **certificates** (mTLS) |
| `FEDLEARN_REQUIRE_CLIENT_AUTH` | unset | `security/interceptor.py`, `security/identity.py` | `"1"` to require a valid **connection token** (SE-1) and enable identity binding (SE-15). Distinct from the mTLS variable above — do not confuse them |
| `FEDLEARN_FL_TOKEN_SECRET` | — | `security/interceptor.py` | Base64 HMAC secret for connection tokens (SE-7) |
| `APP_JWT_SECRET` | — | `security/interceptor.py` | Fallback secret until the dedicated one is provisioned |
| `FEDLEARN_RUN_ID` | unset | `security/interceptor.py` | FR-7: binds the server to its run; a token for another run → `PERMISSION_DENIED` |
| `FEDLEARN_MAX_UPLOAD_BYTES` | 2 GiB | `grpc_servicer.py` | SE-18 streamed-upload byte cap |
| `FEDLEARN_MAX_UPLOAD_CHUNKS` | `100000` | `grpc_servicer.py` | SE-18 streamed-upload chunk cap |
| `FEDLEARN_MAX_UPLOAD_SECONDS` | `600.0` | `grpc_servicer.py` | SE-18 active-streaming deadline; ≤ 0 disables it |
| `FEDLEARN_USE_COMPRESSION` | `"0"` | `communication/serializer.py` | `"1"` to enable LZ4 — read at **import** time; the gRPC streaming path ignores it |
| `FEDLEARN_CHUNK_SIZE_MB` | `"4"` | `communication/serializer.py` | Default `serializer.CHUNK_SIZE`. Does **not** resize gRPC upload chunks (those are an explicit 50 MB) |

Client-side only: `FEDLEARN_CONNECTION_TOKEN` (`security/client_interceptor.py`),
`FEDLEARN_GRPC_CLIENT_KEY` / `FEDLEARN_GRPC_CLIENT_CERT` (`client/grpc_client.py`).
