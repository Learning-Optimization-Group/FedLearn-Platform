# 04 — The Federated Set

> **Part of:** [FedLearn Platform Docs](../README.md) → [FL Runtime Wiki](./README.md)

Which tensors actually cross the wire, and why the answer is not simply "the model".

## Table of Contents
- [Two Filters, Applied in Order](#two-filters-applied-in-order)
- [The Non-Float32 Exclusion](#the-non-float32-exclusion)
- [Subset Federation](#subset-federation)
- [Saving: `merge_non_federated`](#saving-merge_non_federated)
- [Evaluation Strictness](#evaluation-strictness)
- [The `.npz` Contract](#the-npz-contract)
- [Where the Framework Takes Over](#where-the-framework-takes-over)

---

## Two Filters, Applied in Order

Both `fl_server.py` and `client.py` narrow the model to a *federated set* before anything touches the gRPC channel, and both apply the same two filters in the same order:

```
model.state_dict()
   │
   │  (1) WIRE FILTER — federable_state(): keep float32 only
   ▼
   │  (2) ARM FILTER — keep keys matching the arm's trainable prefixes
   ▼
the federated set
```

The order is load-bearing and so is the sharing. Both sides call the **same helpers** from the framework (`fedlearn.estimators.params`) rather than each implementing its own predicate:

```python
from fedlearn.estimators.params import federable_state, non_federable_names
```

> **Client and server must federate an identical key set, and two independent filters would drift. That divergence is how the frozen arm broke twice.** This is the single most repeated failure mode in this codebase's history — a server/client disagreement about which tensors are federated.

Both sides also **log what they withheld**. Silently dropping tensors is what would make this dangerous; a run has to be able to say what it withheld and a reader has to be able to audit it.

---

## The Non-Float32 Exclusion

The safetensors wire is **float32-only by design** — it must decode in the libtorch-free mobile C++ core, so other dtypes raise rather than being silently coerced.

Every `nn.BatchNorm*` module carries an int64 `num_batches_tracked` buffer. Consequently a `FULL`-arm run on **any** BatchNorm model failed on the first `GetGlobalModel` — which excluded ResNets, the most common architecture in the FL literature, from the `FULL` arm entirely. `CIFAR_RESNET18` shipped able to declare only `FROZEN_HEAD` for exactly this reason.

The fix (commit `3b13204`) filters non-float32 tensors **out of the federated set** rather than raising:

```python
def federable_state(state):     # framework/src/fedlearn/estimators/params.py:85
    return OrderedDict((k, v) for k, v in state.items() if v.dtype == torch.float32)

def non_federable_names(state): # …:111
    return [k for k, v in state.items() if v.dtype != torch.float32]
```

Be precise about the scope of this change:

- **What is dropped:** non-float32 tensors. In practice `num_batches_tracked`, a batch **counter** — averaging it across clients is meaningless, so nothing of value is lost and each client keeps its own.
- **What is *not* dropped:** `running_mean` and `running_var`. They are float32 and continue to be averaged. Excluding those too would be **FedBN** — a different algorithm with different convergence behaviour — rather than a fix for what the wire can carry.
- For a float32-only model the function is the identity: same keys, same order, same tensor objects. No existing recipe changes behaviour.

### Where it appears in this layer

**Client** (`client.py:724-737`, in `get_parameters()` on the `FULL` path):

```python
from fedlearn.estimators.params import federable_state, non_federable_names
full = self.net.state_dict()
withheld = non_federable_names(full)
if withheld and not getattr(self, "_logged_withheld", False):
    print(f"[Client] Withholding {len(withheld)} non-float32 tensor(s) from the federated "
          f"set (kept local): {withheld[:4]}{' ...' if len(withheld) > 4 else ''}")
    self._logged_withheld = True
return federable_state(full)
```

The log is latched to once per client so a long run does not repeat it every round.

**Server** (`fl_server.py:658-672`), applied to the parameters loaded from the `.npz`, **before** the arm filter, and with the count carried in `_n_withheld` because the evaluation closure reads it on every path:

```python
_withheld = non_federable_names(initial_parameters)
_n_withheld = len(_withheld)
if _withheld:
    logging.info(f"Withholding {len(_withheld)} non-float32 tensor(s) from the federated set "
                 f"(kept local, restored at save): {_withheld[:4]}…")
…
full_initial_parameters = OrderedDict(initial_parameters)   # keep the complete model
initial_parameters = federable_state(initial_parameters)
```

`_n_withheld` is bound to `0` *before* the `try:` block, so an early-exit or exception path can never hit an unbound local.

`tests/test_declared_arms_are_wire_compatible.py` guards the class of defect this belongs to: **a recipe must not declare an arm whose payload cannot cross the wire.** Note that building the model does *not* catch it — the model is fine; the payload is what fails — so the test builds the payload for each declared arm and checks it against the codec.

---

## Subset Federation

Under `FROZEN_HEAD` or `OVA_LP` the federated set narrows again, to the keys matching the arm's trainable prefixes. The frozen backbone stays local and never rides the wire.

**Client**, `get_parameters()`:

```python
if USE_DERIVED:
    from fedlearn.estimators.params import trainable_state
    return trainable_state(self.net)
```

`trainable_state` returns the `requires_grad` parameters in `named_parameters()` order — the *correct* payload for a subset arm, and the correct `initial_parameters` for a DeComFL server. Passing a full `state_dict()` instead would include buffers and frozen params that the client's `requires_grad` flatten omits, so `d_server > d_client` and the shared-seed perturbation `z` silently misaligns.

`LLM_LORA` follows the same shape via a different mechanism: it uploads `get_peft_model_state_dict(net, save_embedding_layers=False)` narrowed to the adapter keys.

**Client**, `fit()`: the aggregated subset is loaded with `strict=False`, which is what keeps the frozen backbone local and off the wire while still applying the head the server aggregated.

**Server**: the arm filter described in [03](03_training_arms.md#server-side). If the arm's prefixes match **no** key in the `.npz`, the server logs the keys it saw and `exit(1)`s — an empty federated set is a hard failure, not a silently degenerate run.

---

## Saving: `merge_non_federated`

`merge_non_federated` (`fl_server.py:229`) is the save-side half of the load-side contract, and it exists because of a real live failure on 2026-08-13. After a successful 3-round `FROZEN_HEAD` run, the saved model contained **two keys**:

```
PRE  keys (10): conv1.weight … fc3.bias
POST keys  (2): fc3.bias, fc3.weight
```

The server writes the final *global* model to `--model-path`, and under a subset arm the global model **is** the head. So the run overwrote the only full copy of the model with a 2-key file: the artifact was not a usable model (you cannot run inference without a backbone), and the backbone it was trained against was unrecoverable.

```python
def merge_non_federated(final_parameters, full_initial_parameters):
    if not full_initial_parameters:
        return final_parameters
    merged = OrderedDict()
    for key, value in full_initial_parameters.items():
        merged[key] = final_parameters.get(key, value)
    for key, value in final_parameters.items():   # defensive: keys the original lacked
        if key not in merged:
            merged[key] = value
    return merged
```

Four properties:

- **The backbone merged back is the one the run actually used**, not a fresh init — the head was trained against those exact frozen weights, so any other pairing describes a model that never existed.
- **Key order follows the original state_dict**, because ordering is load-bearing for the safetensors wire and for the sha256 an artifact is addressed by.
- **It never raises.** Losing a trained head to protect a backbone would be worse than the bug.
- **It is the identity for the `FULL` arm** with nothing withheld.

The same merge also restores the non-float32 tensors the wire filter withheld, so a BatchNorm model's `num_batches_tracked` is present in the saved `.npz` even though it never crossed the wire.

Guarded by `tests/test_frozen_save_preserves_backbone.py`.

---

## Evaluation Strictness

`evaluation_load_is_strict(model_type, training_arm, withheld)` (`fl_server.py:259`) decides whether `server_side_evaluate` loads the global model with `strict=True`.

```python
if withheld:
    return False        # the wire legitimately withheld non-float32 tensors, even under FULL
if str(model_type).upper() == "TINYNET_GOLDEN":
    return False        # syncs only its 25 trainable fc1 params
try:
    return recipes.trainable_prefixes(model_type, training_arm) is None
except ValueError:
    return True         # unknown recipe/arm: keep the stricter behaviour
```

It was previously the single expression `model_type.upper() != 'TINYNET_GOLDEN'` — correct while that was the only subset-federating recipe, and wrong the moment any recipe could run `FROZEN_HEAD`. It let a completed frozen round fail evaluation with `Missing key(s) in state_dict: conv1.weight, …`.

**Strictness is kept for the `FULL` arm on purpose.** There it is a real guard against a malformed payload, and relaxing it globally to accommodate the frozen arm would discard that. The `withheld` short-circuit is what let BatchNorm models be unblocked for `FULL` without immediately failing evaluation on the very keys the wire had been told not to carry.

Guarded by `tests/test_server_eval_strictness.py`.

---

## The `.npz` Contract

This layer's on-disk model format is a NumPy `.npz`, written by `init_model.py` and rewritten by `fl_server.py`, and read by both plus `infer.py`.

- **`.` is not legal in an `npz` member name**, so every key is stored as `key.replace('.', '__DOT__')` and un-escaped on load.
- **`allow_pickle=False` on every read.** Both `fl_server.py` and `infer.py` load with it explicitly; a non-`ndarray` member is skipped rather than deserialised (`fl_server.py` logs a warning, `infer.py` skips silently).
- **The `.npz` deliberately keeps the FULL model.** The arm is applied at *load* time on the server, not at save time, so the frozen backbone stays recoverable. `merge_non_federated` is what upholds that on the write path.
- **What `init_model.py` stores varies by type**: the adapter for `LLM_LORA`, the 25-parameter trainable layout for `TINYNET_GOLDEN`, the full `state_dict()` for everything else. See [01](01_entry_points.md#init_modelpy--initial-weights).
- **Write target vs init source are separable.** `--model-path` is always the write target; `--init-model-path` (BA-11) lets the backend point initial weights at the immutable content-addressed registry head without that blob ever being overwritten.

`infer.py` loads the `.npz` with `strict=True` for non-LoRA recipes, so an artifact missing its backbone fails loudly rather than running a half-initialised model — which is precisely how the `merge_non_federated` bug would have surfaced downstream.

---

## Where the Framework Takes Over

Everything above is what this layer decides. Once `fl.server.start_server(...)` / `fl.client.start_client(...)` is called, encoding and transport belong to the framework:

- **Encoding** — `fedlearn/communication/safetensors_codec.py`. Byte-deterministic, float32-only, decodable by the libtorch-free mobile C++ client. Never `torch.save` / pickle.
- **Chunking** — size-gated at the call site and unconditional within the streaming path: `GrpcClient.submit_update()` picks the streaming upload only for transformers (`ALWAYS_STREAM_TRANSFORMERS = True`) or blobs over `STREAMING_THRESHOLD_MB = 100`; once streaming, the blob is always chunked at `FEDLEARN_CHUNK_SIZE_MB` (default 4 MB).
- **Heartbeat** — clients hold two gRPC stubs; the training stub blocks during `fit()` while the heartbeat stub runs on a parallel thread. The heartbeat is bidirectional (FR-10): a `HeartbeatResponse` with `should_stop=True` latches `_stop_training`, which the fit loop polls to abort a round. `client.py` wires its `ZOSLClient` to the transport via `set_grpc_client()` and feeds round progress through a callback.
- **TLS** — implemented and opt-in (SE-2). The client uses `insecure_channel` unless `FEDLEARN_GRPC_USE_TLS=1`. Default is plaintext. Client authentication is a separate mechanism, `FEDLEARN_CONNECTION_TOKEN` (SE-14).

See [Framework: gRPC Communication](../framework/02_grpc_communication.md) and [Framework: Client Internals](../framework/04_client_internals.md) for the transport side.
