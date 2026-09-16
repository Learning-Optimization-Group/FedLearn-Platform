# 02 — gRPC Communication Layer

## Table of Contents
- [Overview](#overview)
- [The Protocol Buffer Contract](#the-protocol-buffer-contract)
- [Service Definition — All 10 RPCs](#service-definition--all-10-rpcs)
- [Message Types In Depth](#message-types-in-depth)
- [Protocol Version Negotiation](#protocol-version-negotiation)
- [Serializer — Tensor ↔ Proto Conversion](#serializer--tensor--proto-conversion)
- [Chunked Streaming for Large Models](#chunked-streaming-for-large-models)
- [Streamed-Upload Resource Caps (SE-18)](#streamed-upload-resource-caps-se-18)
- [LZ4 Compression (Optional)](#lz4-compression-optional)
- [TLS Configuration](#tls-configuration)
- [Retry Logic and Backoff](#retry-logic-and-backoff)
- [Keepalive Tuning](#keepalive-tuning)
- [Connection Limits and Message Sizes](#connection-limits-and-message-sizes)
- [Dual-Channel Heartbeat Architecture](#dual-channel-heartbeat-architecture)
- [Regenerating Generated Code](#regenerating-generated-code)
- [Common Errors and Debugging](#common-errors-and-debugging)

---

## Overview

All communication between the Python FL server and its clients goes through a single gRPC service defined in `fedlearn.proto`. The design goals were:

1. **Support arbitrarily large models** — LLMs can exceed 10 GB, and a single unary message would have to be buffered whole. Downloads are *always* server-streaming (`GetGlobalModelStream`); on the **upload** half the client auto-detects model size (and transformer architecture) and switches from a unary call to the client-streaming RPC.
2. **Survive flaky networks** — exponential backoff retries on all transient gRPC errors.
3. **Heartbeat without blocking training** — a second dedicated gRPC channel is used exclusively for heartbeats so they never queue behind large model uploads.
4. **Opt-in TLS** — plaintext is the default; TLS/mTLS is controlled entirely by environment variables, with zero code changes needed to flip between them. Deployed profiles fail closed rather than silently serve plaintext (SE-2).
5. **No pickle on the wire** — model state-dicts travel as a deterministic, float32-only **safetensors** blob, decodable by the libtorch-free mobile C++ core and byte-identical across languages.

---

## The Protocol Buffer Contract

**Canonical source of truth:** `proto/fedlearn/v2/fedlearn.proto` (package `fedlearn.v2`, governed by `buf`).

The framework does **not** own the contract — it keeps a **byte-identical mirror** at
`framework/src/fedlearn/communication/protos/fedlearn.proto` for its own codegen. `scripts/check_proto_mirror.sh`
diff-gates all three in-tree mirrors (framework `fedlearn.proto`, `mobile_client/proto/fedlearn/v2/fedlearn.proto`,
framework `fot.proto` against `proto/fedlearn/fot/v1/fot.proto`) and runs in CI as `proto.yml`. **Never hand-edit a
mirror** — edit the canonical and re-copy.

The `.proto` file is compiled to Python stubs stored in `communication/generated/`:
- `fedlearn_pb2.py` / `fedlearn_pb2.pyi` — message classes + type stubs
- `fedlearn_pb2_grpc.py` — stub and servicer base class
- `fot_pb2.py` / `fot_pb2_grpc.py` — the same for the FoT contract

```proto
syntax = "proto3";
package fedlearn.v2;

option java_package = "com.fedlearn.v2";
option java_multiple_files = true;

service FederatedLearningService {
  // Registration, status and liveness
  rpc RegisterClient        (RegisterClientRequest)        returns (RegisterClientResponse);
  rpc GetServerStatus       (GetServerStatusRequest)       returns (GetServerStatusResponse);
  rpc Heartbeat             (HeartbeatRequest)             returns (HeartbeatResponse);

  // Model distribution (server → clients) and update collection (clients → server)
  rpc GetGlobalModel        (GetGlobalModelRequest)        returns (GetGlobalModelResponse);
  rpc GetGlobalModelStream  (GetGlobalModelRequest)        returns (stream ModelChunk);
  rpc SubmitModelUpdate     (SubmitModelUpdateRequest)     returns (SubmitModelUpdateResponse);
  rpc SubmitModelUpdateStream(stream ModelUpdateChunk)     returns (SubmitModelUpdateResponse);

  // DeComFL-specific
  rpc GetDeComFLConfig      (GetDeComFLConfigRequest)      returns (GetDeComFLConfigResponse);
  rpc SubmitGradientScalars (SubmitGradientScalarsRequest) returns (SubmitGradientScalarsResponse);

  // Telemetry
  rpc ReportClientMetrics   (ReportClientMetricsRequest)   returns (ReportClientMetricsResponse);
}
```

---

## Service Definition — All 10 RPCs

### Standard FL RPCs

| RPC | Direction | Pattern | Purpose |
|-----|-----------|---------|---------|
| `RegisterClient` | client → server | Unary | Registers a client ID + run + protocol version; returns ACCEPTED/REJECTED and the round a late joiner should start on |
| `GetGlobalModel` | server → client | Unary | Download model parameters as a `ModelParameters` proto. **Implemented in the servicer, called by nothing** — see the note below |
| `GetGlobalModelStream` | server → client | **Server streaming** | Download the model as a chunked safetensors blob (50 MB chunks) |
| `SubmitModelUpdate` | client → server | Unary | Upload updated parameters (small, non-transformer models) |
| `SubmitModelUpdateStream` | client → server | **Client streaming** | Upload the model as a chunked safetensors blob (50 MB chunks) |
| `GetServerStatus` | client → server | Unary | Poll server state, round, quorum, active clients, round deadline |
| `Heartbeat` | client → server | Unary | Report liveness + training progress; carries the server's `should_stop` back (FR-10) |
| `ReportClientMetrics` | client → server | Unary | Client-side telemetry reporting (loss / accuracy / compute ms / client type) |

### DeComFL RPCs

| RPC | Direction | Pattern | Purpose |
|-----|-----------|---------|---------|
| `GetDeComFLConfig` | client → server | Unary | Download random seeds + missed-round history |
| `SubmitGradientScalars` | client → server | Unary | Upload O(K×P) scalars instead of full model |

> **Why two separate model-download RPCs — and which one actually runs.** `GetGlobalModel` was
> intended as a unary fast path for small models, but **no shipped client calls it**:
> `GrpcClient.get_global_model()` (`grpc_client.py:138-217`) unconditionally opens
> `GetGlobalModelStream`, and so does the mobile C++ core
> (`FedLearnClient.cpp:253`). The servicer implements both, so the unary handler is live code with
> no live caller — the download path you are debugging is always the streaming one. The
> unary/streaming split that *is* exercised is on the **upload** half (`submit_update`, below).

---

## Message Types In Depth

### Core Data Types

```proto
// A single typed tensor. The ONLY weight-bearing wire type; no torch.save blobs.
message Tensor {
  bytes          data  = 1;   // raw bytes, dtype+dims interpret them
  repeated int64 dims  = 2;   // shape, e.g., [256, 512]
  string         dtype = 3;   // whitelist: "float32","float64","int32","int64","uint8","bool"
}

// A full model state_dict
message ModelParameters {
  map<string, Tensor> tensors              = 1;   // key = parameter name
  int64               num_examples_trained = 2;   // for weighted aggregation
}
```

> **`ModelParameters.tensors` is a protobuf `map`, and a `map` field iterates in an UNSPECIFIED
> order.** The unary upload path therefore does **not** preserve `state_dict` insertion order. Every
> consumer of that path is by-name (`FedAvgAggregator` averages per key; `load_state_dict` matches by
> name), so this is harmless there — but it is exactly why
> `server/subset_federation.validate_subset_update` compares trainable keys as a **set**, not a
> sequence. The order-critical DeComFL flat-vector layout travels a different path entirely
> (`estimators/params.param_layout`) and is unaffected.

### Registration, Status and Heartbeat

Registration carries more than an ID — it binds the client to a run and negotiates the wire version.

```proto
message RegisterClientRequest {
  string client_id        = 1;   // client-chosen handle (display only; NOT trusted for authz)
  string run_id           = 2;   // UUID string of the fl_runs row this client joins
  int32  protocol_version = 3;   // MUST equal the server's; a set-but-mismatched value -> REJECTED
  string enrollment_token = 4;   // backend-minted; MVP validates permissively (log-only)
}

message RegisterClientResponse {
  enum Status { STATUS_UNSPECIFIED = 0; ACCEPTED = 1; REJECTED = 2; }
  Status status           = 1;
  string message          = 2;
  int32  assigned_round   = 3;   // the live round — what a LATE JOINER should start on
  int32  protocol_version = 4;   // the server's version (client logs on mismatch)
}

message GetServerStatusResponse {
  enum ServerState {
    STATE_UNSPECIFIED = 0; INITIALIZING = 1; WAITING_FOR_CLIENTS = 2;
    TRAINING = 3; AGGREGATING = 4; TRAINING_COMPLETE = 5; FAILED = 6;
  }
  ServerState server_state                = 1;
  int32       current_round               = 2;
  int32       required_clients_for_round  = 3;
  int32       received_updates_this_round = 4;
  int32       active_clients              = 5;
  int64       round_deadline_unix_ms      = 6;  // when the round hard-stops — NO infinite wait
}

message HeartbeatRequest {
  string client_id = 1; string run_id = 2; string status = 3;   // free-text phase
  int32 current_step = 4; int32 total_steps = 5; int32 current_round = 6;
}
message HeartbeatResponse {
  bool acknowledged = 1;
  bool should_stop  = 2;   // WIRED (FR-10): the server telling the client to abort
  string message    = 3;
}
```

> **Which states the servicer actually emits.** `GetServerStatus` returns only three of the seven
> `ServerState` values: `TRAINING_COMPLETE` (when `training_complete` **or** `stop_requested`),
> `WAITING_FOR_CLIENTS` (active clients below the quorum), else `TRAINING`. `INITIALIZING`,
> `AGGREGATING` and `FAILED` are declared in the contract but never set by this server.
> `round_deadline_unix_ms` is a **rolling** `now + round_timeout_s`, not a precise per-round
> start-stamp — a client's poll never implies an infinite wait, but the value is not a fixed
> instant either.

### Streaming Chunk Messages

Both chunk messages carry **explicit v2 framing** — the codec, compression flag, total size and payload
hash all travel *on the wire* rather than being inferred from environment variables on each side (the
A3-C1/A3-C3 fix). That is what makes save/load symmetric and lets a receiver bounds-check and verify
before it deserialises anything.

```proto
// Server → Client: one chunk of a model download
message ModelChunk {
  int32  chunk_index    = 1;
  int32  total_chunks   = 2;
  bytes  chunk_data     = 3;
  bool   is_final_chunk = 4;
  int32  current_round  = 5;
  map<string,string> config = 6;  // only sent with chunk_index == 0
  // --- v2 framing fields ---
  string codec       = 7;   // "safetensors" (typed; NOT torch.save) — required, validated
  bool   compressed  = 8;   // on the wire, not inferred from env; codec="lz4+safetensors" if true
  int64  total_bytes = 9;   // full reassembled size; receiver bounds-checks cumulative (H5)
  string sha256      = 10;  // hash of the full reassembled blob; receiver verifies (integrity)
}

// Client → Server: one chunk of a model upload
message ModelUpdateChunk {
  string client_id        = 1;
  string run_id           = 2;
  int32  trained_on_round = 3;
  int32  chunk_index      = 4;
  int32  total_chunks     = 5;
  bytes  chunk_data       = 6;
  bool   is_final_chunk   = 7;
  int64  num_examples     = 8;
  // --- v2 framing fields ---
  string codec       = 9;    // "safetensors"
  bool   compressed  = 10;
  int64  total_bytes = 11;   // server bounds-checks cumulative against the SE-18 cap
  string sha256      = 12;   // verified on reassembly
}
```

> **Asymmetry worth knowing before you rely on the upload framing.** The download path
> (`grpc_servicer.GetGlobalModelStream`) populates `codec`, `total_bytes` and `sha256` on **every**
> chunk. The Python upload path does **not**: `GrpcClient._generate_model_chunks` sets only
> `client_id`, `trained_on_round`, `chunk_index`, `total_chunks`, `chunk_data`, `is_final_chunk` and
> `num_examples`, so `codec`/`compressed`/`total_bytes`/`sha256` arrive at their proto defaults
> (`""`, `false`, `0`, `""`). The server copes by construction rather than by inference: it passes
> `compressed=False` explicitly to `chunks_to_parameters` (deliberately *not* keying off its own
> `FEDLEARN_USE_COMPRESSION`, which the client need not share), and its SE-18 caps are enforced on
> the bytes actually received rather than on the declared `total_bytes` — an unset or lying
> `total_bytes` just skips the cheap up-front rejection. So the fields exist in the contract and are
> live on the download half; treat upload-side integrity as **not yet declared on the wire**.

### DeComFL Messages

```proto
// Seeds organized [local_step][perturbation]
message PerturbationSeeds {
  repeated LocalStepSeeds local_steps = 1;
}
message LocalStepSeeds {
  repeated int64 seeds = 1;   // P seeds for this step — int64 to match the C++ int64_t
}

// Gradient scalars organized [local_step][perturbation]
message GradientScalars {
  repeated LocalStepGradients local_steps = 1;
}
message LocalStepGradients {
  repeated double scalars = 1; // P float64 scalars
}

// Used to help a rejoining client rebuild its local model
message RebuildHistory {
  repeated RoundHistory rounds = 1;
}
message RoundHistory {
  int32 round_number        = 1;
  PerturbationSeeds seeds   = 2;
  GradientScalars average_gradients = 3; // server-averaged (1/N) across all clients
}
```

The DeComFL request/response pair carries three fields the older docs omit — see
[06 — DeComFL](06_decomfl.md#grpc-protocol-for-decomfl) for the full round protocol:

```proto
message GetDeComFLConfigResponse {
  int32              current_round   = 1;
  PerturbationSeeds  current_seeds   = 2;
  RebuildHistory     rebuild_history = 3;
  map<string,string> config          = 4;   // learning_rate, smoothing_param, num_local_steps,
                                            // num_perturbations, model_dim
  // --- v2 determinism contract ---
  string torch_version        = 5;   // advisory: the mobile RandnEngine is torch-version-independent
  string grad_estimate_method = 6;   // "forward" | "central"
  string golden_vector_sha256 = 7;   // RNG-parity fixture; empty => the client skips the check
}

message SubmitGradientScalarsRequest {
  string          client_id        = 1;
  string          run_id           = 2;
  int32           trained_on_round = 3;
  GradientScalars gradients        = 4;
  int64           num_examples     = 5;   // collected; DeComFL aggregation is UNWEIGHTED
  PerturbationSeeds perturbation_seeds = 6; // the client's echo of the server-issued seeds
}
```

> The server reconstructs `z` from **its own** `seed_history`, so `perturbation_seeds` is advisory
> here (observability / a future integrity cross-check) — `grpc_servicer` logs the echoed step count
> and never re-derives from it. The field exists for a FedAvg ZO-SGD variant in which the *client*
> generates the seeds.

---

## Protocol Version Negotiation

`grpc_servicer.SERVER_PROTOCOL_VERSION = 2` is the `fedlearn.v2` version this server speaks, and it
must equal the mobile client's `kProtocolVersion` (`mobile_client/.../FedLearnCoreModule.h`).

`RegisterClient` is permissive-then-strict: a client that sends `0` (field unset) is accepted, but a
client that sends a **set** version different from 2 is `REJECTED` with both versions in the
message. The Python `GrpcClient.register()` currently sends only `client_id` — `run_id`,
`protocol_version` and `enrollment_token` are left unset — so a Python client always takes the
permissive branch. The strict branch exists for the mobile/native clients that do set it.

---

## Serializer — Tensor ↔ Proto Conversion

`communication/serializer.py` is the bridge between PyTorch `OrderedDict[str, Tensor]` and protobuf messages.

### Serialization (PyTorch → Proto)

```python
def parameters_to_proto(
    parameters: OrderedDict[str, torch.Tensor],
    num_examples: int
) -> ModelParameters:
    tensors = {}
    for name, tensor in parameters.items():
        np_array = tensor.cpu().detach().numpy()
        tensors[name] = Tensor(
            data=np_array.tobytes(),     # raw bytes, no pickle
            dims=list(np_array.shape),
            dtype=str(np_array.dtype),
        )
    return ModelParameters(tensors=tensors, num_examples_trained=num_examples)
```

**Key design decision:** Raw `tobytes()` instead of pickle. This is intentional — pickle can execute arbitrary code, making it a security risk in a federated setting where a malicious client could send crafted payloads.

### Deserialization (Proto → PyTorch)

```python
# Whitelist of allowed dtypes — prevents dtype injection attacks
_SAFE_DTYPES = {
    'float16', 'float32', 'float64',
    'int8', 'int16', 'int32', 'int64',
    'uint8', 'bool', 'bfloat16',
}

def proto_to_parameters(proto: ModelParameters):
    parameters = OrderedDict()
    for name, tensor_proto in proto.tensors.items():
        # Security: reject unknown dtypes
        if tensor_proto.dtype not in _SAFE_DTYPES:
            raise ValueError(f"Unsafe dtype '{tensor_proto.dtype}'")

        np_array = np.frombuffer(tensor_proto.data, dtype=np.dtype(tensor_proto.dtype))

        # Security: validate shape vs. data length
        expected_size = math.prod(tensor_proto.dims)   # each dim must also be > 0
        if expected_size != len(np_array):
            raise ValueError("Shape mismatch")

        np_array = np_array.reshape(tensor_proto.dims).copy()

        # Security (SE-3): reject NaN/Inf. One malicious or buggy client could otherwise push
        # non-finite weights that propagate through the average and destroy the global model
        # for every honest client in the round.
        _reject_non_finite(name, np_array)

        parameters[name] = torch.tensor(np_array)
    return parameters, proto.num_examples_trained
```

### Header Validation in the safetensors Decoder

`load_safetensors()` runs on the untrusted gRPC receive path, so it validates the header against the
actual blob rather than trusting it. Every one of these raises rather than mis-reading:

| Check | What it stops |
|---|---|
| `len(blob) >= 8`, `8 + header_len <= len(blob)` | truncated / corrupt / legacy-pickle blob |
| header parses as a JSON **object** | garbage header |
| `dtype == "F32"` | a wrong dtype being read as F32 |
| every shape dim is an `int >= 0` | abusing numpy's `reshape` infer-dimension via a negative dim |
| `0 <= s <= e <= data_len` per tensor | out-of-range offsets slicing leniently and returning whatever bytes exist |
| `(e - s) == 4 * prod(shape)` | a byte count that disagrees with the declared shape |
| **running `Σ(e - s) <= data_len`** | the **decode-amplification guard**: each offset pair can be individually in range while many tensors point at the *same* bytes, so the total decoded size is unbounded while the wire stays small. Canonical safetensors is contiguous, so a running total past the data section can only be overlap/duplication |

---

## Chunked Streaming for Large Models

Streaming bypasses the protobuf `Tensor` message type entirely for large transfers, serialising the whole
state_dict into **one deterministic safetensors blob** and chunking the bytes.

### Why safetensors Instead of Proto (or torch.save) for Streaming?

The wire format used to be a `torch.save()` pickle blob. It is now the framework's own safetensors codec
(`communication/safetensors_codec.py`) — the change buys three things at once:

| Property | Why it matters |
|---|---|
| **No pickle** | `torch.save` blobs are pickles; a malicious client payload could execute code. safetensors is data-only. |
| **Cross-language** | `bytes = u64_le(header_len) ++ header_json_utf8 ++ raw_tensor_data`, header JSON compact (`separators=(",", ":")`), tensors emitted in insertion order. The libtorch-free mobile C++ core (`mobile_client/shared/src/Safetensors.cpp`, used by `ModelManager.cpp`) produces **byte-identical** output, and golden fixtures under `framework/tests/fixtures/decomfl_golden/` pin the contract. |
| **Memory** | Still one contiguous serialized buffer + chunked sends, rather than the ~3× amplification of building a giant proto message. |

The cost: the safetensors wire is **float32-only** and **fail-loud**. `state_dict_to_safetensors()`
rejects, rather than coerces, three things:

| Rejected | Why |
|---|---|
| any non-`float32` tensor | `save_safetensors` would otherwise cast int/bool to F32 and corrupt the model |
| a 0-dim (scalar) tensor | it round-trips with the wrong shape — the wire carries rank≥1 model params |
| a parameter literally named `__metadata__` | it collides with the safetensors metadata block and would be silently dropped |

> **Non-float32 buffers are excluded upstream rather than crashing the round.** Every BatchNorm
> module carries an `int64 num_batches_tracked`, which used to make a full-model federation of *any*
> BatchNorm model — i.e. every ResNet, the most common architecture in the FL literature — fail on
> the first `GetGlobalModel`. `estimators/params.federable_state(state)` is the shared filter that
> keeps only the float32 tensors, and `non_federable_names(state)` returns what it withheld so a run
> can log and audit the exclusion. It must be applied on **both** sides — two independent filters
> would drift, and that divergence has broken the frozen arm before. What is dropped is a batch
> *counter*, so averaging it was meaningless anyway and each client keeps its own; `running_mean` /
> `running_var` are float32 and **continue to be averaged** (excluding those too would be FedBN, a
> different algorithm, not a wire fix).

### Server-Side Streaming (GetGlobalModelStream)

```python
# grpc_servicer.py — GetGlobalModelStream
def GetGlobalModelStream(self, request, context):
    params, current_round, config = self.coordinator.get_global_model_for_client()

    # FR-8 (download half): the same libtorch-free safetensors wire the upload path
    # and the mobile C++ core use. F32-only and fail-loud.
    data_to_send = state_dict_to_safetensors(params, num_examples=0)
    download_codec = "safetensors"

    # Declare the sha256 of the FULL payload so receivers verify the reassembled blob,
    # format-agnostically, before any deserialization. Set on EVERY chunk: the mobile
    # C++ client reads it from the first, the Python client from the final one.
    payload_sha256 = hashlib.sha256(data_to_send).hexdigest()

    chunk_size = 50 * 1024 * 1024  # 50 MB per chunk
    total_size = len(data_to_send)
    num_chunks = (total_size + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_size)
        yield fedlearn_pb2.ModelChunk(
            chunk_index=i,
            total_chunks=num_chunks,
            chunk_data=data_to_send[start:end],
            is_final_chunk=(i == num_chunks - 1),
            current_round=current_round,
            config=config if i == 0 else {},   # config only in first chunk
            codec=download_codec,
            total_bytes=total_size,
            sha256=payload_sha256,
        )
```

> The mobile `FedLearnClient` **rejects any first chunk whose `codec` is not `"safetensors"`** — setting
> `codec` (with `total_bytes`) is what makes the FedAvg download decode instead of throw.

### Client-Side Streaming Reception (get_global_model)

```python
# grpc_client.py — get_global_model
def get_global_model(self):
    req = fedlearn_pb2.GetGlobalModelRequest(client_id=self.client_id)

    # Stream directly into BytesIO — avoids 3× memory from chunks.append() + b''.join().
    # For a 14 GB LLaMA-7B download that triple-allocation OOM'd a Jetson Orin.
    buffer = io.BytesIO()
    current_round, config, codec = 0, {}, ""
    hasher = hashlib.sha256()          # hash INCREMENTALLY — never a second copy of the payload
    declared_sha256 = ""

    for chunk in self.stub.GetGlobalModelStream(req, timeout=3600):
        if chunk.chunk_index == 0:
            current_round = chunk.current_round
            config = dict(chunk.config)
            codec = chunk.codec
        if chunk.sha256:
            declared_sha256 = chunk.sha256   # set on every chunk; take whichever carries it
        buffer.write(chunk.chunk_data)
        hasher.update(chunk.chunk_data)

    # Integrity first: verify the reassembled payload against the server-declared sha256
    # BEFORE deserializing anything. (A pre-integrity server declares none; verification
    # is then skipped with a debug log.)
    if declared_sha256 and hasher.hexdigest() != declared_sha256:
        buffer.close()
        raise ValueError("Global model download failed sha256 integrity check…")

    blob = buffer.getvalue()
    buffer.close()

    # FR-8 (download half) version gate: decode the safetensors wire — the current format —
    # but transparently fall back to a legacy torch.save pickle blob so a new client still
    # works against an OLD server during a staged rollout. The `codec` field is the primary
    # signal; a magic-byte sniff is the backstop when an old server sets no codec.
    is_pickle = len(blob) >= 2 and (blob[:2] == b"PK" or blob[0] == 0x80)
    if codec.endswith("safetensors") if codec else not is_pickle:
        params, _num_examples = chunks_to_parameters(blob, compressed=codec.startswith("lz4"))
    else:
        model_data = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        params = model_data['parameters']

    return params, current_round, config
```

> **Asymmetric legacy tolerance — worth internalising, and easy to get wrong in either direction.**
> The two decode paths genuinely differ:
>
> | path | direction | legacy `torch.save`/pickle blob |
> |---|---|---|
> | `GrpcClient.get_global_model` (`grpc_client.py`) | server → **client** download | **ACCEPTED** — falls back to `torch.load(..., weights_only=True)` |
> | `serializer.chunks_to_parameters` (upload receive **and** the client's safetensors branch) | client → **server** upload | **REJECTED loudly** |
>
> The rule behind the asymmetry: untrusted *client* bytes never get pickle tolerance; only the
> *server's own* bytes do, and only transitionally, so a new client keeps working against an older
> server during a staged rollout. `weights_only=True` prevents arbitrary pickle execution even on
> the tolerant path. The `codec` field is the primary signal; the magic-byte sniff is the backstop
> for an old server that sets no codec. Integrity (sha256) is verified **before** either branch,
> format-agnostically.
>
> **FR-27 — the rejection is gated behind a positive safetensors check, not a bare magic sniff.**
> `chunks_to_parameters` first calls `_looks_like_safetensors(data)`: an 8-byte little-endian u64
> header length that is non-zero, fits inside the blob, and is followed by `{`. Only if that fails
> does it apply the `PK` / `0x80` legacy sniff. This matters because a *valid* safetensors blob can
> begin with those bytes by arithmetic coincidence — a header of 128 or 384 bytes puts `0x80` in the
> first byte, and `header_len ≡ 19280 (mod 65536)` spells `PK`. The bare sniff false-rejected those
> well-formed payloads.

### Client Upload: Auto-Select Unary vs. Streaming

```python
# grpc_client.py — submit_update()
STREAMING_THRESHOLD_MB = 100
ALWAYS_STREAM_TRANSFORMERS = True

def submit_update(self, params, num_examples, round_number):
    total_params = sum(p.numel() for p in params.values())
    size_mb = (total_params * 4) / (1024 * 1024)

    # Detect transformer architecture by parameter naming conventions
    is_transformer = any(
        keyword in name.lower()
        for name in params.keys()
        for keyword in ['transformer', 'bert', 'gpt', 'opt', 'attention', 'encoder', 'decoder']
    )

    if (is_transformer and ALWAYS_STREAM_TRANSFORMERS) or size_mb > STREAMING_THRESHOLD_MB:
        return self._submit_update_stream(params, num_examples, round_number)
    else:
        return self._submit_update_unary(params, num_examples, round_number)
```

The streaming upload delegates serialization + chunking to `serializer.parameters_to_chunks`, which emits
plain dicts that the client wraps in `ModelUpdateChunk` messages:

```python
def _generate_model_chunks(self, params, num_examples, round_number, chunk_size=50*1024*1024):
    # parameters_to_chunks uses the safetensors wire format. compress=False: the gRPC
    # streaming path is uncompressed by design — see serializer.py.
    for chunk_dict in parameters_to_chunks(params, num_examples,
                                           chunk_size=chunk_size, compress=False):
        yield fedlearn_pb2.ModelUpdateChunk(
            client_id=self.client_id,
            trained_on_round=round_number,
            chunk_index=chunk_dict["chunk_index"],
            total_chunks=chunk_dict["total_chunks"],
            chunk_data=chunk_dict["chunk_data"],
            is_final_chunk=chunk_dict["is_final_chunk"],
            num_examples=num_examples,
        )
```

> **Size-gated at the call site, unconditional within the streaming path.** Both halves of that
> sentence are load-bearing, and simplifying it in either direction makes it wrong.
> `submit_update()` *chooses* streaming only when the model looks like a transformer (by parameter
> name keyword, with `ALWAYS_STREAM_TRANSFORMERS = True`) **or** exceeds
> `STREAMING_THRESHOLD_MB = 100`; otherwise the upload is a single unary call. But *once* streaming
> is chosen, chunking is unconditional — a small blob simply emits one chunk.

> **Chunk size — two different defaults, know which applies.** `_generate_model_chunks` passes an explicit
> `chunk_size=50 MB`, so **the streaming RPC path chunks at 50 MB**. The `FEDLEARN_CHUNK_SIZE_MB` env var
> (default **4 MB**) sets `serializer.CHUNK_SIZE`, which is only the fallback when a caller invokes
> `parameters_to_chunks` *without* passing `chunk_size`. Setting the env var does not by itself resize the
> gRPC upload chunks.

---

## Streamed-Upload Resource Caps (SE-18)

`SubmitModelUpdateStream` reassembles an attacker-controlled byte stream into a server-side buffer,
so the servicer bounds three axes. Defaults are generous enough for LLM-scale adapters and all three
are overridable by environment variable:

| Cap | Constant | Default | Env var | Abort code |
|---|---|---|---|---|
| Total payload bytes | `_DEFAULT_MAX_UPLOAD_BYTES` | 2 GiB | `FEDLEARN_MAX_UPLOAD_BYTES` | `RESOURCE_EXHAUSTED` |
| Chunk count | `_DEFAULT_MAX_UPLOAD_CHUNKS` | 100,000 | `FEDLEARN_MAX_UPLOAD_CHUNKS` | `RESOURCE_EXHAUSTED` |
| Active streaming wall-clock | `_DEFAULT_MAX_UPLOAD_SECONDS` | 600 s | `FEDLEARN_MAX_UPLOAD_SECONDS` (≤ 0 disables) | `DEADLINE_EXCEEDED` |

Three details make these actually hold:

- **Enforced before the write, not after.** The byte and chunk checks run *before* `buffer.write()`,
  so the buffer can never exceed the limit even if the client lies about (or omits) `total_bytes` /
  `total_chunks`, or never sends `is_final_chunk`.
- **`total_chunks` is untrusted.** It is used only for the advisory progress log, guarded against a
  malformed `<= 0` value so it cannot `ZeroDivisionError` and get remapped to `INTERNAL`.
  Correctness rides on `is_final_chunk` plus the caps.
- **A dedicated exception type.** `_StreamLimitExceeded` carries its own status code and is caught
  *before* the broad `except ValueError` / `except Exception` clauses, so the abort is not remapped
  to `INVALID_ARGUMENT` / `INTERNAL`.

The deadline bounds a **slow-drip** upload — it is checked on each chunk *arrival*. A client that
connects and then goes fully silent blocks on the next read and is bounded instead by gRPC's
`max_connection_age_ms` / keepalive settings from `server.py`.

**Status-code map for the upload paths** (`tests/test_servicer_status_codes.py` pins these):

| Condition | Code |
|---|---|
| Malformed / non-finite / wrong-shape / empty payload | `INVALID_ARGUMENT` |
| Byte or chunk cap exceeded | `RESOURCE_EXHAUSTED` |
| Upload deadline exceeded | `DEADLINE_EXCEEDED` |
| `client_id` doesn't match the identity bound to the connection token (SE-15) | `PERMISSION_DENIED` |
| Missing / invalid connection token (SE-1) | `UNAUTHENTICATED` |
| Valid token minted for a different run (FR-7) | `PERMISSION_DENIED` |
| Anything else | `INTERNAL` |

---

## LZ4 Compression (Optional)

Compression is **off by default** to preserve backward compatibility. Enable it by setting:

```bash
FEDLEARN_USE_COMPRESSION=1
```

LZ4 is used for its extreme speed (compression happens in milliseconds even for large tensors) at the cost of moderate compression ratios (~1.5–3× for typical model weights):

```python
# serializer.py
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

USE_COMPRESSION = LZ4_AVAILABLE and os.environ.get("FEDLEARN_USE_COMPRESSION", "0") == "1"

def parameters_to_chunks(params, num_examples, compress=None):
    ...
    if compress and LZ4_AVAILABLE:
        compressed = lz4.frame.compress(serialized, compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
        data_to_send = compressed
    else:
        data_to_send = serialized
```

> **Note:** `USE_COMPRESSION` is only the *default* for `parameters_to_chunks` / `chunks_to_parameters`
> when the caller passes no explicit `compress=` / `compressed=`. It is read at module import time, so set
> the env var before starting either process.
>
> **The gRPC streaming upload path ignores it**: `grpc_client._generate_model_chunks` passes
> `compress=False` explicitly — that path is uncompressed by design. And under the v2 framing, receivers
> no longer *infer* compression from their own env at all: the `compressed` flag (and the `codec` string,
> `"lz4+safetensors"` when set) travels on the wire, which is precisely the A3-C3 fix for the old
> both-sides-must-agree footgun described above.

---

## TLS Configuration

TLS is controlled entirely by environment variables. The framework supports both server-only TLS and mutual TLS (mTLS).

> **Plaintext is the default — and that is a real, standing caveat.** With `FEDLEARN_GRPC_USE_TLS` unset,
> the FL boundary is `insecure_channel` / `add_insecure_port`: model weights and updates cross the network
> unencrypted. For cross-network demos, tunnel it. What is *not* true is that encryption is unavailable —
> the mechanism below is implemented and opt-in, and deployed profiles are fail-closed against plaintext.

### Policy Layer — SE-2 Fail-Closed

`security/tls.py` separates *policy* from the *mechanism* in `server.py` / `grpc_client.py`. The backend sets
`FEDLEARN_REQUIRE_TLS=1` on deployed spawns once server certs are provisioned; enforcement is opt-in so
dev/test stay plaintext and nothing breaks before certs exist.

```python
# security/tls.py — check_server_tls_policy()
require = env.get("FEDLEARN_REQUIRE_TLS") == "1"
use     = env.get("FEDLEARN_GRPC_USE_TLS") == "1"

# Fail closed: never silently serve a deployed profile in plaintext.
if require and not use:
    raise TlsPolicyError("FEDLEARN_REQUIRE_TLS=1 but FEDLEARN_GRPC_USE_TLS is not enabled — "
                         "refusing to serve the FL boundary in plaintext on a deployed profile.")

# If TLS is on, the key + cert must actually be present.
if use:
    missing = [n for n in ("FEDLEARN_GRPC_SERVER_KEY", "FEDLEARN_GRPC_SERVER_CERT") if not env.get(n)]
    if missing:
        raise TlsPolicyError(f"FEDLEARN_GRPC_USE_TLS=1 but {', '.join(missing)} not set")
return use
```

### Server-Side TLS Setup

```python
# server.py — the bind decision is delegated to the policy above
use_tls = check_server_tls_policy()

if use_tls:
    with open(os.environ["FEDLEARN_GRPC_SERVER_KEY"], "rb") as f:
        server_key = f.read()
    with open(os.environ["FEDLEARN_GRPC_SERVER_CERT"], "rb") as f:
        server_cert = f.read()

    # Optional: root CA cert for mTLS
    root_cert = None
    if root_cert_path := os.environ.get("FEDLEARN_GRPC_ROOT_CERT"):
        with open(root_cert_path, "rb") as f:
            root_cert = f.read()

    require_client_auth = os.environ.get("FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH", "0") == "1"

    credentials = grpc.ssl_server_credentials(
        [(server_key, server_cert)],
        root_certificates=root_cert,
        require_client_auth=require_client_auth,
    )
    grpc_server.add_secure_port(server_address, credentials)
    logging.info("gRPC TLS enabled (require_client_auth=%s)", require_client_auth)
else:
    grpc_server.add_insecure_port(server_address)
    logging.warning("gRPC server running without TLS. Set FEDLEARN_GRPC_USE_TLS=1 for production.")
```

### Client-Side TLS Setup

```python
# grpc_client.py — _build_channel()
use_tls = os.environ.get("FEDLEARN_GRPC_USE_TLS", "0") == "1"

if not use_tls:
    return grpc.insecure_channel(server_address, options=grpc_options)

credentials = grpc.ssl_channel_credentials(
    root_certificates=_read(os.environ.get("FEDLEARN_GRPC_ROOT_CERT")),
    private_key=_read(os.environ.get("FEDLEARN_GRPC_CLIENT_KEY")),     # for mTLS
    certificate_chain=_read(os.environ.get("FEDLEARN_GRPC_CLIENT_CERT")), # for mTLS
)
return grpc.secure_channel(server_address, credentials, options=grpc_options)
```

### Environment Variable Reference

| Variable | Server | Client | Description |
|----------|--------|--------|-------------|
| `FEDLEARN_GRPC_USE_TLS` | ✓ | ✓ | `"1"` to enable TLS |
| `FEDLEARN_REQUIRE_TLS` | ✓ | — | `"1"` to **require** TLS — set by the backend on deployed profiles; refuses to serve plaintext (SE-2) |
| `FEDLEARN_GRPC_SERVER_KEY` | ✓ | — | Path to server private key (PEM) |
| `FEDLEARN_GRPC_SERVER_CERT` | ✓ | — | Path to server certificate (PEM) |
| `FEDLEARN_GRPC_ROOT_CERT` | optional | optional | Path to CA root cert (for mTLS verification) |
| `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH` | optional | — | `"1"` to require client certificates |
| `FEDLEARN_GRPC_CLIENT_KEY` | — | optional | Client private key for mTLS |
| `FEDLEARN_GRPC_CLIENT_CERT` | — | optional | Client certificate for mTLS |

> **Distinct from TLS: the connection token (SE-1/SE-14).** `FEDLEARN_CONNECTION_TOKEN` authenticates the
> *client* to the FL server and is orthogonal to transport encryption. `security/client_interceptor.py`
> (`maybe_wrap_channel`) attaches it to every call when set, and is a no-op when unset — so dev and
> unauthenticated servers are unaffected. Fetch a token from `GET /api/client/projects/{id}/connection`;
> the desktop launcher sets it automatically.

---

## Retry Logic and Backoff

All unary client calls go through `_retry_unary()`:

```python
_RETRYABLE_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}

def _retry_unary(fn, *, op_name, max_attempts=4, base_delay=0.5, max_delay=8.0):
    attempt = 0
    while True:
        try:
            return fn()
        except grpc.RpcError as e:
            attempt += 1
            if e.code() not in _RETRYABLE_CODES or attempt >= max_attempts:
                raise
            # Exponential backoff: 0.5s, 1s, 2s, 4s (capped at 8s)
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            log.warning("%s retry attempt=%d in %.1fs", op_name, attempt, delay)
            time.sleep(delay)
```

**Retryable codes** are those that represent transient network conditions:
- `UNAVAILABLE` — server temporarily down or restarting
- `DEADLINE_EXCEEDED` — request timed out (will retry with fresh timeout)
- `RESOURCE_EXHAUSTED` — server under load
- `ABORTED` — concurrent modification detected

**Non-retryable codes** (e.g., `FAILED_PRECONDITION`, `NOT_FOUND`, `INTERNAL`) propagate immediately as they indicate logical errors.

The outer `start_client()` loop has its own retry at 10-second intervals for `UNAVAILABLE`, allowing clients to reconnect if the server briefly goes down between rounds.

---

## Keepalive Tuning

Both server and client are tuned to survive AWS NLB/ALB idle timeout (default 350 seconds) and NAT gateway idle timeout:

```python
# Server options (server.py)
options=[
    ('grpc.keepalive_time_ms', 120000),             # send keepalive every 2 min
    ('grpc.keepalive_timeout_ms', 60000),            # wait 60s for keepalive ACK
    ('grpc.keepalive_permit_without_calls', True),   # keepalive even with no active RPCs
    ('grpc.http2.max_pings_without_data', 0),        # allow unlimited pings
    ('grpc.http2.min_time_between_pings_ms', 120000),
    ('grpc.http2.min_ping_interval_without_data_ms', 120000),
    ('grpc.http2.bdp_probe', False),                 # disable BDP probing (saves RTTs)
    ('grpc.http2.max_ping_strikes', 0),              # never disconnect for too many pings
]

# Client options (grpc_client.py)
grpc_options=[
    ('grpc.keepalive_time_ms', 60000),   # more aggressive on client side
    ('grpc.keepalive_timeout_ms', 20000),
    ('grpc.keepalive_permit_without_calls', 1),
    ('grpc.http2.max_pings_without_data', 0),
]
```

---

## Connection Limits and Message Sizes

```python
# Both server and client must agree on message sizes
('grpc.max_send_message_length',    1024 * 1024 * 1024),  # 1 GB
('grpc.max_receive_message_length', 1024 * 1024 * 1024),  # 1 GB

# Server connection age limits
('grpc.max_connection_idle_ms',       7200000),   # 2 hours
('grpc.max_connection_age_ms',       14400000),   # 4 hours
('grpc.max_connection_age_grace_ms',   600000),   # 10 min grace period
```

The 1 GB message limit applies per gRPC message, not per streaming chunk. Since the streaming API uses 50 MB chunks, you could in theory serve a 50+ GB model without hitting the limit.

The server creates `(max_clients * 2) + 10` thread-pool workers:

```python
max_expected_clients = int(os.environ.get('MAX_CLIENTS', 50))
optimal_workers = (max_expected_clients * 2) + 10  # 2 threads per client + overhead
grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=optimal_workers), ...)
```

---

## Dual-Channel Heartbeat Architecture

A critical design choice: the `GrpcClient` opens **two separate gRPC channels** to the same server address.

```python
class GrpcClient:
    def __init__(self, client_id, server_address):
        # Primary channel — used for all heavy operations:
        # RegisterClient, GetGlobalModelStream, SubmitModelUpdateStream
        # maybe_wrap_channel attaches FEDLEARN_CONNECTION_TOKEN if present (SE-1); no-op when unset.
        self.channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
        self.stub = FederatedLearningServiceStub(self.channel)

        # Dedicated heartbeat channel — never blocked by ongoing transfers
        self.heartbeat_channel = maybe_wrap_channel(_build_channel(server_address, grpc_options))
        self.heartbeat_stub = FederatedLearningServiceStub(self.heartbeat_channel)
```

**Why this matters:** During a multi-GB model download or upload, the primary gRPC channel's HTTP/2 connection is saturated with data frames. If heartbeats shared this channel, they would queue behind the data and could appear to time out even though the client is healthy. The server would then incorrectly mark the client as dead.

The heartbeat thread runs on a 5-second interval:

```python
def _heartbeat_loop(self):
    while self.heartbeat_active:
        try:
            self.send_heartbeat()
        except Exception:
            log.debug("[%s] Heartbeat loop exception", self.client_id, exc_info=True)
        # FR-10: interruptible — returns immediately once stop_heartbeat() sets the event,
        # instead of sleeping out the full interval before noticing it should stop.
        self._heartbeat_stop.wait(self.heartbeat_interval)
```

It is launched as a daemon thread (exits automatically when the main process exits) immediately after client registration.

### The Heartbeat Channel Is Also the Stop Signal (FR-10)

The dual-channel split does more than keep liveness flowing. Because `fit()` blocks the *training* stub for
the whole round, the heartbeat stub is the only channel that can reach a busy client — so it carries the
server's stop request back:

```python
res = self.heartbeat_stub.Heartbeat(req, timeout=30.0)
if res.should_stop:
    # Latch it. The training thread's fit loop polls should_stop_training() between local
    # steps and aborts the round. Previously this response was discarded by _heartbeat_loop,
    # which made the server's stop request a silent no-op.
    self._stop_training.set()
    return False
return res.acknowledged
```

This is the cross-stub signal that lets the parallel heartbeat stub halt a `fit()` that is blocking the
training stub. When touching client lifecycle code, preserve both stubs *and* this latch.

---

## Regenerating Generated Code

> **Do not hand-edit `framework/src/fedlearn/communication/protos/fedlearn.proto`.** It is a *mirror*.
> The canonical contract lives at `proto/fedlearn/v2/fedlearn.proto` and is governed by `buf`. A stray edit
> to the mirror fails CI with a `cp` fix — see `proto/README.md`.

The flow is: **edit the canonical → run buf → sync the mirrors → regenerate the framework's stubs.**

```bash
# 1. Edit proto/fedlearn/v2/fedlearn.proto (the single source of truth), then:
cd proto
buf lint                                   # STANDARD ruleset (see buf.yaml for documented excepts)
buf breaking --against '.git#branch=main'  # fail on a breaking change to the wire contract
buf generate                               # writes gen/python, gen/java, gen/ts, gen/cpp

# 2. Sync the in-tree mirrors back out from canonical
cp proto/fedlearn/v2/fedlearn.proto framework/src/fedlearn/communication/protos/fedlearn.proto
cp proto/fedlearn/v2/fedlearn.proto mobile_client/proto/fedlearn/v2/fedlearn.proto
cp proto/fedlearn/fot/v1/fot.proto  framework/src/fedlearn/communication/protos/fot.proto

# 3. Verify no mirror has drifted (this is the CI gate)
./scripts/check_proto_mirror.sh
```

CI enforces all of this as the `proto.yml` workflow: `buf lint`, `buf breaking` against `main`, a
`buf generate` **freshness** check (regeneration must be a no-op, so committed stubs cannot silently rot),
and the mirror check.

The framework's generated files are **committed to the repository** so that users don't need codegen tooling installed at runtime.

---

## Common Errors and Debugging

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `StatusCode.UNAVAILABLE` | Server not running / wrong address | Verify `--server_address` and server logs |
| `StatusCode.RESOURCE_EXHAUSTED` | Message exceeds 1 GB limit (unary) | Switch to streaming; model should auto-detect |
| `StatusCode.DEADLINE_EXCEEDED` | Model download timed out | Increase `timeout=3600` in stub calls; check network |
| `ValueError: Unsafe dtype` | Client sent malicious/malformed proto | Security rejection; investigate client code |
| `ValueError: Shape mismatch` | Corrupted proto payload | Retry; if persistent, check serializer on sender side |
| `ValueError: … contains non-finite values` | A client's update carries NaN/Inf | SE-3 poisoning rejection — a real reject, not a bug. Investigate that client's training (diverged loss / bad LR) |
| `ValueError: Received a legacy pickle/zip blob` | Sender still on the old `torch.save` wire | Update the client; only safetensors is accepted on the receive path |
| `ValueError: Tensor '…' has dtype …; only float32 is supported` | Non-float32 tensor in the state_dict | The safetensors wire is F32-only and fails loud rather than silently casting — cast to float32 before training |
| `ValueError: safetensors: …` (offsets/shape/dtype) | Malformed or malicious blob | Header validation (FR-8) rejecting it before any bytes are mis-read |
| `ValueError: safetensors: total tensor bytes … exceed data section …` | Overlapping/duplicated `data_offsets` | The decode-amplification guard; a canonical blob is contiguous |
| `ValueError: Tensor '…' is 0-dim (scalar)` | A scalar in the state_dict | The wire carries rank≥1 params — reshape or exclude it |
| `ValueError: Parameter name '__metadata__' is reserved` | A param literally named `__metadata__` | It collides with the safetensors metadata block — rename it |
| Download `sha256` integrity failure | Payload corrupted in transit | The client refuses to deserialize a payload that doesn't match the server-declared hash; retry |
| `RESOURCE_EXHAUSTED: streamed upload exceeded the …-byte/chunk cap` | SE-18 upload cap | Raise `FEDLEARN_MAX_UPLOAD_BYTES` / `_CHUNKS` if the model is genuinely that large |
| `DEADLINE_EXCEEDED: streamed upload exceeded the …s deadline` | Slow-drip upload / very slow link | Raise `FEDLEARN_MAX_UPLOAD_SECONDS` (or set ≤ 0 to disable the guard) |
| `UNAUTHENTICATED: connection token rejected: missing x-connection-token metadata` | Server has SE-1 auth on, client has no token | Set `FEDLEARN_CONNECTION_TOKEN` (fetch from `GET /api/client/projects/{id}/connection`) |
| `PERMISSION_DENIED: token run '…' != server run '…'` | Token minted for another run (FR-7) | Re-enroll against this run |
| `PERMISSION_DENIED: client_id does not match the identity bound to this connection token` | One token replayed under several `client_id`s (SE-15) | One token = one `client_id`; this is the anti-Sybil binding, not a bug |
| `REJECTED … Protocol version mismatch` | Client sent a set `protocol_version` ≠ 2 | Rebuild the client against the current `fedlearn.v2` contract |
| Heartbeat timeout on server | Client dead / network split | Normal — server logs will show stale clients; reduce `heartbeat_timeout` from 300s if desired |

Enable verbose gRPC tracing with:
```bash
GRPC_VERBOSITY=debug GRPC_TRACE=all python run_server.py 2>&1 | head -100
```
