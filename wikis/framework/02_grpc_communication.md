# 02 — gRPC Communication Layer

## Table of Contents
- [Overview](#overview)
- [The Protocol Buffer Contract](#the-protocol-buffer-contract)
- [Service Definition — All 10 RPCs](#service-definition--all-10-rpcs)
- [Message Types In Depth](#message-types-in-depth)
- [Serializer — Tensor ↔ Proto Conversion](#serializer--tensor--proto-conversion)
- [Chunked Streaming for Large Models](#chunked-streaming-for-large-models)
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

1. **Support arbitrarily large models** — LLMs can exceed 10 GB, and a single unary message would have to be buffered whole. The client auto-detects model size (and transformer architecture) and switches from a unary call to a chunked *streaming* RPC — server-streaming for downloads, client-streaming for uploads.
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
| `RegisterClient` | client → server | Unary | Registers a client ID; returns ACCEPTED/REJECTED |
| `GetGlobalModel` | server → client | Unary | Download model parameters (small models only) |
| `GetGlobalModelStream` | server → client | **Server streaming** | Download model in 50 MB chunks |
| `SubmitModelUpdate` | client → server | Unary | Upload updated parameters (small models) |
| `SubmitModelUpdateStream` | client → server | **Client streaming** | Upload model in 50 MB chunks |
| `GetServerStatus` | client → server | Unary | Poll current round number and update count |
| `Heartbeat` | client → server | Unary | Report liveness + training progress; carries the server's `should_stop` back (FR-10) |
| `ReportClientMetrics` | client → server | Unary | Client-side telemetry reporting |

### DeComFL RPCs

| RPC | Direction | Pattern | Purpose |
|-----|-----------|---------|---------|
| `GetDeComFLConfig` | client → server | Unary | Download random seeds + missed-round history |
| `SubmitGradientScalars` | client → server | Unary | Upload O(K×P) scalars instead of full model |

> **Why two separate model-download RPCs?** `GetGlobalModel` exists as a fast path for small CNN models where a single unary call is sufficient. For any Transformer-class model, the client always calls `GetGlobalModelStream`. The server-side servicer implements both.

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
  int64  total_bytes = 11;   // server bounds-checks cumulative against max_payload_bytes (H5)
  string sha256      = 12;   // verified on reassembly
}
```

### DeComFL Messages

```proto
// Seeds organized [local_step][perturbation]
message PerturbationSeeds {
  repeated LocalStepSeeds local_steps = 1;
}
message LocalStepSeeds {
  repeated int32 seeds = 1;   // P int32 seeds for this step
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
  GradientScalars average_gradients = 3; // server-averaged across all clients
}
```

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
| **Cross-language** | `bytes = u64_le(header_len) ++ header_json_utf8 ++ raw_tensor_data`. The libtorch-free mobile C++ core (`mobile_client/shared/src/ModelManager.cpp`) produces **byte-identical** output, and a golden fixture pins the contract. |
| **Memory** | Still one contiguous serialized buffer + chunked sends, rather than the ~3× amplification of building a giant proto message. |

The cost: the safetensors wire is **float32-only** and **fail-loud**. `state_dict_to_safetensors()` raises on
any non-float32 tensor rather than silently coercing it to F32 and corrupting the model — cast to float32
before training.

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

    # Stream directly into BytesIO — avoids 3× memory from chunks.append() + b''.join()
    buffer = io.BytesIO()
    current_round = 0

    for chunk in self.stub.GetGlobalModelStream(req, timeout=3600):
        if chunk.chunk_index == 0:
            current_round = chunk.current_round
            config = dict(chunk.config)
        buffer.write(chunk.chunk_data)

    # Integrity first: verify the reassembled payload against the server-declared sha256
    # BEFORE deserializing anything. (A pre-integrity server declares none; verification
    # is then skipped with a debug log.)
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

> **Asymmetric legacy tolerance — worth internalising.** The *download* path still **accepts** a legacy
> torch.save/pickle blob (guarded by `weights_only=True`, which prevents arbitrary pickle execution) purely
> for staged-rollout compatibility with an old server. The *upload/receive* path does **not**: `chunks_to_parameters`
> sniffs for a zip (`PK\x03\x04`) or pickle (`0x80`) prefix and **rejects it loudly**. Untrusted client bytes
> never get pickle tolerance; only the server's own bytes do, and only transitionally.

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

> **Chunk size — two different defaults, know which applies.** `_generate_model_chunks` passes an explicit
> `chunk_size=50 MB`, so **the streaming RPC path chunks at 50 MB**. The `FEDLEARN_CHUNK_SIZE_MB` env var
> (default **4 MB**) sets `serializer.CHUNK_SIZE`, which is only the fallback when a caller invokes
> `parameters_to_chunks` *without* passing `chunk_size`. Setting the env var does not by itself resize the
> gRPC upload chunks.

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
| Download `sha256` integrity failure | Payload corrupted in transit | The client refuses to deserialize a payload that doesn't match the server-declared hash; retry |
| Heartbeat timeout on server | Client dead / network split | Normal — server logs will show stale clients; reduce `heartbeat_timeout` from 300s if desired |

Enable verbose gRPC tracing with:
```bash
GRPC_VERBOSITY=debug GRPC_TRACE=all python run_server.py 2>&1 | head -100
```
