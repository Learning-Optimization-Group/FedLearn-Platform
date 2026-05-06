# 02 — gRPC Communication Layer

## Table of Contents
- [Overview](#overview)
- [The Protocol Buffer Contract](#the-protocol-buffer-contract)
- [Service Definition — All 9 RPCs](#service-definition--all-9-rpcs)
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

1. **Support arbitrarily large models** — LLMs can exceed 10 GB; plain unary gRPC fails above ~4 MB by default. The framework auto-detects model size and switches to bidirectional chunked streaming.
2. **Survive flaky networks** — exponential backoff retries on all transient gRPC errors.
3. **Heartbeat without blocking training** — a second dedicated gRPC channel is used exclusively for heartbeats so they never queue behind large model uploads.
4. **Optional TLS** — controlled entirely by environment variables; zero code changes needed to flip between insecure and mTLS.

---

## The Protocol Buffer Contract

**Source of truth:** `framework/src/fedlearn/communication/protos/fedlearn.proto`

The `.proto` file is compiled to Python stubs stored in `communication/generated/`:
- `fedlearn_pb2.py` — message classes
- `fedlearn_pb2_grpc.py` — stub and servicer base class

```proto
syntax = "proto3";
package fedlearn.v1;

service FederatedLearningService {
  // Client registration
  rpc RegisterClient(RegisterClientRequest) returns (RegisterClientResponse);

  // Model distribution (server → clients)
  rpc GetGlobalModel(GetGlobalModelRequest)       returns (GetGlobalModelResponse);
  rpc GetGlobalModelStream(GetGlobalModelRequest) returns (stream ModelChunk);

  // Model update collection (clients → server)
  rpc SubmitModelUpdate(SubmitModelUpdateRequest)          returns (SubmitModelUpdateResponse);
  rpc SubmitModelUpdateStream(stream ModelUpdateChunk)     returns (SubmitModelUpdateResponse);

  // Status and liveness
  rpc GetServerStatus(GetServerStatusRequest) returns (GetServerStatusResponse);
  rpc Heartbeat(HeartbeatRequest)             returns (HeartbeatResponse);

  // DeComFL-specific
  rpc GetDeComFLConfig(GetDeComFLConfigRequest)          returns (GetDeComFLConfigResponse);
  rpc SubmitGradientScalars(SubmitGradientScalarsRequest) returns (SubmitGradientScalarsResponse);
}
```

---

## Service Definition — All 9 RPCs

### Standard FL RPCs

| RPC | Direction | Pattern | Purpose |
|-----|-----------|---------|---------|
| `RegisterClient` | client → server | Unary | Registers a client ID; returns ACCEPTED/REJECTED |
| `GetGlobalModel` | server → client | Unary | Download model parameters (small models only) |
| `GetGlobalModelStream` | server → client | **Server streaming** | Download model in 50 MB chunks |
| `SubmitModelUpdate` | client → server | Unary | Upload updated parameters (small models) |
| `SubmitModelUpdateStream` | client → server | **Client streaming** | Upload model in 50 MB chunks |
| `GetServerStatus` | client → server | Unary | Poll current round number and update count |
| `Heartbeat` | client → server | Unary | Report liveness + training progress |

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
// A single named tensor
message Tensor {
  bytes  data = 1;       // raw numpy bytes (e.g., float32 little-endian)
  repeated int64 dims = 2; // shape, e.g., [256, 512]
  string dtype = 3;       // e.g., "float32" — validated against whitelist
}

// A full model state_dict
message ModelParameters {
  map<string, Tensor> tensors = 1;       // key = parameter name
  int64 num_examples_trained = 2;        // for weighted aggregation
}
```

### Streaming Chunk Messages

```proto
// Server → Client: one chunk of a model download
message ModelChunk {
  int32  chunk_index   = 1;
  int32  total_chunks  = 2;
  bytes  chunk_data    = 3;   // raw bytes of a torch.save() buffer
  bool   is_final_chunk = 4;
  int32  current_round = 5;
  map<string, string> config = 6;  // only sent with chunk_index == 0
}

// Client → Server: one chunk of a model upload
message ModelUpdateChunk {
  string client_id       = 1;
  int32  trained_on_round = 2;
  int32  chunk_index     = 3;
  int32  total_chunks    = 4;
  bytes  chunk_data      = 5;
  bool   is_final_chunk  = 6;
  int64  num_examples    = 7;
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
        expected_size = math.prod(tensor_proto.dims)
        if expected_size != len(np_array):
            raise ValueError("Shape mismatch")

        parameters[name] = torch.tensor(np_array.reshape(tensor_proto.dims).copy())
    return parameters, proto.num_examples_trained
```

---

## Chunked Streaming for Large Models

The framework uses `torch.save()` / `torch.load()` for streaming, bypassing the protobuf `Tensor` message type entirely for large transfers. This has a significant memory advantage:

### Why torch.save() Instead of Proto for Streaming?

| Approach | Memory for a 10 GB model |
|---------|--------------------------|
| Proto message (all-at-once) | ~30 GB peak (3× amplification from alloc + copy) |
| `torch.save()` to `BytesIO` + chunked | ~12 GB (1.2× — only one extra copy) |
| `torch.save()` + `memoryview` for send | ~10 GB (near zero-copy send) |

### Server-Side Streaming (GetGlobalModelStream)

```python
# grpc_servicer.py — GetGlobalModelStream
def GetGlobalModelStream(self, request, context):
    params, current_round, config = self.coordinator.get_global_model_for_client()

    # Serialize with torch.save — much more memory-efficient than proto for large models
    buffer = io.BytesIO()
    torch.save({'parameters': params, 'num_examples': 0}, buffer)
    data_to_send = buffer.getvalue()
    buffer.close()

    chunk_size = 50 * 1024 * 1024  # 50 MB per chunk
    num_chunks = (len(data_to_send) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(data_to_send))
        yield fedlearn_pb2.ModelChunk(
            chunk_index=i,
            total_chunks=num_chunks,
            chunk_data=data_to_send[start:end],
            is_final_chunk=(i == num_chunks - 1),
            current_round=current_round,
            config=config if i == 0 else {}   # config only in first chunk
        )
```

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

    buffer.seek(0)
    model_data = torch.load(buffer, map_location='cpu', weights_only=True)
    buffer.close()

    return model_data['parameters'], current_round, config
```

> **`weights_only=True`** — This flag, added in PyTorch 2.0, prevents arbitrary pickle execution during `torch.load`. Always required.

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

The streaming upload uses `memoryview` to achieve near-zero-copy chunking:

```python
def _generate_model_chunks(self, params, num_examples, round_number, chunk_size=50*1024*1024):
    buffer = io.BytesIO()
    torch.save(params, buffer)

    view = memoryview(buffer.getbuffer())  # zero-copy view of the buffer
    total_chunks = (len(view) + chunk_size - 1) // chunk_size

    try:
        for i in range(0, len(view), chunk_size):
            chunk_index = i // chunk_size
            yield fedlearn_pb2.ModelUpdateChunk(
                client_id=self.client_id,
                trained_on_round=round_number,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                chunk_data=view[i:i + chunk_size].tobytes(),  # only copies this slice
                is_final_chunk=(chunk_index == total_chunks - 1),
                num_examples=num_examples,
            )
    finally:
        view.release()  # always release the memoryview
```

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

> **Note:** Both server and client must agree on whether compression is active. If only one side compresses, deserialization will fail. The `USE_COMPRESSION` flag is read at module import time, so set the env var before starting either process.

---

## TLS Configuration

TLS is controlled entirely by environment variables. The framework supports both server-only TLS and mutual TLS (mTLS).

### Server-Side TLS Setup

```python
# server.py
use_tls = os.environ.get("FEDLEARN_GRPC_USE_TLS", "0") == "1"

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
else:
    grpc_server.add_insecure_port(server_address)
    logging.warning("Running without TLS. Set FEDLEARN_GRPC_USE_TLS=1 for production.")
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
| `FEDLEARN_GRPC_SERVER_KEY` | ✓ | — | Path to server private key (PEM) |
| `FEDLEARN_GRPC_SERVER_CERT` | ✓ | — | Path to server certificate (PEM) |
| `FEDLEARN_GRPC_ROOT_CERT` | optional | optional | Path to CA root cert (for mTLS verification) |
| `FEDLEARN_GRPC_REQUIRE_CLIENT_AUTH` | optional | — | `"1"` to require client certificates |
| `FEDLEARN_GRPC_CLIENT_KEY` | — | optional | Client private key for mTLS |
| `FEDLEARN_GRPC_CLIENT_CERT` | — | optional | Client certificate for mTLS |

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
        self.channel = _build_channel(server_address, grpc_options)
        self.stub = FederatedLearningServiceStub(self.channel)

        # Dedicated heartbeat channel — never blocked by ongoing transfers
        self.heartbeat_channel = _build_channel(server_address, grpc_options)
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
            log.debug("Heartbeat loop exception", exc_info=True)
        time.sleep(self.heartbeat_interval)
```

It is launched as a daemon thread (exits automatically when the main process exits) immediately after client registration.

---

## Regenerating Generated Code

If you modify `fedlearn.proto`, regenerate the Python stubs:

```bash
# From the framework/ directory
pip install grpcio-tools

python -m grpc_tools.protoc \
    -I src/fedlearn/communication/protos \
    --python_out=src/fedlearn/communication/generated \
    --grpc_python_out=src/fedlearn/communication/generated \
    src/fedlearn/communication/protos/fedlearn.proto

# Fix import path in generated file (protoc generates absolute imports)
sed -i 's/import fedlearn_pb2/from . import fedlearn_pb2/' \
    src/fedlearn/communication/generated/fedlearn_pb2_grpc.py
```

The generated files are **committed to the repository** so that users don't need `grpcio-tools` installed at runtime.

---

## Common Errors and Debugging

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `StatusCode.UNAVAILABLE` | Server not running / wrong address | Verify `--server_address` and server logs |
| `StatusCode.RESOURCE_EXHAUSTED` | Message exceeds 1 GB limit (unary) | Switch to streaming; model should auto-detect |
| `StatusCode.DEADLINE_EXCEEDED` | Model download timed out | Increase `timeout=3600` in stub calls; check network |
| `ValueError: Unsafe dtype` | Client sent malicious/malformed proto | Security rejection; investigate client code |
| `ValueError: Shape mismatch` | Corrupted proto payload | Retry; if persistent, check serializer on sender side |
| `torch.load` fails with `weights_only=True` | Payload contains non-tensor objects | Do not pickle non-tensor data into model state dicts |
| Heartbeat timeout on server | Client dead / network split | Normal — server logs will show stale clients; reduce `heartbeat_timeout` from 300s if desired |

Enable verbose gRPC tracing with:
```bash
GRPC_VERBOSITY=debug GRPC_TRACE=all python run_server.py 2>&1 | head -100
```
