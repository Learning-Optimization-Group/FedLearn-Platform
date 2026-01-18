# FedLearn Framework - Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Diagram](#component-diagram)
4. [Data Flow](#data-flow)
5. [Communication Protocol](#communication-protocol)
6. [Concurrency Model](#concurrency-model)
7. [Design Patterns](#design-patterns)
8. [Scalability](#scalability)

---

## Overview

FedLearn is a distributed federated learning framework built on gRPC that enables privacy-preserving machine learning across multiple clients without centralizing data. The framework supports both small models (CNNs) and large models (LLMs) through adaptive streaming mechanisms.

### Key Features
- **Distributed Training**: Multiple clients train independently on local data
- **Model Agnostic**: Works with any PyTorch model
- **Adaptive Streaming**: Automatically switches between unary and streaming based on model size
- **Heartbeat Monitoring**: Tracks client health in real-time
- **Strategy Pattern**: Pluggable aggregation strategies (FedAvg, custom)
- **Large Model Support**: Handles models up to several GB through chunked transfer

### Design Principles
1. **Separation of Concerns**: Clear boundaries between client, server, and communication layers
2. **Extensibility**: Easy to add new strategies, clients, and protocols
3. **Robustness**: Handles network failures, client dropouts, and stragglers
4. **Efficiency**: Minimizes communication overhead and memory usage

---

## System Architecture

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         FEDERATED LEARNING                         │
│                          ORCHESTRATION                             │
└───────────────────────────────────────────────────────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
                ▼                  ▼                  ▼
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
        │   CLIENT 1   │   │   CLIENT 2   │   │   CLIENT N   │
        │              │   │              │   │              │
        │  Local Data  │   │  Local Data  │   │  Local Data  │
        │  Local Model │   │  Local Model │   │  Local Model │
        └──────────────┘   └──────────────┘   └──────────────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │ gRPC
                                   ▼
                         ┌─────────────────┐
                         │  SERVER/        │
                         │  COORDINATOR    │
                         │                 │
                         │  Global Model   │
                         │  Aggregation    │
                         │  Evaluation     │
                         └─────────────────┘
```

### Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION TIER                            │
│  (Client Application Layer)                                      │
├─────────────────────────────────────────────────────────────────┤
│  • Client Implementation (fit(), get_parameters())              │
│  • Training Loop                                                 │
│  • Data Loading                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COMMUNICATION TIER                           │
│  (gRPC Layer)                                                    │
├─────────────────────────────────────────────────────────────────┤
│  • GrpcClient / GrpcServicer                                    │
│  • Protocol Buffers (Serialization)                             │
│  • Streaming (Chunked Transfer)                                  │
│  • Heartbeat (Keep-Alive)                                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BUSINESS LOGIC TIER                          │
│  (Server Coordination & Strategy)                                │
├─────────────────────────────────────────────────────────────────┤
│  • FLCoordinator (Round Management)                             │
│  • Strategy (Aggregation Logic)                                  │
│  • Evaluation                                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Diagram

### Detailed Component Breakdown

```
┌──────────────────────────── SERVER SIDE ────────────────────────────┐
│                                                                      │
│  ┌──────────────────┐                                               │
│  │   server.py      │                                               │
│  │   ──────────     │                                               │
│  │  start_server()  │                                               │
│  │  • Creates       │                                               │
│  │    coordinator   │                                               │
│  │  • Starts gRPC   │                                               │
│  │  • Runs training │                                               │
│  │    loop          │                                               │
│  └────────┬─────────┘                                               │
│           │ creates                                                 │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────┐                  │
│  │       coordinator.py (FLCoordinator)         │                  │
│  │       ────────────────────────────           │                  │
│  │  STATE MANAGEMENT                            │                  │
│  │  • _global_model_params (OrderedDict)        │                  │
│  │  • _client_updates_received (List)           │                  │
│  │  • _registered_clients (Set)                 │                  │
│  │  • current_round (int)                       │                  │
│  │  • client_heartbeats (Dict)                  │                  │
│  │                                               │                  │
│  │  SYNCHRONIZATION                             │                  │
│  │  • _lock (threading.Lock)                    │                  │
│  │  • _round_complete_event (threading.Event)   │                  │
│  │  • heartbeat_lock (Lock)                     │                  │
│  │                                               │                  │
│  │  METHODS                                     │                  │
│  │  • start_round()                             │                  │
│  │  • wait_for_round_to_complete()              │                  │
│  │  • submit_client_update()                    │                  │
│  │  • _trigger_aggregation_and_evaluation()     │                  │
│  │  • update_client_heartbeat()                 │                  │
│  │  • is_client_alive()                         │                  │
│  └──────────────┬───────────────────────────────┘                  │
│                 │ uses                                              │
│                 ▼                                                   │
│  ┌────────────────────────────────┐                                │
│  │   strategy.py (Strategy)       │                                │
│  │   ────────────────────          │                                │
│  │  • initialize_parameters()     │                                │
│  │  • aggregate_fit()             │                                │
│  │  • evaluate()                  │                                │
│  │                                │                                │
│  │  ┌─────────────────────────┐  │                                │
│  │  │   FedAvg                │  │                                │
│  │  │   • FedAvgAggregator    │  │                                │
│  │  │   • Weighted averaging  │  │                                │
│  │  └─────────────────────────┘  │                                │
│  └────────────────────────────────┘                                │
│                                                                      │
│  ┌──────────────────────────────────────────┐                      │
│  │  grpc_servicer.py (RPC Handlers)         │                      │
│  │  ────────────────────────────             │                      │
│  │  • RegisterClient()                      │                      │
│  │  • GetGlobalModelStream()                │                      │
│  │  • SubmitModelUpdateStream()             │                      │
│  │  • Heartbeat()                           │                      │
│  │  • GetServerStatus()                     │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── CLIENT SIDE ────────────────────────────┐
│                                                                      │
│  ┌──────────────────────────────┐                                   │
│  │   client.py                  │                                   │
│  │   ───────────                │                                   │
│  │  Client (ABC)                │                                   │
│  │  • get_parameters()          │                                   │
│  │  • fit()                     │                                   │
│  │                              │                                   │
│  │  start_client()              │                                   │
│  │  1. Register                 │                                   │
│  │  2. Start heartbeat          │                                   │
│  │  3. Main loop:               │                                   │
│  │     - Get global model       │                                   │
│  │     - Train (fit)            │                                   │
│  │     - Submit update          │                                   │
│  │  4. Cleanup                  │                                   │
│  └────────┬─────────────────────┘                                   │
│           │ uses                                                    │
│           ▼                                                          │
│  ┌──────────────────────────────────────────┐                      │
│  │   grpc_client.py (GrpcClient)            │                      │
│  │   ────────────────────────────            │                      │
│  │  TWO CHANNELS                            │                      │
│  │  • channel (main)                        │                      │
│  │  • heartbeat_channel                     │                      │
│  │                                           │                      │
│  │  METHODS                                 │                      │
│  │  • register()                            │                      │
│  │  • get_global_model()                    │                      │
│  │  • submit_update()                       │                      │
│  │    ├─> _submit_update_unary()           │                      │
│  │    └─> _submit_update_stream()          │                      │
│  │  • start_heartbeat()                     │                      │
│  │  • send_heartbeat()                      │                      │
│  │  • update_status()                       │                      │
│  │                                           │                      │
│  │  ADAPTIVE LOGIC                          │                      │
│  │  • STREAMING_THRESHOLD_MB = 100          │                      │
│  │  • Detects transformer models            │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌───────────────────── COMMUNICATION LAYER ────────────────────────────┐
│                                                                      │
│  ┌────────────────────────────┐                                     │
│  │   fedlearn.proto           │                                     │
│  │   ─────────────             │                                     │
│  │  MESSAGES                  │                                     │
│  │  • Tensor                  │                                     │
│  │  • ModelParameters         │                                     │
│  │  • ModelChunk              │                                     │
│  │  • ModelUpdateChunk        │                                     │
│  │  • HeartbeatRequest/       │                                     │
│  │    Response                │                                     │
│  │                            │                                     │
│  │  SERVICE                   │                                     │
│  │  • RegisterClient          │                                     │
│  │  • GetGlobalModelStream    │                                     │
│  │  • SubmitModelUpdateStream │                                     │
│  │  • Heartbeat               │                                     │
│  └────────────┬───────────────┘                                     │
│               │ generates                                           │
│               ▼                                                      │
│  ┌───────────────────────────┐                                      │
│  │   generated/              │                                      │
│  │   • fedlearn_pb2.py       │                                      │
│  │   • fedlearn_pb2_grpc.py  │                                      │
│  └───────────────────────────┘                                      │
│                                                                      │
│  ┌────────────────────────────────────────┐                        │
│  │   serializer.py                        │                        │
│  │   ──────────────                       │                        │
│  │  SMALL MODELS (Unary)                 │                        │
│  │  • parameters_to_proto()               │                        │
│  │  • proto_to_parameters()               │                        │
│  │                                        │                        │
│  │  LARGE MODELS (Streaming)              │                        │
│  │  • parameters_to_chunks()              │                        │
│  │    - Serialize with torch.save         │                        │
│  │    - Optional LZ4 compression          │                        │
│  │    - Split into 50MB chunks            │                        │
│  │  • chunks_to_parameters()              │                        │
│  │    - Concatenate chunks                │                        │
│  │    - Decompress if needed              │                        │
│  │    - Load with torch.load              │                        │
│  └────────────────────────────────────────┘                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Complete Round Lifecycle

```
INITIALIZATION PHASE
════════════════════
Server:
  1. start_server() called
  2. Create FLCoordinator with Strategy
  3. Set initial_parameters from strategy
  4. Start gRPC server on specified port
  5. Enter training loop

Clients (parallel):
  1. start_client() called
  2. Create GrpcClient
  3. Register with server
  4. Start heartbeat thread
  5. Enter main loop


ROUND N BEGINS
══════════════
Server (Main Thread):
┌──────────────────────────────────────┐
│ coordinator.start_round()            │
│ └─> _round_complete_event.clear()   │
│                                       │
│ coordinator.wait_for_round_to_       │
│   complete()                         │
│ └─> BLOCKS waiting for event        │
└──────────────────────────────────────┘

SERVER STATUS: WAITING


PHASE 1: MODEL DOWNLOAD
════════════════════════
Client 1:
┌──────────────────────────────────────────────────────┐
│ params, round, config =                              │
│   grpc_client.get_global_model()                     │
│                                                       │
│ GrpcClient:                                          │
│   req = GetGlobalModelRequest(client_id)            │
│   for chunk in stub.GetGlobalModelStream(req):      │
│     chunks.append(chunk.chunk_data)                  │
│                                                       │
│   full_data = b''.join(chunks)                       │
│   model_data = torch.load(BytesIO(full_data))       │
│   return model_data['parameters'], round, config    │
└──────────────────────────────────────────────────────┘
                    ▲
                    │ gRPC Stream
                    │
┌──────────────────────────────────────────────────────┐
│ Server (RPC Thread):                                 │
│                                                       │
│ GrpcServicer.GetGlobalModelStream():                │
│   params = coordinator.get_global_model_for_client()│
│                                                       │
│   buffer = BytesIO()                                │
│   torch.save({'parameters': params}, buffer)         │
│   data = buffer.getvalue()                           │
│                                                       │
│   for i in range(num_chunks):                        │
│     yield ModelChunk(                                │
│       chunk_index=i,                                 │
│       chunk_data=data[start:end]                     │
│     )                                                │
└──────────────────────────────────────────────────────┘

Timeline:
T=0s:  Client requests model
T=1s:  Server starts streaming chunk 1/20
T=3s:  Chunk 5/20 transferred
T=5s:  Chunk 10/20 transferred
T=8s:  Chunk 20/20 transferred (final)
T=9s:  Client reconstructs model


PHASE 2: LOCAL TRAINING
════════════════════════
Client 1:
┌──────────────────────────────────────────────────────┐
│ new_params, num_examples = client.fit(params, config)│
│                                                       │
│ MyClient.fit():                                      │
│   model.load_state_dict(params)                      │
│   model.train()                                       │
│                                                       │
│   for epoch in range(5):                             │
│     for batch_idx, (data, target) in train_loader:  │
│       # Training step                                │
│       loss.backward()                                │
│       optimizer.step()                               │
│                                                       │
│       # Update status every 10 batches              │
│       if batch_idx % 10 == 0:                        │
│         grpc_client.update_status(                   │
│           "training", batch_idx, total_batches       │
│         )                                            │
│                                                       │
│   return model.state_dict(), len(dataset)           │
└──────────────────────────────────────────────────────┘

Parallel (Background Thread):
┌──────────────────────────────────────────────────────┐
│ Heartbeat Thread (every 5 seconds):                  │
│   grpc_client.send_heartbeat()                       │
│   └─> HeartbeatRequest(                             │
│         client_id="client_1",                        │
│         status="training",                           │
│         current_step=45,                             │
│         total_steps=100,                             │
│         current_round=1                              │
│       )                                              │
└──────────────────────────────────────────────────────┘
                    │
                    ▼ gRPC (separate channel)
┌──────────────────────────────────────────────────────┐
│ Server (RPC Thread):                                 │
│   GrpcServicer.Heartbeat():                         │
│     coordinator.update_client_heartbeat(...)         │
│     └─> Update timestamp, print progress           │
└──────────────────────────────────────────────────────┘

Timeline:
T=10s:  Training starts
T=15s:  Heartbeat sent (Step 0/100)
T=20s:  Heartbeat sent (Step 20/100)
T=25s:  Heartbeat sent (Step 40/100)
...
T=60s:  Training complete


PHASE 3: MODEL UPLOAD
══════════════════════
Client 1:
┌──────────────────────────────────────────────────────┐
│ success = grpc_client.submit_update(                 │
│   new_params, num_examples, round                    │
│ )                                                     │
│                                                       │
│ GrpcClient.submit_update():                         │
│   # Calculate size                                   │
│   size_mb = calculate_size(params)                   │
│                                                       │
│   # Decide: streaming or unary?                     │
│   if size_mb > 100 or is_transformer:                │
│     return _submit_update_stream(...)                │
│                                                       │
│ GrpcClient._submit_update_stream():                 │
│   def chunk_generator():                             │
│     for chunk_info in parameters_to_chunks(params): │
│       yield ModelUpdateChunk(                        │
│         chunk_data=chunk_info['chunk_data']          │
│       )                                              │
│                                                       │
│   response = stub.SubmitModelUpdateStream(           │
│     chunk_generator()                                │
│   )                                                   │
└──────────────────────────────────────────────────────┘
                    │
                    ▼ gRPC Stream
┌──────────────────────────────────────────────────────┐
│ Server (RPC Thread):                                 │
│                                                       │
│ GrpcServicer.SubmitModelUpdateStream():             │
│   chunks = []                                        │
│   for chunk in request_iterator:                     │
│     chunks.append(chunk.chunk_data)                  │
│                                                       │
│   full_data = b''.join(chunks)                       │
│   params, num_examples =                             │
│     chunks_to_parameters(full_data)                  │
│                                                       │
│   coordinator.submit_client_update(                  │
│     client_id, params, num_examples, round           │
│   )                                                   │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│ Coordinator.submit_client_update():                  │
│   with self._lock:                                   │
│     # Validate round number                         │
│     if trained_on_round != self.current_round:       │
│       return  # Ignore stale/future updates         │
│                                                       │
│     # Add to list                                   │
│     self._client_updates_received.append(            │
│       (params, num_examples)                         │
│     )                                                │
│                                                       │
│     # Check if we have enough                       │
│     if len(self._client_updates_received) ==         │
│        self.clients_per_round:                       │
│                                                       │
│       self._trigger_aggregation_and_evaluation()    │
└──────────────────────────────────────────────────────┘

Timeline:
T=61s:  Client 1 starts upload
T=65s:  Chunk 5/18 uploaded
T=70s:  Chunk 10/18 uploaded
T=75s:  Chunk 18/18 uploaded
T=76s:  Server reconstructs model
T=76s:  Coordinator receives update (1/3 clients)

[Clients 2 and 3 repeat phases 1-3 in parallel]

T=120s: Client 2 submits update (2/3 clients)
T=180s: Client 3 submits update (3/3 clients)
T=180s: TRIGGER AGGREGATION


PHASE 4: AGGREGATION
════════════════════
Coordinator (Main Thread Wakes Up):
┌──────────────────────────────────────────────────────┐
│ _trigger_aggregation_and_evaluation():               │
│                                                       │
│   # Get all updates                                  │
│   results = list(self._client_updates_received)     │
│   # [(params_1, 1000), (params_2, 500),             │
│   #  (params_3, 1500)]                              │
│                                                       │
│   self._client_updates_received.clear()             │
│                                                       │
│   # Aggregate                                        │
│   aggregated_params = self.strategy.aggregate_fit(   │
│     self.current_round,                              │
│     results                                          │
│   )                                                   │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│ Strategy.aggregate_fit():                            │
│   return self.aggregator.aggregate(results)          │
│                                                       │
│ FedAvgAggregator.aggregate():                       │
│   total_examples = 1000 + 500 + 1500 = 3000         │
│                                                       │
│   aggregated = OrderedDict()                         │
│   for params, num_examples in results:               │
│     weight = num_examples / total_examples           │
│     # 0.333, 0.167, 0.500                           │
│                                                       │
│     for key in params:                               │
│       aggregated[key] += params[key] * weight        │
│                                                       │
│   return aggregated                                  │
└──────────────────────────────────────────────────────┘

Mathematical Example:
  Client 1 weight: layer.weight = [[1.0, 2.0], [3.0, 4.0]]  (1000 samples)
  Client 2 weight: layer.weight = [[2.0, 3.0], [4.0, 5.0]]  (500 samples)
  Client 3 weight: layer.weight = [[1.5, 2.5], [3.5, 4.5]]  (1500 samples)

  Aggregated = 0.333*[[1.0, 2.0], [3.0, 4.0]] +
               0.167*[[2.0, 3.0], [4.0, 5.0]] +
               0.500*[[1.5, 2.5], [3.5, 4.5]]
             = [[1.417, 2.417], [3.417, 4.417]]


PHASE 5: EVALUATION
═══════════════════
Coordinator:
┌──────────────────────────────────────────────────────┐
│   # Update global model                              │
│   self._global_model_params = aggregated_params      │
│                                                       │
│   # Evaluate                                         │
│   loss, metrics = self.strategy.evaluate(            │
│     self.current_round,                              │
│     self._global_model_params                        │
│   )                                                   │
└──────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────┐
│ Strategy.evaluate():                                 │
│   if self.evaluate_fn is None:                       │
│     return None                                      │
│                                                       │
│   # User-provided evaluation function               │
│   loss, metrics = self.evaluate_fn(                  │
│     server_round,                                    │
│     parameters                                       │
│   )                                                   │
│                                                       │
│   # Example evaluate_fn:                            │
│   def evaluate_fn(round_num, params):                │
│     model.load_state_dict(params)                    │
│     model.eval()                                     │
│                                                       │
│     total_loss = 0                                   │
│     correct = 0                                      │
│     for data, target in test_loader:                 │
│       output = model(data)                           │
│       loss = criterion(output, target)               │
│       total_loss += loss.item()                      │
│       correct += (output.argmax(1) == target).sum() │
│                                                       │
│     accuracy = correct / len(test_loader.dataset)    │
│     return total_loss / len(test_loader), {          │
│       'accuracy': accuracy                           │
│     }                                                │
└──────────────────────────────────────────────────────┘

Timeline:
T=181s: Aggregation complete
T=182s: Evaluation starts
T=185s: Evaluation complete

Output:
┌──────────────────────────────────────────────────────┐
│ [Server] Round 1 complete.                           │
│ Metrics: {'loss': 1.2345, 'accuracy': 0.78}         │
└──────────────────────────────────────────────────────┘


PHASE 6: ROUND COMPLETION
══════════════════════════
Coordinator:
┌──────────────────────────────────────────────────────┐
│   self.latest_metrics = {"loss": loss, **metrics}    │
│                                                       │
│   # Signal round complete                           │
│   self._round_complete_event.set()                   │
└──────────────────────────────────────────────────────┘
                    │
                    ▼ UNBLOCKS
┌──────────────────────────────────────────────────────┐
│ Server (Main Thread):                                │
│   # wait_for_round_to_complete() returns            │
│                                                       │
│   metrics = coordinator.get_latest_metrics()         │
│   history.append((round_num, metrics))               │
│                                                       │
│   # Advance to next round                           │
│   coordinator.current_round += 1                     │
│                                                       │
│   # Loop continues for next round...                │
└──────────────────────────────────────────────────────┘


CLIENTS CONTINUE
════════════════
All Clients (in parallel):
┌──────────────────────────────────────────────────────┐
│ while True:                                          │
│   params, server_round, config =                     │
│     comm_client.get_global_model()                   │
│                                                       │
│   if server_round == -1:                             │
│     break  # Training complete                      │
│                                                       │
│   if server_round > last_completed_round:            │
│     # New round! Repeat phases 1-3                  │
│     ...                                              │
│   else:                                              │
│     # Server still on same round, wait              │
│     time.sleep(5)                                    │
└──────────────────────────────────────────────────────┘


TRAINING COMPLETION
═══════════════════
Server (after num_rounds):
┌──────────────────────────────────────────────────────┐
│ final_params = coordinator.get_global_model_params() │
│ return history, final_params                         │
└──────────────────────────────────────────────────────┘

Clients (detect completion):
┌──────────────────────────────────────────────────────┐
│ params, server_round, config =                       │
│   comm_client.get_global_model()                     │
│                                                       │
│ if server_round == -1:                               │
│   print("Server finished training. Shutting down.")  │
│   break                                              │
│                                                       │
│ # Cleanup                                            │
│ comm_client.stop_heartbeat()                         │
│ comm_client.close()                                  │
└──────────────────────────────────────────────────────┘
```

---

## Communication Protocol

### gRPC Service Definition

```
service FederatedLearningService {
  ┌─────────────────────────────────────────────────────────┐
  │ RegisterClient                                          │
  │ Request:  RegisterClientRequest                         │
  │ Response: RegisterClientResponse                        │
  │ Type:     Unary                                         │
  │ Purpose:  Client registration at startup                │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │ GetGlobalModelStream                                    │
  │ Request:  GetGlobalModelRequest                         │
  │ Response: stream ModelChunk                             │
  │ Type:     Server Streaming                              │
  │ Purpose:  Download global model in chunks               │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │ SubmitModelUpdateStream                                 │
  │ Request:  stream ModelUpdateChunk                       │
  │ Response: SubmitModelUpdateResponse                     │
  │ Type:     Client Streaming                              │
  │ Purpose:  Upload trained model in chunks                │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │ Heartbeat                                               │
  │ Request:  HeartbeatRequest                              │
  │ Response: HeartbeatResponse                             │
  │ Type:     Unary                                         │
  │ Purpose:  Keep-alive and status updates                 │
  └─────────────────────────────────────────────────────────┘
}
```

### Message Sizes

```
Small Messages (<1KB):
  • RegisterClientRequest
  • RegisterClientResponse
  • HeartbeatRequest
  • HeartbeatResponse
  • GetGlobalModelRequest

Large Messages (Variable):
  • ModelChunk: 50 MB per chunk
  • ModelUpdateChunk: 50 MB per chunk
  
Total Transfer Sizes (Example):
  • Small CNN (10 MB):    1 chunk each direction
  • Medium ResNet (150 MB): 3 chunks each direction
  • GPT-2 (500 MB):       10 chunks each direction
  • LLaMA-7B (14 GB):     280 chunks each direction
```

### Streaming Strategy

```
┌────────────────────────────────────────────────────┐
│           ADAPTIVE STREAMING DECISION               │
├────────────────────────────────────────────────────┤
│                                                     │
│  if is_transformer(model):                         │
│    USE STREAMING  (reason: transformer detected)   │
│                                                     │
│  elif model_size_mb > 100:                         │
│    USE STREAMING  (reason: size > threshold)       │
│                                                     │
│  else:                                             │
│    USE UNARY      (reason: small model)            │
│                                                     │
└────────────────────────────────────────────────────┘

Transformer Detection:
  Keywords in layer names:
    • 'transformer'
    • 'bert'
    • 'gpt'
    • 'attention'
    • 'encoder'
    • 'decoder'
```

---

## Concurrency Model

### Threading Architecture

```
┌───────────────── SERVER ─────────────────┐
│                                           │
│  Main Thread                              │
│  ┌─────────────────────────────────────┐ │
│  │  Training Loop                      │ │
│  │  for round in rounds:               │ │
│  │    coordinator.start_round()        │ │
│  │    coordinator.wait_for_round_to_   │ │
│  │      complete()  [BLOCKS]           │ │
│  │    coordinator.current_round += 1   │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  gRPC ThreadPool (10 workers)             │
│  ┌─────────────────────────────────────┐ │
│  │ Thread 1: RegisterClient RPC        │ │
│  ├─────────────────────────────────────┤ │
│  │ Thread 2: GetGlobalModelStream RPC  │ │
│  ├─────────────────────────────────────┤ │
│  │ Thread 3: SubmitModelUpdateStream   │ │
│  ├─────────────────────────────────────┤ │
│  │ Thread 4: Heartbeat RPC             │ │
│  ├─────────────────────────────────────┤ │
│  │ Thread 5: Heartbeat RPC             │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  Synchronization                          │
│  • coordinator._lock (state mutations)    │
│  • coordinator._round_complete_event      │
│  • coordinator.heartbeat_lock             │
│                                           │
└───────────────────────────────────────────┘

┌───────────────── CLIENT ─────────────────┐
│                                           │
│  Main Thread                              │
│  ┌─────────────────────────────────────┐ │
│  │  Client Loop                        │ │
│  │  while True:                         │ │
│  │    get_global_model()               │ │
│  │    fit()                            │ │
│  │    submit_update()                  │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  Heartbeat Thread (daemon)                │
│  ┌─────────────────────────────────────┐ │
│  │  while heartbeat_active:            │ │
│  │    send_heartbeat()                 │ │
│  │    sleep(5)                         │ │
│  └─────────────────────────────────────┘ │
│                                           │
│  Two gRPC Channels                        │
│  • Main channel (model transfer)          │
│  • Heartbeat channel (keep-alive)         │
│                                           │
└───────────────────────────────────────────┘
```

### Synchronization Mechanisms

```python
# Coordinator uses threading primitives

1. Lock for State Mutations
   self._lock = threading.Lock()
   
   with self._lock:
     self._client_updates_received.append(update)
     if len(self._client_updates_received) == threshold:
       self._trigger_aggregation()

2. Event for Round Completion
   self._round_complete_event = threading.Event()
   
   # Server main thread
   coordinator.wait_for_round_to_complete()
   # → Blocks on event.wait()
   
   # RPC thread (when aggregation done)
   self._round_complete_event.set()
   # → Wakes up main thread

3. Separate Lock for Heartbeats
   self.heartbeat_lock = Lock()
   
   # Prevents heartbeat updates from blocking
   # client update submissions
```

---

## Design Patterns

### 1. Strategy Pattern (Aggregation)

```python
# Abstract strategy
class Strategy(ABC):
    @abstractmethod
    def aggregate_fit(self, results):
        pass

# Concrete strategies
class FedAvg(Strategy):
    def aggregate_fit(self, results):
        return weighted_average(results)

class FedProx(Strategy):
    def aggregate_fit(self, results):
        return weighted_average_with_proximal_term(results)

# Usage
strategy = FedAvg(...)  # Can swap with FedProx
coordinator = FLCoordinator(strategy=strategy)
```

**Benefits**:
- Easy to add new aggregation algorithms
- Algorithms are interchangeable
- Clean separation of concerns

### 2. Template Method Pattern (Client)

```python
# Abstract template
class Client(ABC):
    @abstractmethod
    def fit(self, parameters, config):
        """Subclasses implement training logic"""
        pass
    
    @abstractmethod
    def get_parameters(self):
        """Subclasses implement parameter extraction"""
        pass

# Concrete implementation
class CNNClient(Client):
    def fit(self, parameters, config):
        # Custom training for CNN
        pass
    
    def get_parameters(self):
        return self.model.state_dict()

# Common behavior in start_client()
def start_client(client):
    while True:
        params = get_model()
        new_params = client.fit(params)  # Calls subclass method
        submit(new_params)
```

**Benefits**:
- Common client lifecycle in one place
- Custom training logic in subclasses
- Prevents code duplication

### 3. Facade Pattern (GrpcClient)

```python
# Complex gRPC internals
class GrpcClient:
    def submit_update(self, params, num_examples, round):
        # Hides complexity:
        # - Size calculation
        # - Streaming vs unary decision
        # - Serialization
        # - Chunking
        # - Error handling
        
        if should_stream(params):
            return self._submit_update_stream(...)
        else:
            return self._submit_update_unary(...)

# Simple client interface
client = GrpcClient(client_id, address)
client.submit_update(params, 1000, 1)  # Simple call!
```

**Benefits**:
- Hides gRPC complexity
- Simple API for clients
- Easy to modify internals

### 4. Observer Pattern (Heartbeat)

```python
# Coordinator observes client state
class FLCoordinator:
    def update_client_heartbeat(self, client_id, status, ...):
        self.client_heartbeats[client_id] = {
            'status': status,
            'last_seen': time.time()
        }
        # Print progress if needed
        if current_step % 10 == 0:
            print(f"Client {client_id}: {status}")

# Clients notify coordinator
class GrpcClient:
    def _heartbeat_loop(self):
        while active:
            self.send_heartbeat()  # Notifies coordinator
            time.sleep(5)
```

**Benefits**:
- Loose coupling
- Real-time monitoring
- No polling needed

### 5. Iterator Pattern (Chunking)

```python
# Generator yields chunks one at a time
def parameters_to_chunks(params) -> Generator[Dict, None, None]:
    for i in range(num_chunks):
        yield {
            'chunk_index': i,
            'chunk_data': data[start:end],
            'is_final_chunk': (i == num_chunks - 1)
        }

# Used in streaming
def chunk_generator():
    for chunk_info in parameters_to_chunks(params):
        yield ModelUpdateChunk(**chunk_info)

response = stub.SubmitModelUpdateStream(chunk_generator())
```

**Benefits**:
- Memory efficient (one chunk at a time)
- Works with gRPC streaming
- Clean separation of chunking logic

---

## Scalability

### Horizontal Scalability

```
NUMBER OF CLIENTS
═════════════════

Current: 2-10 clients (tested)
Supported: Up to 100+ clients

Bottleneck: Server aggregation time
Solution: 
  • Asynchronous aggregation
  • Hierarchical aggregation
  • Client sampling


MODEL SIZE
══════════

Small: <100 MB (unary transfer)
Medium: 100 MB - 1 GB (streaming)
Large: 1 GB - 10 GB+ (streaming)

Tested: Up to 2 GB models
Theoretical: Limited by disk space and memory


COMMUNICATION EFFICIENCY
════════════════════════

Baseline: Full model transfer each round
Optimizations:
  • Compression (LZ4): 2-3x reduction
  • Gradient-only transfer: 1x size
  • Sparse updates: Variable reduction
  • Quantization: 4-8x reduction


COMPUTATION TIME
════════════════

Factors:
  • Model size
  • Dataset size
  • Hardware (CPU/GPU)
  • Number of local epochs

Example (CNN on MNIST):
  • Local training: 2 minutes
  • Model download: 5 seconds
  • Model upload: 5 seconds
  • Total per round: ~2.5 minutes

Example (GPT-2 on text):
  • Local training: 30 minutes
  • Model download: 45 seconds
  • Model upload: 60 seconds
  • Total per round: ~32 minutes
```

### Performance Optimizations

```
1. ADAPTIVE STREAMING
   • Small models: Unary (faster)
   • Large models: Streaming (reliable)

2. DUAL CHANNELS
   • Main: Model transfer (blocks)
   • Heartbeat: Keep-alive (non-blocking)

3. COMPRESSION
   • Optional LZ4 compression
   • 2-3x size reduction
   • Minimal CPU overhead

4. THREADING
   • 10 gRPC worker threads
   • Concurrent client handling
   • Background heartbeat threads

5. MEMORY EFFICIENCY
   • Streaming serialization
   • Chunk-by-chunk processing
   • Immediate garbage collection
```

### Future Enhancements

```
1. ASYNCHRONOUS AGGREGATION
   Current: Wait for all clients
   Future: Aggregate as clients arrive

2. CLIENT SELECTION
   Current: Use all connected clients
   Future: Sample subset per round

3. GRADIENT COMPRESSION
   Current: Full parameter transfer
   Future: Gradient sparsification

4. HIERARCHICAL AGGREGATION
   Current: Flat client-server
   Future: Multi-tier aggregation

5. DIFFERENTIAL PRIVACY
   Current: No privacy guarantees
   Future: DP-SGD, secure aggregation
```

---

## Deployment Architecture

### Single-Machine Setup

```
┌─────────────────────────────────┐
│      Single Machine             │
├─────────────────────────────────┤
│  Server Process                 │
│  • Port 50051                   │
│                                 │
│  Client Process 1               │
│  Client Process 2               │
│  Client Process 3               │
└─────────────────────────────────┘

Use Case: Development, debugging
Command:
  Terminal 1: python run_server.py
  Terminal 2: python run_client.py --client-id=1
  Terminal 3: python run_client.py --client-id=2
```

### Multi-Machine Setup

```
┌─────────────────────┐
│  Server Machine     │
│  IP: 192.168.1.100  │
│  Port: 50051        │
└─────────────────────┘
         │
         │ Internet/LAN
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
    ▼         ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│Client1│ │Client2│ │Client3│ │ClientN│
│  GPU  │ │  GPU  │ │  CPU  │ │  GPU  │
└───────┘ └───────┘ └───────┘ └───────┘

Use Case: Real federated learning
Setup:
  Server: python run_server.py --address=0.0.0.0:50051
  Clients: python run_client.py --server=192.168.1.100:50051
```

### Cloud Deployment (AWS)

```
┌──────────────────────────────────────────┐
│           AWS Cloud                       │
├──────────────────────────────────────────┤
│  EC2 Instance (Server)                   │
│  • Type: t3.large                        │
│  • Public IP: X.X.X.X                    │
│  • Security Group: Port 50051 open       │
│                                          │
│  EC2 Instances (Clients)                 │
│  • Type: g4dn.xlarge (GPU)               │
│  • Private IPs: Connect to server        │
└──────────────────────────────────────────┘

Docker Deployment:
  docker run -p 50051:50051 fedlearn-server
  docker run fedlearn-client --server=X.X.X.X:50051
```

---

## Summary

### Key Architectural Decisions

1. **gRPC for Communication**
   - Efficient binary protocol
   - Built-in streaming support
   - Cross-platform compatibility

2. **Adaptive Streaming**
   - Handles models from 1MB to 10GB+
   - Automatic decision based on size
   - Robust to network issues

3. **Dual-Channel Design**
   - Prevents heartbeat blocking
   - Enables long-running transfers
   - Improves reliability

4. **Strategy Pattern for Aggregation**
   - Easy to extend
   - Pluggable algorithms
   - Clean separation of concerns

5. **Thread-Safe Coordination**
   - Supports concurrent clients
   - Prevents race conditions
   - Enables synchronization

### Trade-offs

```
CHOSEN                vs              ALTERNATIVE
──────────────────────────────────────────────────
gRPC/Protobuf         vs  HTTP/REST + JSON
• Faster              vs  • Easier debugging
• Streaming           vs  • Better tooling
• Type-safe           vs  • More familiar

Synchronous Rounds    vs  Asynchronous
• Simpler logic       vs  • Faster convergence
• Better convergence  vs  • Handles stragglers
• Easier debugging    vs  • More efficient

Weighted Averaging    vs  Median/Krum
• Faster              vs  • Byzantine robust
• Standard            vs  • More complex
• Well-understood     vs  • Slower
```

This architecture provides a solid foundation for federated learning while remaining extensible for future enhancements.