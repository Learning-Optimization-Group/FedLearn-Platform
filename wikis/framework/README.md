# FedLearn Python Framework — Wiki

Welcome to the internal documentation for the **FedLearn Python Framework** (`framework/`).

This framework is the core distributed federated learning engine that powers the FedLearn-Platform. It is a standalone Python library (`fedlearn`) built on PyTorch and gRPC, capable of training CNNs, Transformers, and Large Language Models in a privacy-preserving, distributed manner.

---

## Documentation Index

| # | Document | Description |
|---|----------|-------------|
| 1 | [Architecture & Package Overview](01_architecture_overview.md) | High-level module map, package layout, and how all components connect |
| 2 | [gRPC Communication Layer](02_grpc_communication.md) | Proto definitions, serialization, streaming, TLS, retry logic |
| 3 | [Server Internals](03_server_internals.md) | `start_server`, `FLCoordinator`, round lifecycle, heartbeat management |
| 4 | [Client Internals](04_client_internals.md) | `Client` ABC, `GrpcClient`, polling loop, large model streaming |
| 5 | [Aggregation Strategies](05_strategies.md) | `Strategy` ABC, `FedAvg` deep-dive, weighted aggregation, extensibility |
| 6 | [DeComFL — Dimension-Free Federated Learning](06_decomfl.md) | Algorithm 3 & 4, zeroth-order estimation, seed/gradient protocol |
| 7 | [Data Partitioning & Non-IID Scenarios](07_data_partitioning.md) | Dirichlet distribution, heterogeneous splits, practical setup |
| 8 | [Examples Walkthrough](08_examples.md) | End-to-end traces for MNIST, LLM, and ECG federation examples |
| 9 | [Developer Guide — Extending the Framework](09_developer_guide.md) | Adding custom strategies, custom clients, testing, contributing |

---

## Quick Navigation

**New to the framework?** Start with [01 — Architecture Overview](01_architecture_overview.md).

**Debugging a communication error?** Jump to [02 — gRPC Communication](02_grpc_communication.md).

**Implementing a new aggregation algorithm?** See [05 — Strategies](05_strategies.md) and [09 — Developer Guide](09_developer_guide.md).

**Understanding the DeComFL paper implementation?** Go straight to [06 — DeComFL](06_decomfl.md).

---

## Framework At A Glance

```
fedlearn (Python package)
├── server/          – Server entry point, coordinator, strategies, gRPC servicer
├── client/          – Client ABC, gRPC wrapper, DeComFL client
├── communication/   – Proto definitions, serializer, generated stubs
├── estimators/      – Zeroth-order gradient estimators (DeComFL)
└── data/            – Dataset utilities (MNIST partitioning, etc.)
```

The framework is consumed by:
- **`fedlearn-desktop`** Electron app (invokes `run_server.py` / `run_client.py` as child processes)
- **`backend`** Spring Boot API (spawns Python FL servers via `ProcessBuilder` or AWS ECS Fargate)
- **`client-docker`** Docker image for containerised client deployment
