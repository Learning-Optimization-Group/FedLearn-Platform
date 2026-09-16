# FedLearn Python Framework — Wiki

Welcome to the internal documentation for the **FedLearn Python Framework** (`framework/`).

This framework is the core distributed federated learning engine that powers the FedLearn-Platform. It is a standalone Python library (`fedlearn`) built on PyTorch and gRPC, capable of training CNNs, Transformers, and Large Language Models in a privacy-preserving, distributed manner — over gRPC across real processes, or entirely **in-process** through the simulator.

---

## Documentation Index

| # | Document | Description |
|---|----------|-------------|
| 1 | [Architecture & Package Overview](01_architecture_overview.md) | Module map (all 57 modules), package layout, the in-process simulator, how it all connects |
| 2 | [gRPC Communication Layer](02_grpc_communication.md) | Proto definitions, safetensors wire, streaming, upload caps, TLS, retry logic |
| 3 | [Server Internals](03_server_internals.md) | `start_server`, `FLCoordinator`, round lifecycle, dropout deadline, ingress defenses |
| 4 | [Client Internals](04_client_internals.md) | `Client` ABC, `LocalTrainer`, `GrpcClient`, polling loop, server-driven stop |
| 5 | [Aggregation Strategies](05_strategies.md) | The six registered strategies, the factory, Byzantine-robust aggregation, central DP |
| 6 | [DeComFL — Dimension-Free Federated Learning](06_decomfl.md) | Algorithm 3 & 4, zeroth-order estimation, seed/gradient protocol, LR stability envelope |
| 7 | [Data Partitioning & Non-IID Scenarios](07_data_partitioning.md) | The four shipped partitioners, Dirichlet α, heterogeneity, `partition_report` |
| 8 | [Examples & Benchmarks](08_examples.md) | The seven examples, the committed benchmark harnesses, the E2E test scripts |
| 9 | [Developer Guide — Extending the Framework](09_developer_guide.md) | Adding strategies/clients, the proto workflow, testing, contributing |

---

## Quick Navigation

**New to the framework?** Start with [01 — Architecture Overview](01_architecture_overview.md).

**Debugging a communication error?** Jump to [02 — gRPC Communication](02_grpc_communication.md).

**Implementing a new aggregation algorithm?** See [05 — Strategies](05_strategies.md) and [09 — Developer Guide](09_developer_guide.md).

**Understanding the DeComFL paper implementation?** Go straight to [06 — DeComFL](06_decomfl.md).

**Running an experiment at 1,000+ clients?** See the simulator section in
[01 — Architecture Overview](01_architecture_overview.md#the-in-process-simulator) and the
partitioners in [07 — Data Partitioning](07_data_partitioning.md).

---

## Framework At A Glance

```
fedlearn (Python package, 57 modules under src/fedlearn/)
├── server/          – start_server, FLCoordinator, the six strategies, robust aggregation, gRPC servicer
├── client/          – Client ABC, GrpcClient transport, LocalTrainer (first-order), DeComFL client
├── communication/   – Proto mirrors, serializer, deterministic safetensors codec, generated stubs
├── estimators/      – Zeroth-order estimator, canonical perturbation RNG, trainable-parameter manifest
├── simulation/      – In-process federation driver, seeded partitioners, per-client RNG isolation
├── privacy/         – Central-DP mechanism + from-scratch RDP accountant
├── security/        – TLS policy, connection-token verify, gRPC server/client interceptors, identity
├── backbone/        – Frozen-backbone serialization + content-addressed cache (DA-11)
├── bundle/          – Adapter-bundle manifest + JSON schema
├── fot/             – Federation over Text (separate, torch-free research mode)
└── data/            – Raw MNIST download directory used by the examples (no Python code)
```

> `data/` holds only downloaded MNIST IDX files. Partitioning helpers live in
> `simulation/partition.py` — see [07 — Data Partitioning](07_data_partitioning.md).

The framework is consumed by:
- **`fl-runtime/`** — the executable layer the Spring Boot backend actually shells out to (`client.py`, `fl_server.py`, …) does `import fedlearn as fl`
- **`fedlearn-desktop`** Electron app (spawns the FL **client** as a child process — the packaged PyInstaller bundle, or system `python3` against `fl-runtime/client.py` in dev mode)
- **`backend`** Spring Boot API (spawns Python FL servers as local processes via `FlServerManager`)
- **`client-docker`** Docker image for containerised client deployment (build context is the repo root)
