# 17 — Low-Level Design: Data & Artifact Plane (FedLearn Platform v2)

**Document type:** Production build specification — Low-Level Design (LLD).
**Unit:** the Data & Artifact Plane.
**Audience:** a mid-sized local Large Language Model (LLM, ~30 billion parameters, e.g. Qwen/Llama 32B on an Apple M4 Max). It cannot infer missing context. Every interface, version, file path, environment-variable name, and command below is **pre-decided**. Implement exactly what is written; do not substitute alternatives or invent technologies not defined here or in the foundation docs.
**Status:** authoritative for the Data & Artifact Plane of v2. Conforms to and must not contradict:

- `docs/v2/build/02-TECH-STACK.md` (the locked, version-pinned stack),
- `docs/v2/build/03-DATA-MODEL.md` (the database schema),
- `docs/v2/build/04-API-CONTRACTS.md` (REST / gRPC / STOMP contracts).

**Date authored:** 2026-05-29. **Branch:** `main-clean`.

> **Reasoning is inline and load-bearing.** Every nontrivial choice cites the v2 audit finding it derives from: `C2-data-engineering.md` (cited `C2 §x`), `C3-reproducibility.md` (cited `C3 §x` / `C3-1`), `B2-tech-stack.md` (cited `B2 §x`). Existing-code claims cite `file:line` against `main-clean`. External claims cite a Uniform Resource Locator (URL). Uncertainty is flagged explicitly, never papered over.

---

## 0. Abbreviation key (first-use expansion; thereafter the short form)

| Short form | Full form |
|---|---|
| LLD | Low-Level Design |
| LLM | Large Language Model |
| FL | Federated Learning |
| DeComFL | Dimension-Free Communication Federated Learning (zeroth-order optimization; ICLR 2025, RIT/Yang) |
| FedAvg | Federated Averaging |
| IID | Independent and Identically Distributed |
| non-IID | not Independent and Identically Distributed |
| RNG | Random Number Generator |
| PCG64 | Permuted Congruential Generator, 64-bit (a NumPy bit generator) |
| ZO | Zeroth-Order (optimization) |
| API | Application Programming Interface |
| REST | Representational State Transfer |
| gRPC | Google Remote Procedure Call |
| STOMP | Simple Text Oriented Messaging Protocol |
| S3 | (Amazon) Simple Storage Service |
| MinIO | a self-hosted S3-compatible object store (not an acronym) |
| MLflow | Machine-Learning lifecycle tool (not an acronym) |
| npz | NumPy zip (NumPy's `.npz` archive container) |
| sha256 | Secure Hash Algorithm 256-bit |
| JSON | JavaScript Object Notation |
| JSONB | JSON Binary (PostgreSQL binary JSON column type) |
| UUID | Universally Unique Identifier |
| BIGINT | 64-bit signed integer SQL type |
| PK | Primary Key |
| FK | Foreign Key |
| DDL | Data Definition Language |
| SQL | Structured Query Language |
| ORM | Object-Relational Mapping |
| JPA | Jakarta Persistence API |
| HF | Hugging Face |
| HTTP | HyperText Transfer Protocol |
| HTTPS | HTTP Secure |
| URL | Uniform Resource Locator |
| URI | Uniform Resource Identifier |
| ABC | Abstract Base Class (Python `abc` module) |
| DLG | Deep Leakage from Gradients |
| CI | Continuous Integration |
| AGPLv3 | GNU Affero General Public License v3 |
| TTL | Time-To-Live |
| AWS | Amazon Web Services |
| SDK | Software Development Kit |
| RDS | (Amazon) Relational Database Service |
| OLTP | Online Transaction Processing |
| DP | Differential Privacy |

---

## 1. Purpose & single responsibility

The Data & Artifact Plane is the v2 subsystem that makes FL runs **reproducible and resumable** by owning two content-addressed substrates and the interfaces that produce them:

1. **The data substrate** — a framework-side `DataSource` + `Partitioner` interface that (a) loads a dataset, (b) produces a deterministic non-IID Dirichlet partition, and (c) emits a non-reversible `SchemaFingerprint`. This **replaces the four forked `dirichlet_split` copies** (C2 §2.2) and **removes the `flwr-datasets` dependency** (C2 §3.3, B2 §2.3). Its outputs are persisted as content-addressed npz partition files plus three Flyway-owned registry tables (`datasets`, `dataset_versions`, `partition_recipes`) keyed on content hashes.
2. **The artifact substrate** — an S3/MinIO model-artifact store, content-addressed by sha256, wired to `fl_runs` for initial / per-round checkpoint / final models (C3 §5.5, B2 §5 "Artifact/model store"). On top of it sits the MLflow Model Registry and the per-run **determinism manifest** (seed + library/dataset/model hashes; C3 §5.2).

**Single responsibility (one sentence):** *given a dataset version and a partition recipe, deterministically materialize a client's data partition and its fingerprint, and given a run, content-address every model byte it reads or writes so the run can be reproduced and resumed.*

**Explicitly NOT this unit's responsibility** (owned elsewhere, do not implement here):

| Concern | Owner |
|---|---|
| The FL aggregation math (FedAvg `1/N`, DeComFL `1/P`), perturbation generation | FL framework strategies LLD |
| The `fl_runs` lease / reconciler / launcher state machine | Control-plane / orchestration LLD |
| REST controllers, per-run token minting/validation, STOMP fan-out | Control-plane API LLD |
| The gRPC servicer wire handling (chunking transport, heartbeat) | FL framework communication LLD |
| Optimizer (perturbation) seed semantics inside DeComFL | FL framework strategies LLD (this unit owns only the **data** seed) |

> **Reasoning — why a dedicated plane.** C2's verdict is that "the data plane is the least-developed subsystem in the platform and the one most central to the product thesis" (C2 §1). v1 had **no dataset registry, no versioning, no lineage, no schema validation** (C2 §2.5) and **no artifact store** (B2 §1 "Artifact store: None… S3 is a TODO"). Reproducibility (C3) is impossible without both substrates. Isolating them into one plane gives the determinism manifest a stable dataset hash and a stable model hash to reference (C3 §5.2).

---

## 2. Position in the system

### 2.1 Depends-on

| Depends on | For | Reference |
|---|---|---|
| HF `datasets` + NumPy | Loading the raw dataset and computing the Dirichlet split | 02-TECH-STACK §4.3 |
| `safetensors` (model bytes), `numpy` (`np.savez_compressed` for partitions) | The two serialization codecs | 02-TECH-STACK §4.2, C2 §3.5 |
| PyTorch `2.12.0` | `torch.utils.data.DataLoader`, `torch.Generator` for seeded shuffling | 02-TECH-STACK §4.1 |
| S3 / MinIO (object store) + AWS SDK (Python `boto3` / Java `s3`) | Storing/fetching content-addressed blobs | 02-TECH-STACK §7 |
| PostgreSQL `17.10` + Flyway (Boot BOM-managed) | The registry tables `datasets`, `dataset_versions`, `partition_recipes`, `model_artifacts`, `determinism_manifests` | 02-TECH-STACK §5, §9; 03-DATA-MODEL §5 |
| MLflow `3.12.0` (self-hosted) | The Model Registry + run-tracking link-out (`fl_runs.mlflow_run_id`) | 02-TECH-STACK §8 |

### 2.2 Depended-by

| Depended by | Uses | Reference |
|---|---|---|
| FL framework client (`fit()`) | `DataSource.train_loader()` / `eval_loader()` | C2 §3.3 |
| FL framework server | `DataSource` for any server-held global test set; reads/writes checkpoints via the artifact store | 04-API §10 |
| FL server → control plane callbacks | `CheckpointReportDto` carries the artifact id this unit produced (`/api/internal/runs/{runId}/checkpoint`) | 04-API §5.1 |
| Control-plane REST | `/api/datasets/**`, `/api/artifacts/**` register the metadata this unit owns | 04-API §8.2, §9 |
| Reproducibility / determinism | The `determinism_manifests` row + `manifest_json` this unit assembles | 04-API §4.4, C3 §5.2 |

### 2.3 Interfaces CONSUMED (by exact name from 04-API-CONTRACTS.md)

The Python framework code in this unit **produces inputs to** and **consumes** these contracts; it does not implement the REST controllers but must produce exactly the shapes they expect:

| Contract (04-API-CONTRACTS.md) | This unit's relationship |
|---|---|
| `POST /api/datasets/{datasetId}/versions` → `CreateDatasetVersionRequest` / `DatasetVersionDto` (§8.2) | This unit computes the `sha256` the request carries and registers the content-addressed blob URI. |
| `POST /api/datasets/{datasetId}/versions/{versionId}/partitions` → `CreatePartitionRecipeRequest` / `PartitionRecipeDto` (§8.2) | This unit computes the `contentHash` (recipe hash) and reproduces the split from the recipe. |
| `POST /api/artifacts/upload-url` → `ArtifactUploadUrlRequest` / `ArtifactUploadUrlResponse` (§9) | The FL server requests a pre-signed PUT keyed on `<sha256>` before uploading a checkpoint. |
| `POST /api/artifacts` → `RegisterArtifactRequest` / `ArtifactDto` (§9) | After PUT, the FL server registers the artifact metadata. |
| `GET /api/artifacts/{artifactId}/download-url` (§9) | The FL server / a resuming run fetches a pre-signed GET to download a checkpoint. |
| `POST /api/internal/runs/{runId}/checkpoint` → `CheckpointReportDto` (§5.1) | The FL server reports the per-round content-addressed checkpoint pointer this unit uploaded. |
| `GET /api/runs/{runId}/manifest` → `DeterminismManifestDto` (§4.4) | This unit's `manifest_json` backs this response. |
| `GET /api/runs/{runId}/checkpoints` → `CheckpointDto[]` (§4.4) | Backed by `model_artifacts` rows of `kind=CHECKPOINT`. |

### 2.4 Interfaces EXPOSED

| Exposed surface | Consumer |
|---|---|
| Python `DataSource` / `Partitioner` ABCs (§5.1) | FL client + server |
| Python `ArtifactStore` client (§5.4) | FL server (checkpoint/resume); desktop client |
| Python `DeterminismManifest` builder (§5.5) | FL server at run start |
| The content-addressed key scheme (§6.4) and the npz partition format (§6.2) | All clients (cross-language: Python now, C++ mobile later — C2 §3.4) |
| The determinism-manifest JSON schema (§6.6) | MLflow logging + the `determinism_manifests` table |

---

## 3. Tech stack for this unit (pinned, from 02-TECH-STACK.md)

| Technology | Pinned version | One-line reasoning | Source |
|---|---|---|---|
| PyTorch (`torch`) | `2.12.0` | `DataLoader`, CPU-canonical `torch.Generator` for seeded shuffling; pinned for cross-build RNG parity (C3-1). | 02-TECH-STACK §4.1, §24.3 |
| `safetensors` | latest `0.4.x` (`verify-before-use` via `pip index versions safetensors`) | Typed, no-pickle model codec; kills the v1 `torch.save` blob foot-gun (C2 §3.5, B2 §3.4). | 02-TECH-STACK §4.2, §24.3 |
| HF `datasets` | latest (`verify-before-use`) | Replaces `flwr-datasets` as the dataset loader; Apache-2.0, no Flower contamination. | 02-TECH-STACK §4.3, §24.3 |
| NumPy | `1.26+`/`2.x` consistent with the torch build (`verify-before-use`) | `np.random.Generator(np.random.PCG64(data_seed))` for the partitioner; `np.savez_compressed` for npz partitions. | 02-TECH-STACK §4.3, §24.3 |
| PostgreSQL (RDS) | `17.10` | The registry tables; `gen_random_uuid()` is built-in (no extension). | 02-TECH-STACK §5.1, §24.4 |
| Flyway (+ `flyway-database-postgresql`) | Boot 3.5 BOM-managed (`verify-before-use`) | Owns the schema (`V6`, `V7`, `V8`); JPA is validate-only. | 02-TECH-STACK §5.2, §24.4 |
| S3 (managed) / MinIO (self-host) | S3: no version. MinIO: latest stable image tag, **pin image digest** (`verify-before-use`). | Content-addressed model/dataset blob store; never DB blobs. | 02-TECH-STACK §7, §24.4 |
| AWS SDK | Python `boto3` (`verify-before-use`); Java `software.amazon.awssdk:s3` current 2.x | Pre-signed URL + object I/O; same SDK speaks S3 and MinIO. | 02-TECH-STACK §7, §24.4 |
| MLflow (self-hosted) | `3.12.0` | Model Registry + run tracking; `$0`, data-resident (B2 §5). | 02-TECH-STACK §8, §24.4 |
| Python (CPython) | `3.12.9` (`verify-before-use`) | The framework runtime; cp312 torch wheels exist x86-64 + ARM64. | 02-TECH-STACK §1.2, §24.1 |

> **License flag (carry forward, do not re-decide):** MinIO is AGPLv3 — fine for internal self-hosted use; legal review required before redistributing a modified MinIO. The managed-SaaS path uses S3 and sidesteps this (02-TECH-STACK §7). MLflow / `datasets` / `safetensors` are Apache-2.0; PyTorch / NumPy are BSD-style — all commercial-friendly.

---

## 4. Module / file structure

### 4.1 Python framework — `framework/src/fedlearn/data/` (net-new module)

> **On-disk note (verified):** `framework/src/fedlearn/data/` exists but currently contains only an `MNIST/raw/` fixture directory and **no Python module** (verified 2026-05-29). All `.py` files below are net-new. The MNIST raw fixtures are an example artifact, not part of this module; do not import from them.

```
framework/src/fedlearn/data/
├── __init__.py            # re-exports DataSource, Partitioner, SchemaFingerprint, DataSchema, PartitionRecipe
├── schema.py              # DataSchema, SchemaFingerprint dataclasses + canonicalization + schema_hash
├── source.py              # DataSource ABC; HfDataSource concrete impl over HF `datasets`
├── partitioner.py         # Partitioner ABC; DirichletLabelPartitioner, DirichletQtyPartitioner,
│                          #   ShardPartitioner, NaturalPartitioner
├── recipe.py              # PartitionRecipe dataclass + recipe_hash() (sha256 over canonical params)
├── npz_store.py           # content-addressed npz read/write: write_partition(), load_partition() + sha256 verify
├── content_hash.py        # canonical_json(), sha256_hex(), dataset_version_content_hash()
└── errors.py              # DataPlaneError hierarchy (§9)
```

| File | One-line responsibility |
|---|---|
| `schema.py` | Declare the local-data schema contract; compute the non-reversible `schema_hash` and per-label histogram. |
| `source.py` | Load a dataset and expose train/eval `DataLoader`s + `fingerprint()`; the abstraction that kills the four forks. |
| `partitioner.py` | Produce a deterministic non-IID split from labels + a recipe, using an **isolated** `PCG64(data_seed)` generator. |
| `recipe.py` | Hold the recipe params and compute `recipe_hash` — the reproducibility key that replaces the v1 pickle filename. |
| `npz_store.py` | Write/read partitions as content-addressed `.npz` + sidecar sha256; verify before deserialize (replaces pickle). |
| `content_hash.py` | The single canonicalization + sha256 helper used by datasets, recipes, and manifests (one hash definition). |
| `errors.py` | Typed exceptions mapped to gRPC `FAILED_PRECONDITION` / `INVALID_ARGUMENT` on the wire (§9). |

### 4.2 Python framework — `framework/src/fedlearn/artifacts/` (net-new module)

```
framework/src/fedlearn/artifacts/
├── __init__.py
├── store.py               # ArtifactStore: put_model(), get_model(), exists(), key_for() — content-addressed
├── presigned.py           # PresignedClient: request upload-url / download-url from the control plane (§9)
├── safetensors_codec.py   # save_state_dict()/load_state_dict() via safetensors; sha256 of the blob
├── manifest.py            # DeterminismManifest dataclass + build() + to_json() (§6.6)
└── mlflow_lineage.py      # log_run_start(), log_round_metrics(), register_final_model() (C3 §5.4)
```

| File | One-line responsibility |
|---|---|
| `store.py` | The content-addressed object-store client (`boto3`); `key_for(sha256)` → object key; dedupe by hash. |
| `presigned.py` | Broker pre-signed PUT/GET via `/api/artifacts/upload-url` and `/download-url` so blob bytes never transit the JVM. |
| `safetensors_codec.py` | Serialize/deserialize a model `state_dict` to/from safetensors bytes and hash them. |
| `manifest.py` | Assemble the per-run determinism manifest from torch/git/proto versions + model/dataset hashes. |
| `mlflow_lineage.py` | Write params/metrics/tags to MLflow and register the final model version with lineage tags. |

### 4.3 Backend (Java, Spring Boot) — JPA entities + repositories

> The full DDL is in 03-DATA-MODEL §5; do not re-author migrations here. This unit's Java surface is the entity + repository + service layer that maps to those tables. Package root: `com.federated.fl_platform_api.dataplane`.

```
backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/dataplane/
├── entity/
│   ├── Dataset.java              # @Entity -> datasets
│   ├── DatasetVersion.java       # @Entity -> dataset_versions (schema_json JSONB)
│   ├── PartitionRecipe.java      # @Entity -> partition_recipes (data_seed Long)
│   ├── ModelArtifact.java        # @Entity -> model_artifacts (sha256 CHAR(64), kind enum)
│   └── DeterminismManifest.java  # @Entity -> determinism_manifests (1:1 fl_runs; manifest_json JSONB)
├── repo/
│   ├── DatasetRepository.java
│   ├── DatasetVersionRepository.java
│   ├── PartitionRecipeRepository.java
│   ├── ModelArtifactRepository.java
│   └── DeterminismManifestRepository.java
├── service/
│   ├── DatasetService.java       # CRUD + content-hash uniqueness checks (§5.6)
│   ├── ArtifactService.java      # pre-signed URL minting; register-by-sha256 dedupe (§5.6)
│   └── ManifestService.java      # persist + serve the determinism manifest
└── enums/
    ├── Modality.java             # TABULAR | IMAGE | TEXT | TIMESERIES
    ├── PartitionerType.java      # DIRICHLET_LABEL | DIRICHLET_QTY | SHARD | NATURAL
    └── ArtifactKind.java         # INITIAL | CHECKPOINT | FINAL
```

> The Flyway migrations themselves (`V6__dataset_registry.sql`, `V7__fl_runs_and_artifacts.sql`, `V8__determinism_manifest.sql`) live at `backend/fl-platform-api/src/main/resources/db/migration/` and are specified verbatim in 03-DATA-MODEL §5. This unit consumes them; the orchestration/data-model build authors them.

---

## 5. Key interfaces & type signatures

All Python signatures below are **full** — the local model implements the bodies, not the contracts. Type hints are mandatory (the framework is `mypy`-gated per the repo pre-commit config).

### 5.1 `DataSource` and `Partitioner` (the abstractions that kill the four forks)

`framework/src/fedlearn/data/source.py`:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from .schema import DataSchema, SchemaFingerprint
from .recipe import PartitionRecipe

class DataSource(ABC):
    """Loads a dataset and exposes loaders + a non-reversible fingerprint.

    The ONLY abstraction clients and the server use to obtain data. Replaces
    the four forked dirichlet_split copies (C2 §2.2) and the flwr-datasets dep.
    """

    @abstractmethod
    def schema(self) -> DataSchema:
        """The declared schema contract for this client's local data."""
        ...

    @abstractmethod
    def fingerprint(self) -> SchemaFingerprint:
        """Non-reversible metadata ONLY: shapes, dtypes, label set, per-label
        counts. NEVER raw features/labels (the two-plane invariant, C2 §3.1)."""
        ...

    @abstractmethod
    def train_loader(self, batch_size: int) -> DataLoader:
        """Deterministically-shuffled training loader for this partition."""
        ...

    @abstractmethod
    def eval_loader(self, batch_size: int) -> DataLoader | None:
        """Local eval loader, or None if the client holds no eval split."""
        ...


class HfDataSource(DataSource):
    """Concrete DataSource over Hugging Face `datasets`, partitioned by a recipe.

    Args:
        dataset_id:   HF dataset id or a local content-addressed path.
        split:        e.g. "train".
        recipe:       the PartitionRecipe pinned by the run (§5.3).
        partition_id: this client's partition index in [0, recipe.num_partitions).
        label_column: the column holding integer labels.
    """

    def __init__(
        self,
        dataset_id: str,
        split: str,
        recipe: PartitionRecipe,
        partition_id: int,
        label_column: str = "label",
    ) -> None:
        ...
```

`framework/src/fedlearn/data/partitioner.py`:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .recipe import PartitionRecipe

class Partitioner(ABC):
    """Maps an array of labels to a list of index arrays (one per partition).

    CONTRACT: deterministic given (labels, recipe). Uses an ISOLATED
    np.random.Generator(np.random.PCG64(recipe.data_seed)) — NEVER the global
    np.random.* state (C2 §3.3 / prior M5). This is the data-seed namespace;
    the optimizer (perturbation) seed is a separate namespace (C2 §4.3).
    """

    @abstractmethod
    def partition(self, labels: np.ndarray, recipe: PartitionRecipe) -> list[np.ndarray]:
        ...


class DirichletLabelPartitioner(Partitioner):
    """Label-distribution-skew Dirichlet split. The canonical non-IID
    partitioner; ported from the GOOD default_rng version (ecg_loader.py:30-71,
    C2 'salvage'). Returns N index arrays whose union == range(len(labels))."""
    def partition(self, labels: np.ndarray, recipe: PartitionRecipe) -> list[np.ndarray]: ...


class DirichletQtyPartitioner(Partitioner):
    """Quantity-skew Dirichlet split (partition SIZES drawn Dirichlet(alpha));
    labels distributed IID within each partition. Maps recipe.partitioner=DIRICHLET_QTY."""
    def partition(self, labels: np.ndarray, recipe: PartitionRecipe) -> list[np.ndarray]: ...


class ShardPartitioner(Partitioner):
    """Sort-by-label then deal contiguous shards (the classic McMahan shard split).
    alpha is ignored. Maps recipe.partitioner=SHARD."""
    def partition(self, labels: np.ndarray, recipe: PartitionRecipe) -> list[np.ndarray]: ...


class NaturalPartitioner(Partitioner):
    """Identity: data is ALREADY partitioned by where it physically sits
    (real hospital federations, C2 §3.2). Returns the single client's own
    indices; reproducibility = same recipe + same declared schema, NOT same
    bytes (C2 §6 'natural-partition reproducibility is bounded')."""
    def partition(self, labels: np.ndarray, recipe: PartitionRecipe) -> list[np.ndarray]: ...
```

> **Mapping note (04-API vs framework enum):** the REST contract `CreatePartitionRecipeRequest.method` is `DIRICHLET | IID | EXPLICIT` (04-API §8.2), while the framework/DB `partitioner` enum is `DIRICHLET_LABEL | DIRICHLET_QTY | SHARD | NATURAL` (03-DATA-MODEL §5.1). The control-plane `DatasetService` maps `DIRICHLET → DIRICHLET_LABEL`, `IID → DIRICHLET_QTY` with a large `alpha` (or a dedicated IID flag), and `EXPLICIT → NATURAL` with the explicit map stored client-side. **Flagged as the one place the two contracts use different vocabularies** — the mapping table is the single source of truth; do not invent a third vocabulary.

### 5.2 `DataSchema` and `SchemaFingerprint`

`framework/src/fedlearn/data/schema.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class Modality(str, Enum):
    TABULAR    = "TABULAR"
    IMAGE      = "IMAGE"
    TEXT       = "TEXT"
    TIMESERIES = "TIMESERIES"

@dataclass(frozen=True)
class DataSchema:
    """The declared local-data contract. Hashes to schema_hash (compatibility key)."""
    modality: Modality
    feature_shape: tuple[int, ...]      # e.g. (1, 28, 28) or (140,)
    feature_dtype: str                  # whitelist: "float32","float64","int32","int64","uint8","bool"
    label_set: tuple[int, ...]          # observed class ids, SORTED ascending
    num_classes: int

    def schema_hash(self) -> str:
        """sha256 of the canonical JSON of (modality, feature_shape, feature_dtype,
        sorted(label_set), num_classes). Compatibility key, NOT a uniqueness key
        (C2 §6: two clients with same schema, different rows, share schema_hash)."""
        ...

@dataclass(frozen=True)
class SchemaFingerprint:
    """Non-reversible — SAFE to transmit (C2 §3.4). Carried in gRPC registration."""
    schema_hash: str
    sample_count: int
    label_histogram: dict[int, int]     # {label: count} — metadata only; feeds the non-IID heatmap (C2 §3.4)
```

### 5.3 `PartitionRecipe` and `recipe_hash`

`framework/src/fedlearn/data/recipe.py`:

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class PartitionRecipe:
    """Mirrors the partition_recipes row (03-DATA-MODEL §5.1).

    data_seed is the DATA seed (PCG64), distinct from the optimizer seed (C2 §4.3).
    alpha is None for SHARD/NATURAL.
    """
    partitioner: str            # "DIRICHLET_LABEL"|"DIRICHLET_QTY"|"SHARD"|"NATURAL"
    num_partitions: int         # >= 1
    alpha: float | None         # Dirichlet concentration; None for non-Dirichlet
    data_seed: int              # BIGINT; fed to PCG64; NEVER the global RNG
    dataset_version_content_hash: str   # the dataset_versions.content_hash this recipe binds to

    def recipe_hash(self) -> str:
        """sha256(partitioner | num_partitions | alpha | data_seed | dataset_version_content_hash)
        over canonical JSON. This is the reproducibility key that REPLACES the v1
        pickle filename ecg_clients{N}_alpha{a}_frac{f}_seed{s}.pkl (C2 §2.3, §4.2)."""
        ...
```

### 5.4 `ArtifactStore` (content-addressed object store client)

`framework/src/fedlearn/artifacts/store.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class ArtifactKind(str, Enum):
    INITIAL    = "INITIAL"
    CHECKPOINT = "CHECKPOINT"
    FINAL      = "FINAL"

@dataclass(frozen=True)
class StoredArtifact:
    sha256: str
    storage_uri: str            # s3://<bucket>/<sha256>
    size_bytes: int
    kind: ArtifactKind

class ArtifactStore:
    """Content-addressed model store over S3/MinIO (boto3).

    Key invariant: the object KEY is the sha256 of the bytes, so identical
    bytes are stored once (dedup) and every artifact is immutable (C2 §4.2,
    C3 §5.5). See §6.4 for the key scheme.
    """

    def __init__(self, bucket: str, endpoint_url: str | None = None,
                 region: str = "us-east-1") -> None:
        """endpoint_url is set for MinIO (e.g. http://minio:9000); None for AWS S3."""
        ...

    @staticmethod
    def key_for(sha256: str) -> str:
        """Return the object key for a hash. See §6.4 (sharded prefix)."""
        ...

    def put_model(self, blob: bytes, kind: ArtifactKind) -> StoredArtifact:
        """Compute sha256(blob); PUT to key_for(sha256) ONLY if it does not
        already exist (idempotent, dedup). Returns the StoredArtifact."""
        ...

    def get_model(self, sha256: str) -> bytes:
        """GET the object; verify the recomputed sha256 matches before returning
        (integrity / corruption defense, C2 §3.5). Raise ArtifactCorruptError on mismatch."""
        ...

    def exists(self, sha256: str) -> bool:
        """HEAD the object key; True if present (dedup short-circuit)."""
        ...
```

> **Pre-signed path (the production default):** the FL server does NOT hold long-lived object-store credentials. It requests a pre-signed PUT via `POST /api/artifacts/upload-url` (`ArtifactUploadUrlResponse`, 04-API §9), PUTs the blob directly to the URL, then registers metadata via `POST /api/artifacts`. `ArtifactStore` with direct credentials is for the dev/`LOCAL_PROCESS` profile and the backend's own server-side maintenance. `presigned.py` implements the brokered path so blob bytes never transit the JVM (04-API §9, 02-TECH-STACK §7).

### 5.5 `DeterminismManifest`

`framework/src/fedlearn/artifacts/manifest.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class DeterminismManifest:
    """The reproducibility contract, computed once per run at server startup
    (C3 §5.2). Persisted to determinism_manifests (03-DATA-MODEL §5.3) and
    logged to MLflow as params/tags (C3 §5.4)."""
    framework_git_sha: str          # 40-hex git commit of the framework
    proto_version: str              # "fedlearn.v2"
    torch_version: str              # e.g. "2.12.0" — pins CPU-canonical RNG parity (C3-1)
    numpy_version: str              # e.g. "1.26.4"
    torch_cuda_version: str | None  # None when CPU-only
    rng_device: str                 # MUST be "cpu" (C3 §5.1; DB CHECK enforces this)
    rng_engine: str                 # "torch.Generator(cpu)"
    use_deterministic_algorithms: bool
    seed: int                       # the OPTIMIZER/perturbation seed (NOT the data seed)
    initial_model_sha256: str | None
    dataset_split_sha256: str | None   # == the recipe-derived partition npz sha256
    dataset_version_id: str | None
    partition_recipe_id: str | None
    golden_vector_sha256: str | None   # RNG golden-vector fixture hash (04-API §4.4)
    platform_os: str | None            # "linux"
    platform_arch: str | None          # "x86_64" | "arm64"

    @staticmethod
    def build(*, seed: int, dataset_version_id: str | None,
              partition_recipe_id: str | None,
              initial_model_sha256: str | None,
              dataset_split_sha256: str | None,
              golden_vector_sha256: str | None) -> "DeterminismManifest":
        """Fill version/platform fields from the live environment
        (torch.__version__, numpy.__version__, platform.system()/machine(),
        the framework git sha) and force rng_device='cpu'."""
        ...

    def to_json(self) -> str:
        """Canonical JSON (sorted keys) matching the §6.6 schema; this exact
        string is what goes into determinism_manifests.manifest_json."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
```

### 5.6 Backend service signatures (Java, abbreviated)

```java
// DatasetService.java — content-hash uniqueness is the core invariant.
public interface DatasetService {
    DatasetDto createDataset(UUID orgId, Long createdBy, CreateDatasetRequest req);
    DatasetVersionDto createVersion(UUID datasetId, CreateDatasetVersionRequest req); // 409 VERSION_SHA_EXISTS on dup
    PartitionRecipeDto createRecipe(UUID datasetVersionId, CreatePartitionRecipeRequest req); // recipe_hash dedup
}

// ArtifactService.java — pre-signed URL + register-by-sha256 dedup.
public interface ArtifactService {
    ArtifactUploadUrlResponse mintUploadUrl(UUID orgId, ArtifactUploadUrlRequest req); // pre-signed PUT, TTL ~15 min
    ArtifactDto register(UUID orgId, RegisterArtifactRequest req);  // 409 SHA_EXISTS on (org_id, sha256) dup
    PresignedDownload mintDownloadUrl(UUID artifactId, Long callerUserId); // ORG_MEMBER(artifact.org) check
}
```

### 5.7 DB row types this unit owns (exact, from 03-DATA-MODEL §5)

| Table | Key columns (type) | Content-hash column | FK lineage |
|---|---|---|---|
| `datasets` | `id UUID` PK, `org_id UUID NOT NULL`, `name VARCHAR(255)`, `modality VARCHAR(32)` | — | `org_id → organizations`, `created_by → users(BIGINT)` |
| `dataset_versions` | `id UUID` PK, `dataset_id UUID`, `version INT`, `schema_json JSONB`, `sample_count BIGINT` | `content_hash CHAR(64)` — `UNIQUE(dataset_id, content_hash)` | `dataset_id → datasets` (CASCADE) |
| `partition_recipes` | `id UUID` PK, `dataset_version_id UUID`, `partitioner VARCHAR(32)`, `num_partitions INT`, `alpha DOUBLE PRECISION`, `data_seed BIGINT NOT NULL` | `recipe_hash CHAR(64)` — `UNIQUE(dataset_version_id, recipe_hash)` | `dataset_version_id → dataset_versions` (CASCADE) |
| `model_artifacts` | `id UUID` PK, `org_id UUID NOT NULL`, `storage_uri VARCHAR(512)`, `size_bytes BIGINT`, `kind VARCHAR(16)`, `round_idx INT` | `sha256 CHAR(64)` — `UNIQUE(org_id, sha256)` | `org_id → organizations`, `fl_run_id → fl_runs(nullable, SET NULL)` |
| `determinism_manifests` | `id UUID` PK, `fl_run_id UUID UNIQUE NOT NULL`, version/seed columns, `manifest_json JSONB NOT NULL` | `initial_model_sha256 CHAR(64)`, `dataset_split_sha256 CHAR(64)` | `fl_run_id → fl_runs` (CASCADE, 1:1) |

---

## 6. Core algorithms & flows

### 6.1 Dirichlet label-skew partition (the salvaged algorithm, isolated RNG)

Ported verbatim from the **good** `default_rng` version (C2 §4.3 "salvage `ecg_loader.py:30-71`"). The critical change from the v1 forks is the **isolated generator** — never `np.random.seed` / `np.random.shuffle` (the v1 `client.py:248-270` fork used the global state and clobbered DeComFL's seed; C2 §2.2).

```python
def partition(self, labels: np.ndarray, recipe: PartitionRecipe) -> list[np.ndarray]:
    # ISOLATED generator — the data-seed namespace (C2 §3.3/§4.3). Never global.
    rng = np.random.Generator(np.random.PCG64(recipe.data_seed))
    n = recipe.num_partitions
    classes = np.unique(labels)
    # idx_by_class[c] = shuffled indices of all samples with label c
    idx_by_class = {int(c): rng.permutation(np.where(labels == c)[0]) for c in classes}

    parts: list[list[int]] = [[] for _ in range(n)]
    for c in classes:
        idx_c = idx_by_class[int(c)]
        # Draw the class's split proportions across the n partitions.
        proportions = rng.dirichlet(np.repeat(recipe.alpha, n))
        # Cumulative cut points; np.split deterministically given the same rng draw.
        cuts = (np.cumsum(proportions)[:-1] * len(idx_c)).astype(int)
        for pid, chunk in enumerate(np.split(idx_c, cuts)):
            parts[pid].extend(chunk.tolist())

    # Sort each partition's indices for a CANONICAL, hashable ordering (so the
    # npz sha256 is stable across runs) and return as int64 arrays.
    return [np.array(sorted(p), dtype=np.int64) for p in parts]
```

> **Determinism proof obligation (CI gate, C2 §3.3):** `DirichletLabelPartitioner(recipe).partition(labels, recipe)` MUST be byte-identical across runs and platforms for a fixed `(labels, recipe)`. The golden-partition test (§10) checks the sha256 of the concatenated, sorted index arrays against a committed reference. Sorting the indices is what makes the hash stable regardless of `extend` insertion order.

### 6.2 Content-addressed npz partition format (replaces the pickle cache)

The v1 pickle cache (`ecg_loader.py:105-152`) is **deleted** — it is both an arbitrary-code-execution risk and the cause of the stale-split trap (C2 §2.3, §3.5). Replacement: `np.savez_compressed` + a sidecar sha256, verified before load.

```python
# npz_store.py
def write_partition(out_dir: str, recipe: PartitionRecipe,
                    partition_indices: list[np.ndarray]) -> str:
    """Write one .npz holding all partition index arrays, return its sha256.
    File name IS the content hash (content-addressed): <sha256>.npz."""
    buf = io.BytesIO()
    # Keys "p0","p1",... are deterministic; np.savez_compressed is reproducible
    # for the same arrays. Do NOT include timestamps in the archive.
    np.savez_compressed(buf, **{f"p{i}": a for i, a in enumerate(partition_indices)})
    blob = buf.getvalue()
    digest = hashlib.sha256(blob).hexdigest()
    path = os.path.join(out_dir, f"{digest}.npz")
    with open(path, "wb") as f:
        f.write(blob)
    with open(path + ".sha256", "w") as f:    # sidecar checksum
        f.write(digest)
    return digest

def load_partition(path: str) -> list[np.ndarray]:
    """Verify the sidecar sha256 BEFORE np.load (defends C2 RCE + detects
    corruption). np.load(allow_pickle=False) — never unpickle objects."""
    with open(path, "rb") as f:
        blob = f.read()
    expected = open(path + ".sha256").read().strip()
    if hashlib.sha256(blob).hexdigest() != expected:
        raise PartitionChecksumError(path)
    npz = np.load(io.BytesIO(blob), allow_pickle=False)  # allow_pickle=False is mandatory
    return [npz[f"p{i}"] for i in range(len(npz.files))]
```

> **`dataset_split_sha256` lineage anchor:** the npz `digest` above is exactly the value written to `determinism_manifests.dataset_split_sha256` (03-DATA-MODEL §5.3) and `DeterminismManifestDto.datasetSha256` (04-API §4.4). One hash, one definition.

### 6.3 Dataset-version content hash (metadata only — raw data never seen)

Raw training data is client-private and the platform never sees it (the two-plane invariant, C2 §3.1). So `dataset_versions.content_hash` hashes the **canonical schema + aggregate-stats manifest**, not raw rows (C2 §6 "content hash of client-private data").

```python
# content_hash.py
def dataset_version_content_hash(schema: DataSchema, sample_count: int,
                                 label_histogram: dict[int, int]) -> str:
    """sha256 over the canonical JSON of the schema + aggregate stats. This is a
    COMPATIBILITY/version key, not a per-row uniqueness key (C2 §6, flagged)."""
    payload = {
        "schema": {
            "modality": schema.modality.value,
            "feature_shape": list(schema.feature_shape),
            "feature_dtype": schema.feature_dtype,
            "label_set": sorted(schema.label_set),
            "num_classes": schema.num_classes,
        },
        "sample_count": sample_count,
        "label_histogram": {str(k): label_histogram[k] for k in sorted(label_histogram)},
    }
    return sha256_hex(canonical_json(payload))

def canonical_json(obj) -> bytes:
    # sorted keys, no whitespace, UTF-8 — the SINGLE canonicalization used everywhere.
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()
```

### 6.4 S3 / MinIO key scheme (locked)

| Concern | Decision | Reasoning |
|---|---|---|
| Bucket (models) | `fedlearn-artifacts` (configurable via `FEDLEARN_ARTIFACT_BUCKET`) | Matches 03-DATA-MODEL §5.2 example `s3://fedlearn-artifacts/<sha256>`. |
| Bucket (datasets) | `fedlearn-data` (configurable via `FEDLEARN_DATA_BUCKET`) | Matches 04-API §8.2 example `s3://fedlearn-data/<sha256>`. |
| Object key | **sharded prefix**: `<sha256[0:2]>/<sha256[2:4]>/<sha256>` | Two 1-byte hex shards spread keys so a flat namespace does not hotspot one prefix; the full sha256 is still the immutable name. |
| `storage_uri` stored in DB | `s3://<bucket>/<sha256[0:2]>/<sha256[2:4]>/<sha256>` | Exactly reconstructable from `sha256` + bucket; written to `model_artifacts.storage_uri` / `dataset_versions.uri`. |
| Tenancy | one bucket, key prefix is the hash (NOT the org) | Dedup is global by bytes; tenant scoping is enforced by the DB `UNIQUE(org_id, sha256)` row + the `ORG_MEMBER` authz on the download-url endpoint (04-API §9), not by the object key. Bytes are opaque and access is brokered. |
| Immutability | objects are write-once; `put_model` is a no-op if `exists(sha256)` | Content addressing makes overwrite meaningless (same bytes ⇒ same key). |

```python
@staticmethod
def key_for(sha256: str) -> str:
    if len(sha256) != 64:
        raise ValueError(f"expected 64-hex sha256, got {len(sha256)}")
    return f"{sha256[0:2]}/{sha256[2:4]}/{sha256}"
```

> **Reasoning — content addressing.** v1's "version key" was a pickle filename whose params (N/alpha/frac/seed) could stay fixed while the data changed, silently reusing a stale split (C2 §2.3). Keying every blob on the sha256 of its bytes makes the artifact immutable by construction, deduplicated across runs/orgs, and integrity-checkable on read. This is the same content-addressed-snapshot pattern as DVC/lakeFS, scaled to metadata-only because raw data is client-private (C2 §3.2).

### 6.5 End-to-end: run start → per-round checkpoint → resume

ASCII sequence (the artifact-plane slice only; the FL aggregation/lease are owned elsewhere):

```
FL server (run start)        ArtifactStore / control plane            PostgreSQL / MLflow
─────────────────────        ─────────────────────────────           ───────────────────
1. build DeterminismManifest
   (seed, torch_ver, git_sha,
    dataset_split_sha256,
    initial_model_sha256)
2. hash initial model bytes ──► POST /api/artifacts/upload-url ──────► (no-op if exists)
   sha256(safetensors blob)     (pre-signed PUT, key = sha256)
3. PUT blob to pre-signed URL ─► S3/MinIO  <bucket>/<shard>/<sha256>
4. POST /api/artifacts ─────────► register model_artifacts row ──────► INSERT model_artifacts
   (sha256, kind=INITIAL)                                              (kind=INITIAL, org dedup)
5. log_run_start(manifest) ─────────────────────────────────────────► MLflow params/tags
6. backend persists manifest ───────────────────────────────────────► INSERT determinism_manifests
                                                                       (1:1 fl_runs, manifest_json)
   ── per round r ──
7. aggregate (owned elsewhere)
8. hash checkpoint bytes,
   PUT (steps 2-4, kind=CHECKPOINT, round_idx=r)
9. POST /api/internal/runs/{runId}/checkpoint  (CheckpointReportDto: round, artifactId, sha256, sizeBytes)
10. log_round_metrics(r, loss, accuracy, uplink_bytes, scalars_transmitted) ─► MLflow + round_results

   ── resume (server restart / late client) ──
R1. read fl_runs.round_idx + the latest model_artifacts(kind=CHECKPOINT, max round_idx)
R2. GET /api/artifacts/{artifactId}/download-url  (pre-signed GET)
R3. ArtifactStore.get_model(sha256)  ── verifies sha256 on read ──► resume from checkpoint
```

> **Reasoning — per-round content-addressed checkpoint = resumability.** v1 had a destructive in-place model save and no run entity, so a restart lost everything (C3 §4.1, R14/R16). Writing every round's model as an immutable `<sha256>` object, pointed to by a `model_artifacts(kind=CHECKPOINT, round_idx=r)` row FK'd to `fl_runs`, lets a restarted server (or a rebuilding DeComFL client) resume from the durable record (04-API §4.4 reasoning). The DeComFL path additionally rebuilds from `RebuildHistory` seeds (gRPC, 04-API §10.2) — orthogonal to this; the checkpoint is the FedAvg-path and audit anchor.

### 6.6 Determinism-manifest JSON schema (the exact on-disk / on-wire shape)

This is the value of `determinism_manifests.manifest_json` (03-DATA-MODEL §5.3) and the source of `DeterminismManifestDto` (04-API §4.4). It is a superset/structured form of the C3 §5.2 manifest. The local model emits **exactly** these keys (canonical JSON, sorted keys):

```json
{
  "framework_git_sha": "abc1234def5678901234567890abcdef12345678",
  "proto_version": "fedlearn.v2",
  "torch_version": "2.12.0",
  "numpy_version": "1.26.4",
  "torch_cuda_version": null,
  "rng_device": "cpu",
  "rng_engine": "torch.Generator(cpu)",
  "use_deterministic_algorithms": false,
  "seed": 42,
  "strategy": "DeComFL",
  "hyperparameters": {
    "learningRate": 0.001,
    "mu": 0.001,
    "numPerturbations": 10,
    "numLocalSteps": 5,
    "gradEstimateMethod": "forward"
  },
  "dataset_version_id": "0f1c2d3e-4a5b-6c7d-8e9f-0a1b2c3d4e5f",
  "dataset_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "partition_recipe_id": "1a2b3c4d-5e6f-7081-92a3-b4c5d6e7f809",
  "dataset_split_sha256": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
  "initial_model_sha256": "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae",
  "golden_vector_sha256": "fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04fae5511b68fbf8fb9",
  "platform": { "os": "linux", "arch": "x86_64" }
}
```

| Key | Maps to DB column / DTO field | Reasoning |
|---|---|---|
| `seed` | `determinism_manifests.seed`; `DeterminismManifestDto.seed` | The **optimizer/perturbation** seed (CPU-canonical RNG), C3 §5.1. Distinct from `data_seed`. |
| `rng_device` | `determinism_manifests.rng_device` (DB `CHECK = 'cpu'`) | CPU-canonical invariant — the single highest-impact C3 fix (C3-1, §5.1). |
| `torch_version` | `determinism_manifests.torch_version`; `DeterminismManifestDto.torchVersion` | A federation refuses to mix torch versions (C3 §6.3). |
| `dataset_split_sha256` | `determinism_manifests.dataset_split_sha256`; `DeterminismManifestDto.partitionRecipeId` neighbor | The npz partition hash from §6.2 — lineage anchor (C3 §5.2). |
| `initial_model_sha256` | `determinism_manifests.initial_model_sha256`; `DeterminismManifestDto.modelInitSha256` | The model the run started from — lineage anchor (C3 §5.2). |
| `golden_vector_sha256` | `DeterminismManifestDto.goldenVectorSha256` | The RNG golden-vector fixture hash the client validates against (04-API §4.4, §10.2). |

> **Two-seed namespace, restated (load-bearing).** `determinism_manifests.seed` and `fl_runs.config -> 'seed'` are the **optimizer** seed (DeComFL perturbations, CPU-canonical `torch.Generator`). `partition_recipes.data_seed` is the **data** seed (`np.random.PCG64`). They live in different rows so re-partitioning data does not perturb the optimizer RNG and vice versa (C2 §4.3, C3 §5.1). The manifest records the optimizer seed and references the recipe (which carries the data seed) by id — never collapses them.

---

## 7. Data it owns

### 7.1 Tables (03-DATA-MODEL §5; this unit owns the rows, the orchestration build authors the DDL)

| Table | Migration | Ownership note |
|---|---|---|
| `datasets` | `V6__dataset_registry.sql` (03-DATA-MODEL §5.1) | Lineage root; `UNIQUE(org_id, name)`; `org_id NOT NULL` (rule R-C). |
| `dataset_versions` | `V6` | Immutable; `content_hash CHAR(64)`, `UNIQUE(dataset_id, content_hash)`; `schema_json JSONB`. |
| `partition_recipes` | `V6` | `data_seed BIGINT NOT NULL`; `recipe_hash CHAR(64)`, `UNIQUE(dataset_version_id, recipe_hash)`. |
| `model_artifacts` | `V7__fl_runs_and_artifacts.sql` (03-DATA-MODEL §5.2) | `sha256 CHAR(64)`, `UNIQUE(org_id, sha256)`; `kind ∈ {INITIAL,CHECKPOINT,FINAL}`; `fl_run_id` nullable FK SET NULL; `round_idx` set for CHECKPOINT. |
| `determinism_manifests` | `V8__determinism_manifest.sql` (03-DATA-MODEL §5.3) | 1:1 with `fl_runs` (`fl_run_id UNIQUE NOT NULL`); `rng_device CHECK = 'cpu'`; `manifest_json JSONB NOT NULL`. |

> **This unit does NOT own** `fl_runs`, `round_results`, `projects`, or the identity tables — it only **reads** lineage FKs from `fl_runs` (`dataset_version_id`, `partition_recipe_id`, `initial_model_artifact_id`, `final_model_artifact_id`, `mlflow_run_id`) and **writes** `model_artifacts.fl_run_id` / `determinism_manifests.fl_run_id`.

### 7.2 In-memory structures (Python)

| Structure | Lifetime | Purpose |
|---|---|---|
| `list[np.ndarray]` partition index arrays | transient, per partition build | Output of `Partitioner.partition`; serialized to npz then discarded. |
| `PartitionRecipe` (frozen dataclass) | per run | The recipe pinned by the run; passed to `HfDataSource`. |
| `DeterminismManifest` (frozen dataclass) | per run, built once at start | Serialized to JSON, persisted, logged to MLflow; not mutated after build. |
| `ArtifactStore` (boto3 client) | process lifetime | Reused connection; thread-safe per boto3 session-per-thread guidance. |
| In-process LRU of `sha256 → bool` for `exists()` | optional, bounded | Avoids repeated HEAD on the same hash within one run; bounded to avoid memory pressure. |

### 7.3 Object-store buckets

| Bucket | Holds | Key scheme (§6.4) |
|---|---|---|
| `fedlearn-artifacts` | model bytes (INITIAL/CHECKPOINT/FINAL), safetensors | `<sha256[0:2]>/<sha256[2:4]>/<sha256>` |
| `fedlearn-data` | content-addressed dataset blobs (registered, not uploaded through the API) | same |

---

## 8. Configuration & environment variables

| Env var | Type | Default | Profile / mode | Purpose |
|---|---|---|---|---|
| `FEDLEARN_ARTIFACT_BUCKET` | string | `fedlearn-artifacts` | all | Model-artifact bucket name. |
| `FEDLEARN_DATA_BUCKET` | string | `fedlearn-data` | all | Dataset-blob bucket name. |
| `FEDLEARN_S3_ENDPOINT_URL` | string (URL) | unset (= AWS S3) | self-host | MinIO endpoint, e.g. `http://minio:9000`; unset means real AWS S3. |
| `FEDLEARN_S3_REGION` | string | `us-east-1` | all | AWS region; required even for MinIO (SDK demands a region). |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | string | from instance role / secrets manager | dev / `LOCAL_PROCESS` only | Direct creds; production uses pre-signed URLs + instance role, not these. |
| `FEDLEARN_DATA_SEED` | int (BIGINT) | none — MUST come from `partition_recipes.data_seed` | all | The data-partition seed; never the optimizer seed. |
| `FEDLEARN_PARTITION_CACHE_DIR` | string (path) | `~/.cache/fedlearn/partitions` | client | Where content-addressed `<sha256>.npz` partitions are written/read. |
| `FEDLEARN_BACKEND_URL` | string (URL) | injected by launcher (04-API §13) | all | Base URL for `/api/artifacts/**` and `/api/internal/**`. |
| `FEDLEARN_RUN_TOKEN` | string | injected by launcher (04-API §13) | all | Per-run scoped token for the artifact-register and checkpoint callbacks. |
| `MLFLOW_TRACKING_URI` | string (URL) | unset | all | MLflow server, e.g. `http://mlflow:5000`; absent ⇒ MLflow logging is skipped (best-effort). |
| `MLFLOW_S3_ENDPOINT_URL` | string (URL) | = `FEDLEARN_S3_ENDPOINT_URL` | self-host | MLflow's own artifact backend (it reuses the same MinIO/S3). |

Backend (Spring `application.properties` / profile-scoped):

| Property | Type | Default | Profile | Purpose |
|---|---|---|---|---|
| `app.artifact.bucket` | string | `fedlearn-artifacts` | all | Mirrors `FEDLEARN_ARTIFACT_BUCKET`. |
| `app.artifact.s3-endpoint` | string | unset (AWS S3) | self-host | MinIO endpoint. |
| `app.artifact.presign-ttl-seconds` | int | `900` (15 min) | all | Pre-signed URL lifetime; long enough for a large-model PUT, short enough to bound exposure. |
| `app.mlflow.tracking-uri` | string | unset | `ec2demo`/`production` | MLflow link-out for `fl_runs.mlflow_run_id`. |

> **Reasoning — pre-signed-URL TTL = 900 s.** A multi-GB model PUT over a constrained uplink must complete inside the URL's validity; 15 minutes covers a slow large-model upload while keeping the bearer URL short-lived (04-API §9 brokers metadata + pre-signed URLs precisely so blob bytes never transit the JVM).

---

## 9. Error handling & edge cases

`framework/src/fedlearn/data/errors.py` defines a typed hierarchy; the gRPC servicer maps them to status codes per 04-API §10.3.

| # | Failure mode | Where | Exact handling | gRPC mapping (04-API §10.3) |
|---|---|---|---|---|
| 1 | Partition npz checksum mismatch | `load_partition` | `raise PartitionChecksumError(path)`; do **not** `np.load` the bytes | n/a (client-local; abort run, log) |
| 2 | Client schema mismatch vs pinned `dataset_versions.schema_json` | server registration | reject at handshake; `raise SchemaMismatchError(schema_hash)` | `FAILED_PRECONDITION` |
| 3 | Unknown `feature_dtype` (outside whitelist) | `DataSchema` validation | `raise InvalidSchemaError(dtype)` | `INVALID_ARGUMENT` |
| 4 | `alpha` is None but `partitioner ∈ {DIRICHLET_LABEL, DIRICHLET_QTY}` | `PartitionRecipe`/recipe create | reject before partitioning; `400 VALIDATION_FAILED` (REST) | `INVALID_ARGUMENT` |
| 5 | `num_partitions < 1` | recipe create | DB `CHECK(num_partitions >= 1)` + app validation; `400 VALIDATION_FAILED` | — |
| 6 | Empty partition (a client gets 0 indices under extreme low `alpha`) | `partition` | allow empty array but **warn**; `train_loader` on an empty partition raises `EmptyPartitionError` at fit time | `FAILED_PRECONDITION` |
| 7 | Duplicate dataset-version content hash | `DatasetService.createVersion` | `UNIQUE(dataset_id, content_hash)` violation → `409 VERSION_SHA_EXISTS` | — |
| 8 | Duplicate artifact sha256 in org | `ArtifactService.register` | `UNIQUE(org_id, sha256)` violation → treat as success-idempotent OR `409 SHA_EXISTS` per 04-API §9 | — |
| 9 | Object-store PUT fails mid-upload (network) | `ArtifactStore.put_model` / pre-signed PUT | retry with exponential backoff (3 attempts); on final failure do **not** register the row (no orphan metadata) | n/a |
| 10 | `get_model` recomputed sha256 ≠ requested | `ArtifactStore.get_model` | `raise ArtifactCorruptError(sha256)`; never return corrupt bytes | n/a |
| 11 | MLflow unreachable | `mlflow_lineage.*` | best-effort: log a WARN, continue the run; MLflow is lineage, not the system of record (the DB row is). Never crash the run on telemetry failure (mirrors 04-API §5 "best-effort callback") | n/a |
| 12 | Checkpoint callback to control plane fails | `presigned.py` / checkpoint POST | best-effort, short timeout, try/except; a telemetry failure must not crash the run (04-API §5 reasoning) | n/a |
| 13 | `data_seed` collides semantically with optimizer seed | recipe create | enforce the two-namespace rule in code review + the manifest schema; the DB stores them in different columns/tables (C2 §4.3) | — |
| 14 | npz `allow_pickle` accidentally True | `load_partition` | hard-code `allow_pickle=False`; a unit test asserts it (defends the v1 RCE class, C2 §3.5) | — |
| 15 | `NATURAL` partition reproducibility expectation | docs + recipe | record `partitioner=NATURAL`; surface to the user that reproducibility = "same recipe + declared schema", not bit-exact (C2 §6 open question) | — |

> **Reasoning — fail-closed schema admission (#2).** v1 had no dataset descriptor on the wire, so a client could join with wrong-shaped/wrong-labeled data and silently poison aggregation (C2 §2.6). Comparing the client's `schema_hash` against the project's pinned `dataset_versions.schema_json` and rejecting on mismatch (`FAILED_PRECONDITION`) is a one-time O(1) handshake cost that closes the gap and adds **zero per-round communication** — critical so it does not undermine DeComFL's O(K·P) wedge (C2 §3.6).

---

## 10. Testing strategy

**Frameworks:** Python `pytest` (framework unit/golden tests, runs without GPU — the project conventions `framework` commands); Java JUnit + **Testcontainers-PostgreSQL** for the registry repositories (03-DATA-MODEL §6 mandates a Testcontainers-PG profile because the `test` profile disables Flyway). Object-store tests use a MinIO Testcontainer or `moto`/`localstack` for `boto3`.

| Test (named) | Asserts |
|---|---|
| `test_dirichlet_label_partition_deterministic` | `DirichletLabelPartitioner.partition(labels, recipe)` returns byte-identical index arrays across two calls with the same `data_seed`; sha256 of the concatenated sorted arrays equals a committed golden value (C2 §3.3 golden-partition gate). |
| `test_partition_isolated_rng` | Calling `partition` does **not** mutate `np.random.get_state()` (proves no global-RNG use; defends C2 §2.2 / prior M5). |
| `test_partition_union_complete` | The union of all partition index arrays equals `range(len(labels))` with no overlap (no lost/duplicated samples). |
| `test_recipe_hash_stable` | `PartitionRecipe.recipe_hash()` is stable across runs and changes when any of `{partitioner, num_partitions, alpha, data_seed, dataset_version_content_hash}` changes (replaces the pickle filename, C2 §4.2). |
| `test_npz_roundtrip_and_checksum` | `write_partition` then `load_partition` returns identical arrays; corrupting one byte makes `load_partition` raise `PartitionChecksumError` (C2 §3.5). |
| `test_npz_rejects_pickle` | `load_partition` uses `allow_pickle=False`; a pickled-object npz raises rather than executing (defends the v1 RCE, C2 §2.1). |
| `test_dataset_version_content_hash_metadata_only` | The hash is computed from schema+stats only; two `DataSchema`s with the same fields but constructed separately produce the same hash (compatibility key, C2 §6). |
| `test_artifact_store_dedup` | `put_model(blob)` twice writes the object once (`exists()` short-circuits) and both return the same `sha256`/`storage_uri` (C2 §4.2). |
| `test_artifact_store_integrity` | `get_model` raises `ArtifactCorruptError` when the stored bytes are tampered (sha256 mismatch). |
| `test_key_for_sharded` | `key_for(sha256)` returns `<2>/<2>/<64-hex>` and rejects a non-64-hex input (§6.4). |
| `test_manifest_json_schema` | `DeterminismManifest.to_json()` contains exactly the §6.6 keys, `rng_device == "cpu"`, sorted keys, and round-trips into `manifest_json`. |
| `test_schema_mismatch_rejected` | A client whose `schema_hash` ≠ the pinned version's hash is rejected with the `SchemaMismatchError` → `FAILED_PRECONDITION` mapping (C2 §2.6). |
| `DatasetVersionRepositoryIT` (Testcontainers-PG) | `UNIQUE(dataset_id, content_hash)` raises on a duplicate; `schema_json` JSONB round-trips via `@JdbcTypeCode(SqlTypes.JSON)` (03-DATA-MODEL §7). |
| `ModelArtifactRepositoryIT` (Testcontainers-PG) | `UNIQUE(org_id, sha256)` enforces per-tenant dedup; `kind` enum maps to the `CHECK` set exactly. |
| `DeterminismManifestRepositoryIT` (Testcontainers-PG) | `fl_run_id UNIQUE NOT NULL` enforces 1:1 with `fl_runs`; inserting `rng_device != 'cpu'` violates the `CHECK`. |

> **Cross-language note (flagged, C3 §9 risk 2):** the RNG **golden-vector parity** test (Python ↔ C++ mobile) is owned by the FL framework strategies LLD, not this unit. This unit owns the **partition** golden test and the **manifest** schema test. Do not duplicate the perturbation-RNG test here.

---

## 11. Build & run (this unit in isolation)

```bash
# --- Python data + artifact modules (framework) ---
cd framework
pip install -e .                      # installs torch 2.12.0, safetensors, datasets, numpy, boto3, mlflow
pytest tests/test_partitioner.py tests/test_npz_store.py \
       tests/test_artifact_store.py tests/test_manifest.py   # the §10 Python tests

# --- Stand up a local MinIO + MLflow for integration (dev only) ---
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  quay.io/minio/minio:<pinned-digest> server /data --console-address ":9001"
export FEDLEARN_S3_ENDPOINT_URL=http://localhost:9000
export FEDLEARN_S3_REGION=us-east-1
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export FEDLEARN_ARTIFACT_BUCKET=fedlearn-artifacts FEDLEARN_DATA_BUCKET=fedlearn-data
# create buckets:
python -c "import boto3,os; c=boto3.client('s3',endpoint_url=os.environ['FEDLEARN_S3_ENDPOINT_URL']); \
[c.create_bucket(Bucket=b) for b in ('fedlearn-artifacts','fedlearn-data')]"

# --- Backend registry repositories against real Postgres (Testcontainers profile) ---
cd backend/fl-platform-api
SPRING_PROFILES_ACTIVE=test ./gradlew test \
  --tests "com.federated.fl_platform_api.dataplane.*IT"   # Testcontainers-PG; NOT the H2 test profile

# --- Verify the Flyway chain V1->V8 against real Postgres 17.10 (03-DATA-MODEL §6 mandate) ---
# (run via the Testcontainers-PG CI profile; the `test` profile disables Flyway and must stay that way)
```

**Verification checklist (done-conditions):**
1. `pytest` green for partitioner determinism + npz checksum + artifact dedup + manifest schema.
2. A partition written then loaded round-trips and rejects a corrupted file.
3. `key_for(sha256)` produces the sharded key and `put_model` deduplicates.
4. The `determinism_manifests` row's `manifest_json` matches §6.6 and `rng_device='cpu'`.
5. Testcontainers-PG repositories enforce all three uniqueness constraints.

---

## 12. Reasoning & alternatives (what was rejected and why)

| Decision | Rejected alternative | Why (audit-tied) |
|---|---|---|
| Own the Dirichlet split (`DataSource`/`Partitioner`) | Keep `flwr-datasets` | Violates the "no Flower" invariant, is bundle bloat, and is the only thing implementing partitioning today; removing it is the act of taking ownership (C2 §3.3, B2 §2.3, 02-TECH-STACK §4.3). |
| Single `Partitioner` interface, four concrete partitioners | Four forked `dirichlet_split` copies with three RNG strategies | The forks drift and one used the global RNG, clobbering DeComFL's seed; collapse to one with an isolated `PCG64` generator (C2 §2.2, §4.3). |
| Content-addressed npz + sidecar sha256 | Pickle split cache (`*.pkl`) | Pickle is an arbitrary-code-execution vector and the cause of the stale-split trap; npz + checksum is safe, versioned, and integrity-checked (C2 §2.3, §3.5, 02-TECH-STACK §4.2). |
| Content-addressed (sha256) model store, key = hash | DB blobs / bare filesystem path (`Project.modelPath`) | "Models never belong in Postgres rows" (B2 §5); a bare path has no integrity, no versioning, no dedup (C3 §5.5). |
| Pre-signed PUT/GET, blob bytes never transit the JVM | Stream blobs through the Spring Boot request path | v1's in-JVM `getvalue()`/slice doubled memory; multi-GB models would blow the heap (04-API §9, 02-TECH-STACK §7). |
| Two seed namespaces (`data_seed` vs optimizer `seed`) | One global `seed=42` (v1) | The v1 seed was doubly load-bearing — re-partitioning data perturbed the optimizer RNG and vice versa (C2 §2.3, §4.3, C3 §5.1). |
| `dataset_versions.content_hash` over schema+stats, not raw rows | Hash raw data | Raw training data is client-private and the platform never sees it (the two-plane invariant, C2 §3.1, §6). |
| safetensors for model bytes | `torch.save` / pickle blob | Eliminates the pickle code-execution risk and the v1 `KeyError: 'parameters'` chunk-asymmetry foot-gun (B2 §3.4, 04-API §10.1). |
| MLflow self-hosted, best-effort, DB row is system of record | Weights & Biases (SaaS); or MLflow as the system of record | W&B is not data-resident and not free at scale (B2 §5); MLflow is `$0`, in-VPC, HIPAA-friendly. MLflow being best-effort means a telemetry outage never crashes a run. |
| Sharded key prefix `<2>/<2>/<64>` | Flat `<sha256>` key | Spreads keys to avoid prefix hotspotting at scale while keeping the hash as the immutable name (§6.4). |
| Schema admission gate at handshake (O(1)) | Per-round data validation, or no validation | No validation silently poisons aggregation (C2 §2.6); per-round validation would add data chatter that undermines DeComFL's O(K·P) wedge (C2 §3.6). |
| Counter-based RNG (Philox/Threefry) **not** adopted now | Adopt it for the data partitioner | C3 §5.1 flags counter-based RNG as a later hardening epic that changes perturbation values and needs paper-alignment review; the v2 floor is the isolated `PCG64` data seed + the CPU-canonical torch optimizer seed (C3 §9 risk 5). Flagged as a deferred decision. |

> **Open uncertainty (flagged, C2 §6):** for `NATURAL` partitions (real hospital federations) the platform **cannot** reproduce exact bytes — only the recipe + declared schema. "Reproducibility" there means "same recipe + same schema," not bit-exact replay. This must be communicated to users; do not over-promise bit-exact reproduction for `NATURAL`.

---

## 13. Build task checklist for the ~30B local model (ordered, dependency-respecting)

Each task is one file/feature with a done-condition. Do them in order; later tasks depend on earlier ones.

1. **`content_hash.py`** — implement `canonical_json`, `sha256_hex`, `dataset_version_content_hash`. **Done:** `sha256_hex(canonical_json({"b":2,"a":1}))` is stable and key-order-independent.
2. **`schema.py`** — implement `Modality`, `DataSchema` (+ `schema_hash` using `content_hash`), `SchemaFingerprint`. **Done:** `test_dataset_version_content_hash_metadata_only` passes; `feature_dtype` outside the whitelist raises `InvalidSchemaError`.
3. **`recipe.py`** — implement `PartitionRecipe` + `recipe_hash`. **Done:** `test_recipe_hash_stable` passes.
4. **`errors.py`** — implement the `DataPlaneError` hierarchy (`PartitionChecksumError`, `SchemaMismatchError`, `InvalidSchemaError`, `EmptyPartitionError`, `ArtifactCorruptError`). **Done:** all importable; each carries the offending value.
5. **`partitioner.py`** — implement `Partitioner` ABC + `DirichletLabelPartitioner` (the §6.1 algorithm, isolated `PCG64`). **Done:** `test_dirichlet_label_partition_deterministic`, `test_partition_isolated_rng`, `test_partition_union_complete` pass.
6. **`partitioner.py` (cont.)** — implement `DirichletQtyPartitioner`, `ShardPartitioner`, `NaturalPartitioner`. **Done:** each returns N arrays whose union is complete; `NATURAL` returns the single client's indices.
7. **`npz_store.py`** — implement `write_partition` / `load_partition` (content-addressed, sidecar sha256, `allow_pickle=False`). **Done:** `test_npz_roundtrip_and_checksum`, `test_npz_rejects_pickle` pass.
8. **`source.py`** — implement `DataSource` ABC + `HfDataSource` (load HF dataset, apply recipe via the partitioner + npz cache, build seeded `DataLoader`s, emit `fingerprint()`). **Done:** loaders yield this partition's samples only; `fingerprint()` returns counts/histogram, never raw rows.
9. **`data/__init__.py`** — re-export the public symbols. **Done:** `from fedlearn.data import DataSource, Partitioner, PartitionRecipe, DataSchema, SchemaFingerprint` works.
10. **`artifacts/safetensors_codec.py`** — implement `save_state_dict`/`load_state_dict` + blob sha256. **Done:** round-trips a `state_dict`; no pickle path.
11. **`artifacts/store.py`** — implement `ArtifactStore` (`key_for` §6.4, `put_model` dedup, `get_model` verify, `exists`). **Done:** `test_artifact_store_dedup`, `test_artifact_store_integrity`, `test_key_for_sharded` pass against MinIO/`moto`.
12. **`artifacts/presigned.py`** — implement `PresignedClient` (request `/api/artifacts/upload-url`, PUT, `POST /api/artifacts`, `/download-url`); read `FEDLEARN_BACKEND_URL` + `FEDLEARN_RUN_TOKEN`. **Done:** uploads a blob via a pre-signed URL and registers it; bytes never go through the backend.
13. **`artifacts/manifest.py`** — implement `DeterminismManifest` + `build` + `to_json` (§6.6 schema, `rng_device='cpu'`). **Done:** `test_manifest_json_schema` passes.
14. **`artifacts/mlflow_lineage.py`** — implement `log_run_start`, `log_round_metrics`, `register_final_model` (best-effort; skip if `MLFLOW_TRACKING_URI` unset). **Done:** with MLflow up, a run logs params/metrics and registers a model version tagged with `fl_runs.id`; with MLflow down, the run continues and logs a WARN.
15. **Backend enums** — `Modality`, `PartitionerType`, `ArtifactKind` Java enums matching the `CHECK` sets exactly (03-DATA-MODEL §7). **Done:** `EnumType.STRING` names equal the DB `CHECK` values.
16. **Backend entities** — `Dataset`, `DatasetVersion` (`schema_json` JSONB via `@JdbcTypeCode(SqlTypes.JSON)`), `PartitionRecipe` (`data_seed` Long), `ModelArtifact`, `DeterminismManifest` (`@OneToOne FlRun`, `manifest_json` JSONB). **Done:** JPA boots in `validate` mode against the V6/V7/V8 schema with no mismatch.
17. **Backend repositories + services** — `DatasetService` (content-hash uniqueness → `409 VERSION_SHA_EXISTS`), `ArtifactService` (pre-signed mint, register-by-sha256 dedup → `409 SHA_EXISTS`, `ORG_MEMBER` download check), `ManifestService`. **Done:** `DatasetVersionRepositoryIT`, `ModelArtifactRepositoryIT`, `DeterminismManifestRepositoryIT` pass on Testcontainers-PG.
18. **Wire-up: schema admission gate** — in the FL server registration path, compare the client `SchemaFingerprint.schema_hash` to the pinned `dataset_versions.schema_json` hash; reject on mismatch → `FAILED_PRECONDITION`. **Done:** `test_schema_mismatch_rejected` passes; a matching client is admitted.
19. **Wire-up: run-start manifest + initial-model artifact** — at server start, build the manifest, content-address + register the initial model (`kind=INITIAL`), persist the manifest row, call `log_run_start`. **Done:** a started run has one `determinism_manifests` row and one `model_artifacts(kind=INITIAL)` row.
20. **Wire-up: per-round checkpoint** — after each round, content-address + register the checkpoint (`kind=CHECKPOINT`, `round_idx=r`), `POST /api/internal/runs/{runId}/checkpoint` (`CheckpointReportDto`), `log_round_metrics`. **Done:** `GET /api/runs/{runId}/checkpoints` returns one `CheckpointDto` per completed round.
21. **Wire-up: resume** — on server restart, read `fl_runs.round_idx` + the latest `kind=CHECKPOINT` artifact, `GET .../download-url`, `ArtifactStore.get_model` (verify sha256), resume. **Done:** a killed-then-restarted run continues from the last checkpoint round, not round 0.

---

*End of 17-LLD-data-and-artifacts.md. All existing-code claims cite `file:line` against `main-clean`; all design decisions cite `C2-data-engineering.md`, `C3-reproducibility.md`, `B2-tech-stack.md`, or the foundation docs 02/03/04 under `docs/v2/build/`. Conforms to the locked tech stack and the data-model / API contracts; contradicts none.*
