# C2 — Data Engineering / The Data Plane (v2 Greenfield)

**Scope:** the data plane of FedLearn. FL exists to train on decentralized, heterogeneous data; this report assesses where that data physically lives, how it is partitioned, versioned, validated, and protected, and designs the v2 data subsystem for a production-grade startup.

**Date:** 2026-05-29 · **Branch:** `main-clean` (+ `origin/fed-mobile` for the mobile C++ client) · **Builds on:** `docs/audit/2026-05-27/03-framework.md`

---

## 0. How this builds on the 2026-05-27 audit

The prior framework audit (`03-framework.md`) already flagged the data-adjacent defects from a *correctness/security* lens. I take those as given and do not re-litigate them; I extend them into a *data-plane architecture* verdict:

| Prior finding | Prior framing | This report extends it to |
|---|---|---|
| **C2** `pickle.load` on attacker-mutable split caches (`client-docker/scripts/client.py:328`, `ecg_loader.py:114`) | RCE security bug | The split-cache mechanism itself is the wrong primitive — it conflates *partition reproducibility* with *on-disk caching* and has no integrity, no versioning, no schema. Rebuild, not patch. |
| **C3** `compressed` flag inferred from env var | brittle round serialization | Symptom of a missing **data contract** between peers. v2 negotiates serialization + schema in the proto, not the environment. |
| **H6** `flwr_datasets` still a runtime dep (`client.py:85`, `requirements.txt:7-8`) | "remove the dep" | The dep is the *only* thing implementing partitioning today; removing it requires owning a partitioning module — designed below. |
| **M5** `np.random.seed(seed)` mutates global RNG (`decomfl_strategy.py:82`) | two servers clobber each other | Seed is **doubly load-bearing**: it controls both data partition reproducibility *and* DeComFL perturbation reproducibility (`decomfl_strategy.py:82,107`). The data plane must own a seed namespace distinct from the optimizer's. |

The prior audit did **not** assess: where training data physically lives, dataset versioning/lineage (none exists), the platform-managed vs client-private boundary, or data-at-rest encryption. Those are the core of this report.

---

## 1. Executive summary

The FL premise — "raw data never leaves the device" — is **contradicted by the current ECG implementation**. The canonical example dataset (`ecg.csv`, 5.7 MB) is **committed into the repository in two places** and **shipped inside the Spring-Boot-spawned server** (`backend/fl-platform-api/src/main/resources/scripts/ecg_data/ecg.csv`), where `fl_server.py:97-106` loads the *entire* dataset centrally and `ecg_loader.py:30-71` Dirichlet-partitions it. The "federation" is a **simulation**: one machine owns all the data and hands out index slices. The pneumonia demo plan (`docs/guides/pneumonia_demo_plan.md`) describes the *intended* real topology (each device pre-downloads and caches its own partition as `.pt`), but that path is aspirational and undocumented in code.

There is **no dataset registry, no versioning, no lineage, and no schema validation**. Partition reproducibility exists only as a side effect of a hardcoded `seed=42` (`ecg_loader.py:141`, `client.py:331`, `fl_server.py`), persisted to unsigned pickle caches. Three independent, drifting copies of the *same* Dirichlet code exist (`client-docker`, `framework/examples/`, `client.py` inline) plus two more partitioners on mobile (C++ `DataLoader` and JS `DatasetLoader`) that do not partition at all — they ship fixed JSON MNIST subsets bundled into the app.

**Verdict at a glance:** the data plane is the least-developed subsystem in the platform and the one most central to the product thesis. The simulation-grade partitioning code is **kill** (delete the duplicates, the central CSV, the pickle caches). The data-plane *capability* must be **rebuilt** as a first-class subsystem: a dataset/partition registry in Flyway-owned tables, a `DataSource` plugin interface in the framework, content-addressed manifests for reproducibility, a declared+validated local data schema contract, and a hard platform-managed vs client-private boundary enforced in the proto. Removing `flwr-datasets` (prior H6) is a forcing function, not a chore — it is the act of taking ownership of partitioning.

---

## 2. Current state — evidence

### 2.1 Where training data physically lives (today)

| Path | What | Implication |
|---|---|---|
| `backend/fl-platform-api/src/main/resources/scripts/ecg_data/ecg.csv` (5.7 MB) | Full ECG dataset, **committed**, bundled into the server JAR resources | Server has all raw data; partitioning is centralized simulation. Also bloats the JAR and leaks the dataset into every deploy artifact. |
| `framework/examples/{ecg_federation,ecg_decomfl_*}/ecg_data/ecg.csv` (4× copies) | Same 5.7 MB CSV duplicated | No single source of truth; drift risk. |
| `client-docker/scripts/data_splits/*.pkl` | Pickled `{X_train, X_test, y_train, y_test, client_indices}` cache (`ecg_loader.py:105,144-152`) | Attacker-writable RCE (prior C2); also the *de facto* "versioning" layer with no integrity/version. |
| `mobile_client/{,android/}data/MNIST/raw/*` and `src/{assets/mnist-data,data}/mnist_train_100.json` | Raw MNIST + 100-sample JSON, committed into the app | Client "data" is a fixed bundled fixture, not user data. |
| Pneumonia (intended) | `keremberke/chest-xray-classification` via HuggingFace `datasets`, pre-cached to `.pt` per device (`pneumonia_demo_plan.md:46-79,183`) | The *only* place a real client-private topology is described — and it's a manual runbook, not code. |

**The central architectural contradiction:** `pneumonia_demo_plan.md:40` sells "*The raw images never leave the device*" while the shipped ECG code does the literal opposite. v2 must make the marketing claim *structurally true*, not a per-demo manual setup.

### 2.2 Partitioning — three drifting copies + an external dep

- `client-docker/scripts/data_loaders/ecg_loader.py:30-71` — `dirichlet_split`, uses `np.random.default_rng(seed)` (correct, isolated RNG).
- `framework/examples/ecg_decomfl_framework_integration/data.py:27-68` — **byte-for-byte duplicate** of the above.
- `client-docker/scripts/client.py:248-270` — a **third** `dirichlet_split` that uses the *global* `np.random.seed(seed)` + `np.random.shuffle` (`client.py:250,258`) — RNG-global, conflicts with DeComFL's global seed.
- `client-docker/scripts/client.py:363` + `backend/.../fl_server.py:31` — the CNN/CIFAR path delegates partitioning to `flwr_datasets.FederatedDataset` (the dep CLAUDE.md says doesn't exist; prior H6). `requirements.txt:7-8` pins `flwr==1.20.0`, `flwr-datasets==0.5.0`; `client-docker/requirements.txt:10` pins `flwr-datasets>=0.3.0`.

So partitioning logic is forked **four** ways with **three** different RNG strategies. Reproducibility is accidental.

### 2.3 Partition reproducibility & seed control

- Seed is hardcoded `42` everywhere (`ecg_loader.py:141`, `client.py:331`, `data.py:135`). Not surfaced in the project config, not stored, not auditable.
- The cache filename *is* the only "version key": `ecg_clients{N}_alpha{alpha}_frac{frac}_seed{seed}.pkl` (`ecg_loader.py:105`). Change the dataset contents without changing N/alpha/frac/seed and you silently reuse a stale split — a reproducibility trap (no content hash in the key).
- **Seed is doubly load-bearing.** `decomfl_strategy.py:82` calls `np.random.seed(seed)` then `decomfl_strategy.py:107` draws perturbation seeds from that same global stream. The data partitioner *also* reaches for the global RNG in the `client.py` variant. A single global seed therefore couples *data partition layout* and *DeComFL gradient perturbations* — two things that must be independently reproducible. Prior M5 flagged the RNG-global bug; the data-plane consequence is that you cannot re-partition data without perturbing the optimizer's randomness, and vice versa.

### 2.4 Data loading paths

- **Python (server-spawned & docker client):** `pandas.read_csv` → numpy → `ECGDataset(Dataset)` → `torch.utils.data.DataLoader` (`ecg_loader.py:16-27,227`). LLM path: HF `datasets.load_dataset` → tokenize → `Subset` (`client.py:281-359`). CNN path: `flwr_datasets` (`client.py:363`).
- **Mobile C++ (`origin/fed-mobile`):** `mobile_client/shared/src/DataLoader.cpp` parses JSON with `nlohmann::json`, `torch::from_blob(...).clone()` into fixed `{n,1,28,28}` MNIST tensors; `getBatches()` does naive contiguous slicing (no shuffle). `DataLoader.h` hardcodes the MNIST shape. No partitioning, no schema beyond the hardcoded 28×28.
- **Mobile JS (`origin/fed-mobile`):** `mobile_client/src/federated/DatasetLoader.js` loads `mnist_train_100.json` via `require`, `getRandomSamplesPerClass` uses `Array.sort(() => Math.random()-0.5)` (a *biased, non-uniform, unseeded* shuffle) and `tf.randomUniform` for shuffling — **no reproducibility at all** on mobile.

There is **no shared `DataSource` abstraction**: the `BaseClient`/`DeComFLClient` interfaces (`framework/src/fedlearn/client/client.py:21-27`, `decomfl_client.py:121`) take `fit(parameters, config)` and assume the subclass already holds loaders. Data acquisition is entirely the caller's problem, which is why it forked.

### 2.5 Versioning / lineage

**None.** No dataset IDs, no version pins, no manifest, no checksum, no lineage from "model result" → "data snapshot." The V4 migration added a *Model Hub* (`projects.model_published`, `model_tags`, etc., `V4__...sql:13-16`) but **no dataset entity exists** in any of V1–V5. A published model in the hub has no recorded provenance of what data trained it — fatal for a regulated (health) use case and for reproducible research (DeComFL is an RIT paper artifact).

### 2.6 Platform-managed vs client-private boundary

Undefined in code. Today the boundary is *accidental*: in ECG the platform owns everything; in pneumonia the boundary is a human following a runbook. The proto (`fedlearn.v1`) carries model chunks and gradient scalars but **no dataset descriptor / schema fingerprint**, so the server cannot verify a connecting client trains on schema-compatible data. A client can join with wrong-shaped or wrong-labeled data and silently poison aggregation (related to prior H2's silent sample-count capping).

---

## 3. v2 design — the FedLearn Data Plane

Design goals, calibrated to a startup: (a) make "data stays on the client" *structurally true*; (b) cheap to operate (no mandatory heavyweight data lake); (c) reproducible runs are a product feature (FL run observability — see B3); (d) schema mismatches fail closed, early, with a clear error; (e) DeComFL's seed semantics preserved and *decoupled* from data seeds.

### 3.1 Two-plane model — the hard boundary

```
 PLATFORM-MANAGED PLANE                         CLIENT-PRIVATE PLANE
 (Spring Boot + registry DB + server)           (desktop / docker / mobile / jetson)
 ┌─────────────────────────────────┐            ┌──────────────────────────────────┐
 │ Dataset Registry (Flyway tables)│            │ Local data (NEVER transmitted)    │
 │  - dataset spec + schema contract│            │  - raw files, on-device DB, etc.  │
 │  - partition recipe (alpha,seed, │            │  - encrypted at rest (§3.5)       │
 │    N, content-hash, version)     │  contract  │                                   │
 │  - run → dataset-version lineage │ ─────────▶ │ DataSource plugin                 │
 │  - server-held GLOBAL TEST SET   │            │  - validates local data vs schema │
 │    (opt-in, for global eval only)│ ◀───────── │  - emits SchemaFingerprint + stats│
 └─────────────────────────────────┘  fingerprint└──────────────────────────────────┘
        ↑ metadata + small held-out test only          ↑ training data + labels live here only
```

**Invariant (new):** the only data that may cross into the platform plane is (1) the optional held-out **global test set** that a project *owner* explicitly uploads for global evaluation, and (2) **non-reversible metadata** — schema fingerprints, per-label counts, sample counts, dataset-version hashes. Raw training features/labels **never** appear in any RPC, log, or DB row. This is enforceable and testable (a CI test that greps the proto + servicer for any field that could carry raw samples).

> Note this *legitimizes* the server holding a small test set (the pneumonia plan already does this, `pneumonia_demo_plan.md:67,77`) while *forbidding* the current ECG behavior of the server holding the entire training corpus.

### 3.2 Dataset & partition registry (Flyway `V6`)

Schema is Flyway-owned (platform invariant). New tables (sketch — not final DDL):

```sql
-- V6__data_plane.sql
CREATE TABLE datasets (
    id           UUID PRIMARY KEY,
    org_id       UUID NOT NULL REFERENCES organizations(id),
    name         VARCHAR(255) NOT NULL,
    modality     VARCHAR(32)  NOT NULL,   -- TABULAR | IMAGE | TEXT | TIMESERIES
    created_by   BIGINT REFERENCES users(id),
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE (org_id, name)
);

-- Immutable, content-addressed versions. content_hash is the lineage anchor.
CREATE TABLE dataset_versions (
    id             UUID PRIMARY KEY,
    dataset_id     UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version        INTEGER NOT NULL,          -- monotonic per dataset
    content_hash   CHAR(64) NOT NULL,         -- sha256 of canonical schema+stats manifest
    schema_json    JSONB NOT NULL,            -- the declared schema contract (§3.4)
    sample_count   BIGINT,                    -- aggregate, metadata only
    created_at     TIMESTAMP WITH TIME ZONE NOT NULL,
    UNIQUE (dataset_id, version)
);

-- A reproducible partition recipe. NO raw indices stored server-side for
-- client-private data; indices are derived deterministically on the client.
CREATE TABLE partition_recipes (
    id                 UUID PRIMARY KEY,
    dataset_version_id UUID NOT NULL REFERENCES dataset_versions(id),
    partitioner        VARCHAR(32) NOT NULL,  -- DIRICHLET_LABEL | DIRICHLET_QTY | SHARD | NATURAL
    num_partitions     INTEGER NOT NULL,
    alpha              DOUBLE PRECISION,      -- null for non-Dirichlet
    data_seed          BIGINT NOT NULL,       -- DATA seed, distinct from optimizer seed
    recipe_hash        CHAR(64) NOT NULL      -- sha256(partitioner|N|alpha|data_seed|content_hash)
);

-- Lineage: every FL run pins exactly one (dataset_version, partition_recipe).
ALTER TABLE projects ADD COLUMN dataset_version_id UUID REFERENCES dataset_versions(id);
ALTER TABLE projects ADD COLUMN partition_recipe_id UUID REFERENCES partition_recipes(id);
-- (per-run results table from B3 also FKs partition_recipe_id for full lineage)
```

- **Content-addressing for lineage.** `recipe_hash = sha256(partitioner | N | alpha | data_seed | dataset_version.content_hash)` is the reproducibility key — it *replaces* the brittle pickle filename (`ecg_loader.py:105`). Change the data and the hash changes, so you can never silently reuse a stale split. This mirrors the DVC/lakeFS content-addressed-snapshot pattern (lakeFS, DVC), scaled down to metadata-only since raw data is client-private.
- **`NATURAL` partitioner** is first-class: in real federations (hospitals), data is *already* partitioned by where it sits — there is no Dirichlet step, just "this client owns its rows." The registry records `partitioner=NATURAL` and stores only the schema contract + per-client fingerprints.

### 3.3 `DataSource` plugin interface (framework) — kills the four forks

Introduce one abstraction in `framework/src/fedlearn/data/` that both Python clients and the spawned server consume:

```python
# framework/src/fedlearn/data/source.py  (NEW)
class DataSource(ABC):
    @abstractmethod
    def schema(self) -> DataSchema: ...                 # declared contract (§3.4)
    @abstractmethod
    def fingerprint(self) -> SchemaFingerprint: ...     # non-reversible: shapes, dtypes, label set, counts
    @abstractmethod
    def train_loader(self, batch_size: int) -> DataLoader: ...
    @abstractmethod
    def eval_loader(self, batch_size: int) -> DataLoader | None: ...

class Partitioner(ABC):
    @abstractmethod
    def partition(self, labels: np.ndarray, recipe: PartitionRecipe) -> list[np.ndarray]: ...
```

- Ship `DirichletLabelPartitioner` (port `ecg_loader.py:30-71`, the *good* `default_rng` version) and `NaturalPartitioner` (identity). Delete the three other copies.
- **Decouple seeds:** `Partitioner` is constructed with `np.random.Generator(np.random.PCG64(recipe.data_seed))` — never the global RNG (fixes the data side of prior M5). The DeComFL optimizer keeps its own `optimizer_seed` generator. Two seeds, two namespaces, both stored (`partition_recipes.data_seed`; DeComFL seed in its own column).
- **Reproducibility test:** golden-output test asserts `DirichletLabelPartitioner(recipe)` produces byte-identical partitions across runs and platforms — this is the regression gate the prior audit's H6 "assert partition output matches" asked for, now made first-class.
- This also **removes `flwr-datasets`** (prior H6): the CIFAR path calls `HF datasets.load_dataset("cifar10")` + `DirichletLabelPartitioner`, not `FederatedDataset`. `datasets==3.1.0` is already a dep (`requirements.txt:12`).

### 3.4 Heterogeneous client schema declaration + validation

Each client declares its local schema and the platform validates it **at handshake, before any round**:

```protobuf
// additions to fedlearn.v1 (handshake/registration message)
message DataSchema {
  enum Modality { TABULAR = 0; IMAGE = 1; TEXT = 2; TIMESERIES = 3; }
  Modality modality = 1;
  repeated int64 feature_shape = 2;   // e.g. [1,28,28] or [140]
  string  feature_dtype = 3;          // "float32"
  repeated int64 label_set = 4;       // observed class ids
  int64   num_classes = 5;
}
message SchemaFingerprint {           // non-reversible — safe to transmit
  string  schema_hash = 1;            // sha256 of DataSchema canonical form
  int64   sample_count = 2;
  map<int64,int64> label_histogram = 3;  // {label: count} — metadata only
}
```

- Server compares each client's `schema_hash` against the project's pinned `dataset_versions.schema_json` and **rejects** (gRPC `FAILED_PRECONDITION`) on mismatch — closes the silent-poisoning gap from §2.6. This is the structural fix the literature calls "data-quality-aware client selection" (MDPI 12(20):3229) and feature/label-skew handling (FL-Joint, Springer), implemented as an admission gate rather than a post-hoc loss.
- `label_histogram` feeds **FL-run observability** (B3): the dashboard can render per-client non-IID heatmaps from fingerprints alone, without ever seeing raw data. This is the data-plane half of "performance observability of FL runs."
- The mobile clients (C++ `DataLoader`, JS `DatasetLoader`) must emit the same fingerprint — today the JS shuffle is unseeded/biased (`DatasetLoader.js` `sort(() => Math.random()-0.5)`), so mobile cannot currently participate in a reproducible run. v2 requires every `DataSource` (incl. mobile) to implement `fingerprint()`.

### 3.5 Data-at-rest privacy / encryption

Raw data lives only in the client-private plane, so encryption is a *client-side* concern, tiered by deployment:

| Deployment | At-rest control | Cost/feasibility |
|---|---|---|
| Desktop (Electron) | OS-native: the desktop already uses `safeStorage` (OS keychain) for JWT per CLAUDE.md — extend the same `safeStorage`-derived key to encrypt the local `.pt` partition cache. | Free; already in stack. |
| Docker / Jetson | Document a bind-mounted volume on a LUKS/dm-crypt or eCryptfs path; do **not** invent crypto. Cache files written by `DataSource` use `np.savez_compressed` + a sidecar SHA-256 (replaces pickle, fixes prior C2). | Ops runbook, near-zero code. |
| Mobile | iOS Data Protection / Android `EncryptedFile` (Jetpack Security) for the on-device dataset. | Platform-native APIs; no custom crypto. |
| Platform plane | The optional global test set is the only data the server stores: encrypt that column/blob at rest (RDS/EBS encryption or app-level envelope encryption). | Cheap on managed cloud. |

**Hard rule (testable):** the `data_splits/*.pkl` pickle path is deleted entirely. Caches become content-addressed `.npz` + checksum; load verifies the checksum before deserializing (defends prior C2 and detects corruption). `torch.load(weights_only=True)` invariant from prior C3 stays for model params; data never goes through `torch.load` of arbitrary objects.

### 3.6 Communication-round bounding (FL-context, per CLAUDE.md mandate)

- **Aggregation strategy assumed:** FedAvg and DeComFL unchanged. The data plane is *orthogonal* to aggregation — it only changes how loaders are constructed and validated.
- **Client heterogeneity:** label-skew + quantity-skew explicitly modeled (Dirichlet alpha stored per recipe) and *measured* (fingerprints). Feature-skew (different domains) is admitted only when `schema_hash` matches; cross-domain federation is out of scope for v2.
- **Communication-round-bounded:** schema validation is a **one-time handshake cost** (O(1) per client per run), not per-round. Fingerprints are tens of bytes. The data plane adds **zero per-round communication** — critical because DeComFL's whole value proposition is O(K×P) per-round cost independent of model dimension; we must not undermine it with per-round data chatter.

---

## 4. Decision table

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| Central ECG CSV bundled in server (`fl_server.py:97-112`, `scripts/ecg_data/ecg.csv`) | **kill** | Server holding the full training corpus contradicts the FL premise; delete from JAR + repo. |
| Duplicated `dirichlet_split` (`client.py:248`, `data.py:27`, examples ×4) | **kill** | Four drifting forks with three RNG strategies; collapse into one `Partitioner`. |
| `ecg_loader.py` good `default_rng` Dirichlet logic (`:30-71`) | **salvage** | Correct algorithm — port verbatim into `framework/src/fedlearn/data/` as the canonical partitioner. |
| Pickle split cache (`ecg_loader.py:105-152`, `client.py:328`) | **rebuild** | Replace with content-addressed `.npz` + checksum; fixes prior C2 and adds versioning. |
| `flwr_datasets` partitioning (`client.py:363`, `fl_server.py:31`, reqs) | **kill** | Violates the no-`flwr` invariant (prior H6); replace with HF `datasets` + own partitioner. |
| Partition reproducibility / seed control | **rebuild** | Promote `data_seed` to a stored, registry-owned value distinct from the DeComFL optimizer seed (prior M5). |
| Dataset registry / versioning / lineage | **rebuild** | Does not exist; add Flyway `V6` tables + content-addressed manifests. |
| Client schema declaration + validation | **rebuild** | Add `DataSchema`/`SchemaFingerprint` to `fedlearn.v1`; reject mismatches at handshake. |
| Platform vs client-private data boundary | **rebuild** | Currently accidental; codify the two-plane invariant + CI test that raw data never enters an RPC. |
| Data-at-rest encryption | **rebuild** | None today; tier by deployment using OS-native primitives (no custom crypto). |
| Server-held global test set (pneumonia pattern) | **salvage** | Legitimate platform-plane data; formalize as the *only* allowed inbound data, encrypted at rest. |
| Mobile C++ `DataLoader` (`fed-mobile`) | **refactor** | Sound tensor loading, but hardcoded MNIST shape, no shuffle, no fingerprint; conform to `DataSource`. |
| Mobile JS `DatasetLoader` (`fed-mobile`) | **rebuild** | Unseeded biased shuffle (`sort(()=>Math.random()-0.5)`); cannot produce reproducible partitions. |
| `DataSource` / `Partitioner` interface | **rebuild** (net-new) | The missing abstraction whose absence caused all the forking. |

---

## 5. Prioritized recommendations

**P0 — invariants & safety (do first; small, high-leverage)**
1. **Delete the central training CSV from the server path.** Remove `scripts/ecg_data/ecg.csv` from JAR resources and de-duplicate the 5 repo copies; the server keeps *only* an optional held-out test set. Codify the two-plane invariant with a CI test that fails if any `fedlearn.v1` field or log line can carry raw samples.
2. **Replace pickle caches with content-addressed `.npz` + SHA-256** (folds in prior C2 fix) and delete the `data_splits/*.pkl` path everywhere.
3. **Split the seed namespace:** introduce `data_seed` (partitioner) vs `optimizer_seed` (DeComFL), both via `np.random.Generator`, never the global RNG (folds in prior M5).

**P1 — the data-plane core**
4. **Land `framework/src/fedlearn/data/` with `DataSource` + `Partitioner`.** Port the good `default_rng` Dirichlet, add `NaturalPartitioner`, delete the other three forks, and **drop `flwr-datasets`** (prior H6) with a golden-partition regression test.
5. **Flyway `V6` data-plane registry** (`datasets`, `dataset_versions`, `partition_recipes`) + `projects.dataset_version_id/partition_recipe_id`. Content-addressed `recipe_hash` replaces the pickle filename as the reproducibility key.
6. **Add `DataSchema`/`SchemaFingerprint` to the proto** and gate client admission on `schema_hash` match (fail closed). Feed `label_histogram` to the FL-run dashboard (B3).

**P2 — privacy & scale**
7. **Tiered at-rest encryption** using OS-native primitives (desktop `safeStorage`, mobile EncryptedFile/Data-Protection, docker LUKS runbook, encrypted RDS/EBS for the test set). No custom crypto.
8. **Lineage wiring:** every FL run result row FKs `partition_recipe_id`; surface "trained on dataset vX, recipe hash …" in the Model Hub (closes the V4 provenance gap). Optionally log the recipe hash to MLflow (matches the prior audit's MLflow recommendation in `04-observability.md`).
9. **Mobile conformance:** make C++/JS loaders implement `fingerprint()`; replace the JS biased shuffle with a seeded Fisher–Yates.

---

## 6. Open questions / uncertainty (flagged, not papered over)

- **DeComFL data flow under true decentralization.** DeComFL communicates gradient *scalars + seeds* (`decomfl_strategy.py:22-23`); the server reconstructs the model by replaying perturbations. I have **not** verified whether the current server-side reconstruction assumes access to a representative data sample for any step (it should not, but the central-CSV pattern makes this ambiguous). **Verify** `decomfl_strategy.py` + `decomfl_client.py.fit` make no hidden assumption that the server sees training data before declaring the two-plane invariant "free." (Prior audit M6 already notes `decomfl_client.fit` is untested.)
- **Natural-partition reproducibility is bounded.** For `NATURAL` partitions (real hospital data), the platform *cannot* reproduce the exact data — only the schema fingerprint + recipe. "Reproducibility" for real federations means "same recipe + same declared schema," not "same bytes." This is correct FL semantics but must be communicated to users so they don't expect bit-exact replay.
- **Content-hash of client-private data.** We hash the *schema + aggregate stats manifest*, not raw data (which the platform never sees). Two clients with identical schemas but different rows share a `schema_hash` — intended, but means `schema_hash` is a *compatibility* key, not a uniqueness key. A separate per-client `data_fingerprint` (hash of local content, computed and kept client-side, optionally reported) covers drift detection if needed.
- **`flwr-datasets` removal blast radius on ARM64.** Prior audit notes pyarrow/`flwr-datasets` pinning gymnastics on Jetson (`client-docker/requirements.txt:24-31`). Removing `flwr-datasets` should *simplify* the ARM64 dep tree, but `datasets==3.1.0` still pulls `pyarrow` — verify the arm64 wheel resolves on `l4t-pytorch:r35.2.1` before claiming the dep removal is a net ARM64 win.

---

## 7. Sources

- Flower Datasets — `DirichletPartitioner` (seed semantics, default 42): https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.DirichletPartitioner.html
- Flower Datasets — partitioner module source (reproducibility via RNG seed): https://flower.ai/docs/datasets/_modules/flwr_datasets/partitioner/dirichlet_partitioner.html
- Li et al., "Federated Learning on Non-IID Data Silos: An Experimental Study" (Dirichlet β quantity skew): https://arxiv.org/pdf/2102.02079
- "Non-IID data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions" (2024): https://arxiv.org/html/2411.12377v2
- "Understanding Federated Learning from IID to Non-IID dataset: An Experimental Study" (2025): https://arxiv.org/html/2502.00182v1
- "Data Quality-Aware Client Selection in Heterogeneous Federated Learning" (admission/quality gating): https://www.mdpi.com/2227-7390/12/20/3229
- "FL-Joint: joint aligning features and labels in federated learning for data heterogeneity": https://link.springer.com/article/10.1007/s40747-024-01636-4
- lakeFS — "DVC vs. Git-LFS vs. Dolt vs. lakeFS: Data Versioning Compared" (content-addressed snapshots): https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/
- lakeFS — "Git-Like Data Versioning Meets MLOps … MLflow … lineage" (run↔data-commit lineage): https://lakefs.io/blog/git-like-data-versioning-meets-mlops-lakefs-with-mlflow-datachain-neptune-quilt/
- DVC User Guide (lightweight file-based versioning model): https://doc.dvc.org/user-guide
