# B7 — Coding Standards, Maintainability & Developer Experience (v2)

**Agent:** B7 (Standards / DX / Monorepo / CI-CD / Proto Codegen)
**Date:** 2026-05-29
**Scope:** Cross-cutting toolchain for the polyglot v2 monorepo (Java + Python + TypeScript + C++ across `backend/`, `framework/`, `frontend/`, `fedlearn-desktop/`, `client-docker/`, and `mobile_client/`).
**Builds on:** `docs/audit/2026-05-27/05-tooling-ci.md` (cited as **[PRIOR §n]**). This report **extends and does not duplicate** that report. Where the prior audit's current-state matrix is still accurate I reference it rather than re-deriving it; where greenfield-v2 framing changes the recommendation, I say so explicitly.

---

## 0. Executive summary

The prior audit nailed the per-unit current state and a sensible incremental remediation path **[PRIOR §1–6]**. This report re-frames those findings for a **greenfield v2 startup** and adds the three things the v1-incremental lens could not: (1) a **monorepo build-orchestration decision** now that a sixth unit (native C++ `mobile_client/`) is in scope; (2) a **single-source-of-truth proto codegen pipeline across four languages** — today there are **three hand-maintained, already-drifted proto copies** and **three independent codegen mechanisms**; (3) a **release-engineering / versioning** scheme to replace the hand-bumped `1.0.5-beta` string copied across three files.

The single most load-bearing, cheapest-to-fix gap remains the one the prior audit flagged P0 **[PRIOR §2 item 1]**: **there is zero PR-time CI.** Both `.github/workflows/*.yml` fire on `push: tags: ['v*']` only (`desktop-release.yml:3-6`, `release-desktop.yml:23-26`); pre-commit is opt-in. A PR today can merge with red tests, broken imports, vulnerable deps, or leaked secrets, and nothing stops it. For a startup heading toward a compliance floor (see B4), that is both an engineering and an audit-trail liability.

I also surface a fact the prior audit understated: the backend runs **Spring Boot 3.4.5** (`backend/fl-platform-api/build.gradle:4`), and **the 3.4 line reached open-source end-of-life on 2025-12-31** ([spring.io](https://spring.io/blog/2026/04/23/spring-boot-3-5-14-available-now/), [endoflife.date/spring-boot](https://endoflife.date/spring-boot)). It is not "one minor behind" — it is **unsupported**. That escalates the dependency-hygiene finding from medium to high.

---

## 1. Verdicts on current tooling (decision table)

| Subsystem / tool | Verdict | One-line rationale |
|---|---|---|
| **PR-time CI (the absence of it)** | **rebuild** | No PR gate exists; the only two workflows are tag-driven release builds. This is the highest-leverage DX fix. |
| `release-desktop.yml` (tag release pipeline) | **salvage** | Genuinely good multi-arch PyInstaller+Electron matrix with sane timeouts and partial-failure handling; keep it, just add `--publish never` already present and wire signing. |
| `desktop-release.yml` (the *other* release workflow) | **kill** | Duplicate, older, node-18, no Python/native step — superseded by `release-desktop.yml`. Two release workflows on the same `v*` tag is a footgun. |
| Pre-commit config (`.pre-commit-config.yaml`) | **refactor** | Solid hook set but opt-in, has a known-bad `prettier v4.0.0-alpha.8` pin, references `frontend/eslint.config.js` while the repo ships `eslint.config.mjs`, and is not enforced in CI. |
| Proto codegen (3 copies, 3 mechanisms) | **rebuild** | One canonical `fedlearn.proto`, hand-run `grpcio-tools` for Python, hand-rolled gRPC cross-compile for C++, drifted mobile copies. Needs `buf` as single source of truth. |
| Backend static analysis (none) | **rebuild** | No linter, formatter, SpotBugs, JaCoCo, or coverage gate on 132 Java files doing JWT + `ProcessBuilder` shell spawning. |
| Framework lint/type (ruff + mypy strict) | **salvage** | Best-configured unit in the repo; just widen ruff `select` and add coverage + `pip-audit`. |
| Frontend/desktop lint/type (ESLint flat + tsc strict) | **refactor** | Good baseline, but `recommended`-only (not `strictTypeChecked`), zero frontend tests, version drift, config-path mismatch. |
| Dependency hygiene (Renovate/Dependabot/audit) | **rebuild** | Nothing exists; lockfiles will rot; backend is on an EOL Spring Boot line. |
| Secret scanning | **refactor** | Only `detect-private-key`; needs gitleaks in CI + history baseline. |
| Versioning / release notes | **refactor** | `1.0.5-beta` hand-copied across 3 manifests; no changelog automation; no per-unit independent versioning story. |
| Monorepo build orchestration (none) | **refactor** | No Nx/Turbo/Bazel today; six units, four languages, shared proto. Needs *some* task orchestration — but **not** Bazel (see §3). |

---

## 2. The DX problem statement for v2

v1 grew six deployable units in four languages with **no shared build graph and no shared CI**. The friction this creates compounds at startup hiring velocity:

- A change to `fedlearn.proto` requires a developer to remember to regenerate Python stubs (`framework/.../generated/`), hand-edit or re-copy the mobile C++ proto (two copies: `mobile_client/shared/proto/fedlearn.proto`, `mobile_client/src/federated/protos/fedlearn.proto`), and nothing regenerates anything for the backend (Java declares `option java_package = "com.fedlearn.v1"` at `fedlearn.proto:5` but **no Java proto build exists** — the backend talks STOMP/JSON, not gRPC, to the FL server). The drift is already real: the mobile copy has `SubmitModelUpdateReque` where canonical has `SubmitModelUpdateRequest` (confirmed by diff of `framework/.../protos/fedlearn.proto` against `git show origin/fed-mobile:mobile_client/shared/proto/fedlearn.proto`). A typo'd RPC name is a silent wire-incompatibility waiting to ship.
- There is no single "run all checks" entrypoint. A contributor cannot know what "green" means without reading five subdir READMEs.
- The generated Python protobuf files are **committed** (`framework/src/fedlearn/communication/generated/fedlearn_pb2.py`, `.pyi`, `_pb2_grpc.py`) with the same mtime as the `.proto`, i.e. they are a manual artifact that *can* go stale silently.

A production-grade startup needs: one command to lint/test/typecheck/build any unit, one source of truth for the wire contract, one CI gate on every PR, and bot-driven dependency upgrades. None of that requires heavy infrastructure.

---

## 3. Monorepo orchestration: Nx vs Turborepo vs Bazel

**Recommendation: do NOT adopt Bazel. Use a thin orchestration layer — `make` + per-language native tools, optionally Nx if/when JS/TS task-graph caching becomes a bottleneck. Bazel is the wrong fit for a 6-person-scale startup.**

| Tool | Polyglot fit | Cost to adopt | Verdict for FedLearn v2 |
|---|---|---|---|
| **Turborepo** | JS/TS only ([daily.dev](https://daily.dev/blog/monorepo-turborepo-vs-nx-vs-bazel-modern-development-teams/)) | Low | **Reject** — would only cover `frontend/` + `fedlearn-desktop/`; ignores Java/Python/C++ (4 of 6 units). |
| **Nx** | JS/TS first-class; Java/Python/C++ only via custom executors/run-commands, not native ([daily.dev](https://daily.dev/blog/monorepo-turborepo-vs-nx-vs-bazel-modern-development-teams/)) | Medium | **Optional later** — useful as the JS/TS task runner + remote cache for `frontend`+`desktop`+`mobile`(RN/TS layer). Not the whole-repo graph. |
| **Bazel** | True polyglot, hermetic, the "go-to for massive multi-language monorepos" but "demands explicit declarations for every dependency and output … often requiring a dedicated build infrastructure team" ([daily.dev](https://daily.dev/blog/monorepo-turborepo-vs-nx-vs-bazel-modern-development-teams/), [aviator.co](https://www.aviator.co/blog/monorepo-tools/)) | **Very high** | **Reject for now** — the C++/libtorch/CUDA/Jetson cross-compile matrix (`mobile_client/scripts/build_*.sh` already hand-rolls gRPC v1.62 cross-compilation) is exactly where Bazel's hermeticity *would* pay off, but the BUILD-file authoring + toolchain-pinning burden is a full-time role this team does not have. Revisit only at Series-A+ scale (see B6 tiers) if build determinism across the C++ targets becomes a recurring incident source. |

**Rationale.** The honest constraint is team size, not technical capability. Nx/Turbo give nothing to Java/Python/C++ except a `run-commands` shell-out wrapper — at which point a `Makefile` (or `Taskfile.yml`) is simpler and language-agnostic. The right v2 baseline:

- **Root `Makefile`/`Taskfile`** with per-unit targets (`make lint`, `make test`, `make proto`, `make build`) that delegate to each unit's native tool (`./gradlew`, `pytest`, `npm`, `cmake`). This is what CI calls; it is also what a new hire runs locally. Zero new conceptual surface.
- **`dorny/paths-filter`** in CI to only run the units a PR touches (the prior audit already sketched this — **[PRIOR §4 ci.yml]**). This gives the "affected-only" build behavior Nx is prized for, at near-zero cost, for a 6-unit repo.
- **Reconsider Nx** specifically for the JS/TS triangle (`frontend` + `desktop` renderer + `mobile` RN layer) **if** they start sharing a component library (C5 recommends shadcn/Radix shared across surfaces). At that point Nx's project graph + remote cache earns its keep for *those three* — but it should not own the Gradle/pytest/cmake builds.

> **Uncertainty flagged:** Whether `mobile_client/` ultimately shares enough TS with `frontend`/`desktop` to justify Nx depends on C5's design-system decision and B5's desktop-strategy decision (Electron-vs-Tauri-vs-native). If desktop goes native/Tauri and mobile keeps its own RN toolchain, the JS/TS surface may be too small to warrant Nx at all. Defer the Nx call to after B5/C5 land.

---

## 4. Proto codegen across four languages (the highest-value DX rebuild)

**Recommendation: adopt `buf` as the single source of truth for `fedlearn.proto`. Generate Python, TypeScript, Java, and C++ from one `buf.gen.yaml`; lint + breaking-change detection on every PR; never hand-copy or hand-generate again.**

### 4.1 Current state (evidence)

- **One canonical proto:** `framework/src/fedlearn/communication/protos/fedlearn.proto` (`package fedlearn.v1`).
- **Python:** generated by hand-run `grpcio-tools`; output **committed** to `framework/src/fedlearn/communication/generated/{fedlearn_pb2.py,fedlearn_pb2.pyi,fedlearn_pb2_grpc.py}`. No regeneration step in any build.
- **C++ (mobile):** `mobile_client/scripts/build_grpc_android.sh` / `build_grpc_ios.sh` clone gRPC `v1.62.0` from source, build host `protoc` + `grpc_cpp_plugin`, then cross-compile — i.e. a bespoke, slow, source-built toolchain pinned by hand.
- **Two drifted mobile copies** of the proto (`mobile_client/shared/proto/`, `mobile_client/src/federated/protos/`), already containing a `SubmitModelUpdateReque` typo vs canonical (confirmed via diff; also flagged in `00-DESIGN.md:48-51`).
- **Java:** no proto build at all; `option java_package` is declared but unused (backend uses STOMP/JSON to the FL server, not gRPC).
- **TypeScript:** no proto usage today (frontend↔backend is JSON over STOMP). v2 may want typed proto for any future direct-to-server browser path; not required now.

### 4.2 Why `buf`

`buf generate` runs Protobuf plugins over `.proto` files producing **Go, TypeScript, Java, Python, C++, Rust** ([buf.build/docs/generate](https://buf.build/docs/generate/)). It directly addresses every pain above:

- **Single source of truth + breaking-change detection.** `buf breaking` in CI catches the exact class of drift that produced `SubmitModelUpdateReque`. `buf lint` enforces naming/style on the contract.
- **Managed mode** moves `java_package`/etc. out of the `.proto` into `buf.gen.yaml`, keeping the schema language-neutral so each consumer generates its own way ([buf.build/docs/generate/managed-mode](https://buf.build/docs/generate/managed-mode/)). This lets the backend opt into Java stubs later without touching the canonical file.
- **Remote plugins** are pinned by version on the BSR — reproducible codegen with no local `protoc`/plugin install ([buf.build/docs/bsr/remote-plugins](https://buf.build/docs/bsr/remote-plugins/)). For Python/TS/Java this removes the "did you run the right grpcio-tools version" footgun entirely.
- **C++ is supported**, including BSR-hosted `protocolbuffers/cpp` and `grpc/cpp` plugins and **CMake `FetchContent`-consumable generated C++ SDKs** ([buf.build/blog/bsr-generated-sdks-for-cpp](https://buf.build/blog/bsr-generated-sdks-for-cpp)). This is the path that retires `mobile_client/scripts/build_grpc_android.sh`'s hand-rolled source build of `protoc`/`grpc_cpp_plugin`.

> **Uncertainty flagged honestly:** The mobile C++ client also needs the gRPC **runtime/library** cross-compiled for Android arm64 / iOS, which buf does **not** provide — buf generates *stubs*, not the linked gRPC C++ runtime. So `build_grpc_android.sh`/`build_grpc_ios.sh` shrink to "cross-compile the gRPC runtime library" and lose the proto-codegen responsibility; they do not disappear. I have **not** validated that the BSR CMake C++ SDK cross-compiles cleanly under the Android NDK / iOS toolchain — that needs a spike before committing the mobile arm to buf-generated C++. Treat the C++ leg as "strongly recommended, spike-gated"; the Python/TS/Java legs are unambiguous wins.

### 4.3 Concrete v2 layout

```
proto/                          # promote to a top-level, language-neutral home
  buf.yaml                      # module + lint + breaking config
  buf.gen.yaml                  # managed mode + per-language plugin outputs
  fedlearn/v1/fedlearn.proto    # the ONE canonical file
```

`buf.gen.yaml` (illustrative — pin actual plugin versions during implementation):

```yaml
version: v2
managed:
  enabled: true
  override:
    - file_option: java_package
      value: com.fedlearn.v1
plugins:
  - remote: buf.build/protocolbuffers/python
    out: framework/src/fedlearn/communication/generated
  - remote: buf.build/grpc/python
    out: framework/src/fedlearn/communication/generated
  - remote: buf.build/protocolbuffers/cpp     # mobile shared/ — spike-gate first
    out: mobile_client/shared/generated
  - remote: buf.build/grpc/cpp
    out: mobile_client/shared/generated
  # Java + TS generated only if/when a gRPC consumer appears on those sides
```

Decision: **stop committing generated stubs** OR commit them but enforce `buf generate --diff`-style freshness in CI (the latter is friendlier for `pip install -e` consumers who don't run buf). Recommend: commit Python stubs (consumers `pip install -e framework`), but add a CI job that runs `buf generate` and fails if the working tree differs — converting today's silent-staleness risk into a hard gate.

---

## 5. CI/CD — closing the load-bearing gap

The prior audit's five-file CI sketch **[PRIOR §4]** is good and I adopt it wholesale. Extensions for v2:

### 5.1 Architecture (extends [PRIOR §4])

```
.github/workflows/
  ci.yml         # PR + push:main orchestrator → dorny/paths-filter → reusable jobs
  backend.yml    # gradle: spotless + checkstyle + spotbugs(+findsecbugs) + test + jacoco
  framework.yml  # ruff + ruff-format --check + mypy + pytest --cov + pip-audit + bandit
  frontend.yml   # eslint + tsc --noEmit + prettier --check + vitest + npm audit + knip
  desktop.yml    # same shape as frontend + electron-builder --dir smoke
  mobile.yml     # NEW: cmake configure + C++ unit tests + RN/TS lint/typecheck
  proto.yml      # NEW: buf lint + buf breaking (against main) + buf generate freshness
  security.yml   # gitleaks + trivy(client-docker image) + pip-audit (always-run)
  release.yml    # salvaged from release-desktop.yml, tag-triggered
```

### 5.2 Deltas vs the prior sketch (what greenfield-v2 adds)

1. **`proto.yml` is new and non-negotiable.** `buf lint` + `buf breaking` is the structural fix for the drift in §4. Without it, a typed-contract repo with three copies will drift again.
2. **`mobile.yml` is new.** `mobile_client/` is now in scope (`00-DESIGN.md:32`). At minimum: RN/TS ESLint+tsc on PR, and a CMake **configure + compile-check** of `shared/src/*.cpp` on a Linux runner (full Android NDK / iOS cross-compile + libtorch can stay tag-gated for cost — see §5.4). This catches C++ syntax/type breaks without paying the full cross-compile bill per PR.
3. **Required status checks via branch protection.** The CI files mean nothing unless `main` (and the v2 trunk) is protected so PRs cannot merge red. This is a GitHub repo-settings change, not a file — call it out explicitly in the v2 setup runbook. (This is the literal mechanism that closes the "PR can merge broken" gap.)
4. **Concurrency + caching.** `concurrency: { group: ci-${{ github.ref }}, cancel-in-progress: true }` to kill superseded runs (cost). Gradle, npm, and pip caches keyed on lockfiles. The existing `release-desktop.yml:111-112` already caches npm correctly — copy that pattern.
5. **Spring Boot EOL is a CI-visible problem.** Add OWASP dependency-check (or Renovate vulnerability alerts, §7) so the EOL `3.4.5` shows as a failing/again-PR'd item rather than silent rot.

### 5.3 Coverage gates per stack (extends [PRIOR §3])

| Stack | Tool | v2 gate (start lenient, ratchet) | Note |
|---|---|---|---|
| Java backend | JaCoCo | `minimum = 0.70`, ratchet quarterly | None today; 132 files of auth/process-spawn code unmeasured. |
| Python framework | `pytest-cov` | `--cov-fail-under=60`, plus `hypothesis` invariants on FedAvg/DeComFL aggregation | Prior audit's 60% is the right floor; raise to 75% on `serializer.py` + strategy code given the chunking/ZO correctness stakes (A3, B1). |
| TS frontend | vitest | `--coverage` reporting, **no fail-under yet** (zero tests today) → gate at 40% once auth/STOMP paths are covered | Start by *measuring*; gating from zero blocks all PRs. |
| TS desktop | jest (already present) | 50% on IPC/`safeStorage`/dockerode paths | Security-sensitive surface (A5). |
| C++ mobile | ctest + a unit-test framework (Catch2/GoogleTest) | report-only initially; gate the DeComFL/ZO estimator math (B1 fidelity) | The on-device DeComFL math is the paper-fidelity crux — it deserves a deterministic unit-test suite even before a coverage %. |

### 5.4 Cost discipline (startup lens)

- **PR builds** run lint/typecheck/test only — cheap, ubuntu-latest. The expensive multi-arch PyInstaller+Electron+libtorch matrix stays **tag-gated** in `release.yml` (the salvaged `release-desktop.yml` already does this well, including `timeout-minutes: 45` and partial-failure tolerance at `release-desktop.yml:99-101,175`).
- **macOS/Windows runners are ~10x the per-minute cost of Linux.** Keep them out of the PR path; the desktop PR job runs `electron-builder --dir` on Linux only **[PRIOR §4 desktop.yml]**.
- **buf remote plugins** avoid per-CI-run `protoc` toolchain builds — meaningful given the C++ source-built `protoc` today takes minutes.

---

## 6. Linting / formatting / type-check gates (per stack)

This refines the prior audit's P0/P1 lists **[PRIOR §3]** with v2-specific calls.

| Stack | Lint | Format | Type | v2 additions beyond [PRIOR] |
|---|---|---|---|---|
| **Java** | Checkstyle (Google base) | Spotless + palantir-java-format | Error Prone | **ArchUnit** to lock `controller→service→repository` and forbid `ProcessBuilder` outside the FL-lifecycle package (directly enforces the spawn-isolation A1 cares about). |
| **Python** | ruff — widen to `select=["ALL"]` minus noise (`D,COM,ANN1xx,FIX,TD`) | ruff-format | mypy `strict` (already best-in-repo) | Trim the broad `[[tool.mypy.overrides]]` whitelist (`pyproject.toml` lists `flwr.*`, `flwr_datasets.*`, `ray.*`, `opentelemetry.*` etc.) — several are dead deps (A4 flags the `flwr-datasets` leak; B3 flags dead OTel). Removing dead overrides tightens the type net. |
| **TS (frontend)** | ESLint flat → bump to `tseslint.configs.strictTypeChecked` + `eslint-plugin-jsx-a11y` | Prettier | tsc strict | **CSP-aware** lint isn't a thing, but `eslint-plugin-react` + the a11y plugin support C5/A2 (missing CSP, a11y). Add `knip` for dead-export pruning. |
| **TS (desktop)** | ESLint flat → `strictTypeChecked` + `eslint-plugin-security` | Prettier | tsc strict | Electron security lint for `nodeIntegration`/`contextIsolation`/`webSecurity` (A5). |
| **C++ (mobile)** | **clang-tidy** + **clang-format** (NEW) | clang-format | compiler `-Wall -Wextra -Werror` in CI | No C++ standards exist today. A native FL core doing libtorch math with no formatter/linter is a maintainability hole. clang-format config shared in `mobile_client/`. |

**Config-hygiene fixes (carry over + new):**
- Fix the pre-commit ESLint path: it points at `frontend/eslint.config.js` (`.pre-commit-config.yaml`) but the prior audit's matrix and the repo use `eslint.config.mjs` — a silent no-op risk. (New finding vs prior, which only noted the `recommended`-vs-`strict` gap.)
- Replace `prettier v4.0.0-alpha.8` pin (alpha, in a security-adjacent gate) with a stable `v3.x` mirror **[PRIOR §6.4]**.
- Add `.editorconfig`, `.nvmrc` (node 22), `.tool-versions` at root **[PRIOR §6.1-3]**. Unify TS to one version across `frontend`/`desktop` **[PRIOR §6.7]**.

---

## 7. Dependency hygiene & supply chain

The prior audit's Renovate config **[PRIOR §5]** is good; adopt it. v2 escalations:

1. **Spring Boot 3.4.5 is EOL, not merely behind.** 3.4 reached open-source EOL **2025-12-31**; current is 3.5.14 / 4.0.x ([spring.io](https://spring.io/blog/2026/04/23/spring-boot-3-5-14-available-now/), [endoflife.date/spring-boot](https://endoflife.date/spring-boot)). **For a greenfield v2, target the current supported line and pin a managed-version policy.** This is a security exposure (no security patches), not a hygiene nicety — raise to **high**.
2. **Renovate (not Dependabot).** Renovate's grouping (Spring stack as one PR, ML stack monthly, dev-deps automerge) fits a polyglot repo far better; the prior audit's `renovate.json` is fit-for-purpose. Add a `mobile_client/` group for the Android Gradle + NDK deps and a `cmake`/gRPC-version manager note (Renovate's git-submodule/regex managers can pin the `GRPC_VERSION="v1.62.0"` string in the build scripts).
3. **Per-stack vuln scans in CI** (none today): `pip-audit` (framework, client-docker), `npm audit --audit-level=high` (frontend, desktop), OWASP dependency-check or Renovate vuln alerts (backend Gradle), Trivy on the built `client-docker` image **[PRIOR §3]**.
4. **gitleaks in CI + one-time history baseline.** Repo carries historical chat logs **[PRIOR §2 item 3]**; a baseline scan of history is needed before the v2 trunk is locked.
5. **Lockfile discipline.** `client-docker` and `framework/requirements.txt` have unpinned/range deps (`flwr-datasets>=0.3.0`, `protobuf>=4.21.6,<5.0.0` at `framework/requirements.txt:76`). For reproducible FL runs (C3 reproducibility depends on this), pin with hashes — `pip-compile`/`uv` lock for Python, and **note that the `protobuf>=4.x required by flwr` comment is itself a dead-dep artifact** (no `flwr` dependency per platform invariant; A4 confirms the leak). The buf migration (§4) lets that pin float to whatever the buf-generated stubs require, decoupled from a phantom Flower constraint.
6. **SBOM at release** (CycloneDX for gradle/pip/npm) — cheap, and a prerequisite for the B4 compliance floor.

---

## 8. Release engineering & versioning

**Current state:** `version = '1.0.5-beta'` is hand-copied across `backend/fl-platform-api/build.gradle:8`, `frontend/package.json:4`, `fedlearn-desktop/package.json:3`. There are **two** tag-triggered release workflows on the same `v*` tag — a race/duplication footgun (§1). No changelog automation.

**Recommendation for v2:**

1. **Kill `desktop-release.yml`; keep `release-desktop.yml`** as the single release pipeline (it is strictly better — node 22, Python native step, full arch matrix, signing-ready env, partial-failure tolerance).
2. **Independent versioning per deployable unit is the right model**, not one repo-wide version. The backend API, the Python framework (a `pip install -e` package), the desktop installer, and the mobile app version on different cadences. A single `1.0.5-beta` across all three is already lying.
   - Drive each unit's version from `release-please` (Google) or `changesets` (npm-native, good for the JS/TS units) using **conventional commits** (which the existing log already loosely follows — codify with `commitlint` **[PRIOR §3]**).
   - Tag scheme: `backend-vX.Y.Z`, `framework-vX.Y.Z`, `desktop-vX.Y.Z`, `mobile-vX.Y.Z`; the desktop release workflow keys on `desktop-v*`.
3. **The wire contract (`fedlearn.proto`) gets its own version discipline** via buf: `buf breaking` enforces backward compatibility within `fedlearn.v1`; a true break requires a `fedlearn.v2` package. This is the only versioning that *must* be repo-global because all six units share it. This is the cleanest place SemVer actually maps to a real compatibility contract.
4. **Auto-CHANGELOG** from conventional commits (release-please does this) — replaces hand-written release notes (the `release-desktop.yml:206-251` inline body becomes a generated artifact).

> **Note on `mobile_client/` lift:** `00-DESIGN.md:42` defers the subtree lift to a v2 implementation step. The versioning/CI for `mobile.yml`, `proto.yml` C++ leg, and clang-tooling therefore **lands when the subtree lands**, not before. The proto reconciliation (fixing `SubmitModelUpdateReque`) should happen *as part of* that lift, gated by `buf breaking` against canonical.

---

## 9. Prioritized recommendations

### P0 — week 1 (closes the load-bearing gap; near-zero cost)
1. Add `ci.yml` + reusable per-unit workflows on `pull_request` + `push:main`; **enable branch protection requiring them**. (Without branch protection the files are decorative.)
2. Add `proto.yml`: `buf lint` + `buf breaking`. Even before full buf codegen migration, this stops the drift class that already bit the mobile copy.
3. `gitleaks` in CI + history baseline; promote pre-commit from opt-in to CI-enforced.
4. Kill `desktop-release.yml`; keep `release-desktop.yml`.
5. File the Spring Boot EOL upgrade as a tracked P0 security item.

### P1 — weeks 2–4
6. Migrate proto codegen to `buf` for Python + (managed-mode-ready) Java/TS; **spike** the C++/CMake leg before committing mobile to it.
7. Backend static-analysis stack (Spotless + Checkstyle + SpotBugs/find-sec-bugs + JaCoCo + ArchUnit) **[PRIOR §3]**.
8. Renovate config **[PRIOR §5]** + per-stack vuln scans + lockfile pinning (drop the phantom-`flwr` protobuf pin rationale).
9. Frontend vitest + bump both ESLint configs to `strictTypeChecked`; fix the `eslint.config.js`→`.mjs` pre-commit path.
10. Root `Makefile`/`Taskfile` as the one "run all checks" entrypoint; `.editorconfig`/`.nvmrc`/`.tool-versions`.

### P2 — when the mobile subtree lands / at stability
11. `mobile.yml` (RN/TS lint+tsc; CMake configure+compile-check; clang-format/clang-tidy; deterministic DeComFL/ZO unit tests).
12. `release-please`/`changesets` per-unit versioning + conventional-commit `commitlint` + auto-CHANGELOG.
13. SBOM (CycloneDX) at release; license-check.
14. Re-evaluate **Nx** for the JS/TS triangle only if a shared component library (C5) materializes — not before.

---

## 10. Cross-references to sibling agents
- **A4** (client-docker): the `flwr-datasets` runtime-dep leak and the dead-`flwr` protobuf pin in `framework/requirements.txt:76` — §7.5.
- **A6 / B1** (mobile / paper-alignment): the `SubmitModelUpdateReque` proto drift and on-device DeComFL fidelity — the §4 buf migration + §5.3 C++ test gate are the enforcement mechanisms.
- **B3** (observability): dead `opentelemetry.*` mypy override (`pyproject.toml`) is a dead-dep signal — trim per §6.
- **B4** (security/compliance): gitleaks history baseline, SBOM, and EOL-Spring upgrade are compliance-floor prerequisites.
- **B5 / C5** (desktop strategy / design): the Nx-vs-Makefile call (§3) and per-unit versioning (§8) depend on whether desktop stays Electron and whether a shared component library emerges.

---

## Files & sources inspected
**Codebase:** `.pre-commit-config.yaml`; `.github/workflows/desktop-release.yml`, `release-desktop.yml`; `backend/fl-platform-api/build.gradle`; `framework/pyproject.toml`, `framework/requirements.txt`; `framework/src/fedlearn/communication/protos/fedlearn.proto`; `framework/src/fedlearn/communication/generated/`; `frontend/package.json`, `fedlearn-desktop/package.json`; `origin/fed-mobile:mobile_client/scripts/build_grpc_android.sh`, `mobile_client/shared/proto/fedlearn.proto` (drift diff); `docs/audit/2026-05-27/05-tooling-ci.md`; `docs/audit/2026-05-29/00-DESIGN.md`.
**Market/tooling:** buf docs ([generate](https://buf.build/docs/generate/), [managed-mode](https://buf.build/docs/generate/managed-mode/), [remote-plugins](https://buf.build/docs/bsr/remote-plugins/), [C++ SDKs](https://buf.build/blog/bsr-generated-sdks-for-cpp)); monorepo comparison ([daily.dev](https://daily.dev/blog/monorepo-turborepo-vs-nx-vs-bazel-modern-development-teams/), [aviator.co](https://www.aviator.co/blog/monorepo-tools/)); Spring Boot EOL ([spring.io](https://spring.io/blog/2026/04/23/spring-boot-3-5-14-available-now/), [endoflife.date](https://endoflife.date/spring-boot)).
