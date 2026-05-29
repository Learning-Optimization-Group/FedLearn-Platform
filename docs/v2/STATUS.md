# FedLearn v2 — Implementation Status

**Branch:** `temp` (the session's v2 work, on the `origin/main` lineage)
**As of:** 2026-05-29
**Companion docs:** design lives in `docs/v2/build/` (HLD/LLD/tech-stack/data-model/API); the audit that drives all of it is in `docs/audit/2026-05-29/`.

> **Legend**
> ✅ **Done & verified here** (built/tested on a clean checkout in this environment)
> 🟡 **Code-complete; build/test gated on CI or a build machine** (toolchain not available here)
> 🔵 **Designed (LLD written); not yet built**
> ⛔ **Blocked / non-engineering gate**

---

## 1. Summary

FedLearn v2 is a greenfield rebuild informed by the 18-report audit. This session delivered the
**design corpus**, the **P0 DeComFL correctness fix**, the **complete mobile FL (Federated
Learning) client in code**, the **canonical v2 gRPC proto**, the **PR-time CI (Continuous
Integration) foundation**, and a **backend Gradle-wrapper fix**. The remaining platform units are
**designed but not built**. Everything below is on `temp`; `main-clean` is preserved separately
(see §6).

---

## 2. Completed this session

| Area | Status | Evidence |
|---|---|---|
| Audit (18 expert reports + verified synthesis) | ✅ | `docs/audit/2026-05-29/` (21 files) |
| v2 design docs (HLD, tech-stack, data-model, API contracts, 9 unit LLDs, build sequence, local-model guide, controls checklist) | ✅ | `docs/v2/build/` (18 docs) |
| **DeComFL correctness fix** — Bug 1 (1/P factor + O(K·P) loop hoist), Bug 2 (CPU-canonical RNG wiring), B-2 (process-global RNG hygiene), wiki note | ✅ | framework suite **59 passed, 1 skipped** on a clean checkout |
| Determinism contract — `canonical_perturbation` + frozen golden vectors | ✅ | bit-exact reproduction verified (`test_perturbation.py`) |
| Canonical v2 proto `fedlearn.v2` + `buf` config + mirror checksum gate | ✅ (gate verified) | `proto/`; `scripts/check_proto_mirror.sh` passes |
| **Mobile FL client** (React Native + native C++ libtorch) — see §3 | 🟡 | code-complete; build gated (no CPU libtorch / NDK / Xcode / buf here) |
| PR-time CI foundation — `ci.yml`, `security.yml`, `renovate.json`, `.editorconfig`, `.nvmrc`, `.tool-versions` | 🟡 | workflow YAML valid; framework leg green; branch protection is a manual repo-admin step |
| Backend Gradle-wrapper fix (committed the wrapper jar, un-ignored) | ✅ | `./gradlew --version` → `Gradle 8.10.2` on a clean checkout |

---

## 3. Mobile FL client — per-increment (15-LLD §13 tasks 1–19)

The mobile unit is **complete in code, end-to-end**. Only the determinism contract is runnable in
this environment; the native/app layers are gated on a build machine.

| Increment (tasks) | What | Status |
|---|---|---|
| Determinism contract (2–5) | `canonical_perturbation`, golden vectors, parity tests | ✅ verified bit-exact |
| C++ FL core (6–8) | `ModelManager`, `ZerothOrderEstimator`, `DeComFLClient`, Sha256 | 🟡 gtests gated on CPU libtorch |
| C++ gRPC layer (9–10) | `FedLearnClient` (dual-channel TLS+mTLS, streaming, heartbeat-abort), `FederatedLoop`, `DataLoader` | 🟡 opt-in target; needs buf stubs + gRPC runtime |
| TurboModule bridge (11–13) | typed spec + CXX TurboModule + Android JNI + iOS provider | 🟡 needs RN New-Arch codegen |
| RN app + screens (14–15) | `lib/` (join flow, native wrapper, device-class), Training/Library/Testing screens (real softmax), navigation | 🟡 needs `npm`/Metro |
| Prebuild + CI (18–19) | `export_model.py` (verified: 1M/10M tiers), `build_{libtorch,grpc}_arm64.sh`, `fetch_demo_data.sh`, `mobile.yml` | 🟡 `export_model.py` verified; scripts syntax-checked; `mobile.yml` valid |
| App projects + lifecycle (16–17) | Android/iOS Gradle/Xcode projects, foreground service, native device-metrics provider | 🟡 needs Gradle/Xcode/NDK |

**Mobile build-out remaining** (template scaffolding + cross-cutting, not new FL logic):
- iOS Xcode project (`FedLearn.xcodeproj`/`.xcworkspace`) and a real release signing config (Android Gradle wrapper now exists for the backend; the mobile RN project needs its own from a fresh `react-native@0.80` init template).
- The shared `@fedlearn/tokens` OKLCH design-system package (the screens use a local placeholder).
- On-device training-data wiring (`FedLearnCoreModule::setTrainingDataFromFiles`).
- Set repo variable `MOBILE_NATIVE_CI=true` once the project builds, to enable the `.so`-size gate.

---

## 4. Verified on a clean checkout (this environment)

A pristine `git worktree` at `temp`'s commit, fresh virtualenv, no working-tree contamination:
- **Framework `pytest`: 59 passed, 1 skipped** (DeComFL fix + correctness + perturbation + existing suite).
- **Workflow YAML** (`ci.yml`, `security.yml`, `mobile.yml`) all valid.
- **Proto-mirror gate** passes (sha match).
- **Backend bootstraps** (`./gradlew --version` → 8.10.2).

## 5. Gated on CI / a build machine (toolchains absent in this sandbox)
- Mobile **C++ core** — needs a **CPU libtorch** (the installed `torch` is a CUDA build; `find_package(Torch)` errors without CUDA libs). `mobile.yml`'s `cpp-parity` job downloads CPU libtorch and runs the parity gtests; the Python golden-vector contract it checks against **is** verified here.
- Mobile **RN / Android / iOS / bridge / proto codegen** — need React Native toolchain / NDK (Native Development Kit) / Xcode / `buf`.
- Full **backend** test (`./gradlew test`), **frontend**, **desktop** — need network + lockfiles (not session work for backend/frontend/desktop — those are unchanged from `origin/main`).

---

## 6. Remaining — designed (LLD), not built

The rest of the v2 platform has LLDs in `docs/v2/build/` but no implementation yet. From the audit's
prioritized queue (`docs/audit/2026-05-29/README.md` §5):

| Unit | Status | Notes |
|---|---|---|
| Identity / RBAC + managed-Postgres cutover | 🔵 | **Prerequisite** for the orchestration substrate; `origin/main` lacks the V5 identity layer (`organizations`, `projects.org_id`). |
| FL orchestration substrate | 🔵 | `FlServerLauncher` (k8s Jobs primary) + `fl_runs` lease + reconciler + round deadline/quorum + per-org quotas. **Blocked** on identity + Postgres above. |
| Backend control plane rebuild | 🔵 | role enum, org-scoped authz, per-run scoped result tokens, ProjectService decomposition. |
| Observability stack | 🔵 | Micrometer/Prometheus + Grafana/Loki/Tempo + OTel; W3C traceparent JVM→Python→client→mobile; incremental RoundResult POST. |
| Data & artifact plane | 🔵 | dataset registry + content-addressed S3/MinIO + MLflow + determinism manifest. |
| Security & compliance | 🔵 | gRPC default-secure (TLS+mTLS), org isolation, DP layer, SOC 2 Type 2 + HIPAA-readiness. |
| Frontend rebuild + `@fedlearn/tokens` | 🔵 | TanStack Query, CSP/HSTS, Vitest/Playwright; shared OKLCH design system. |
| Desktop → Tauri v2 | 🔵 | signed minisign updater, Rust command layer, fail-closed IPC bridge. |

**Non-engineering gate (above everything):**
- ⛔ **RIT IP resolution for DeComFL** — under RIT policy, the IP is likely RIT-owned; an IPMO license/spin-out is the go/no-go gate the whole product is downstream of (`docs/audit/2026-05-29/C4-business-gtm-ip.md`).

---

## 7. Branch & lineage note

- **`temp`** — all of this session's work, on the **`origin/main`** lineage.
- **`main-clean`** — a separate, preserved branch (older/divergent framework: a dict-keyed
  `seed_history` and the V5 identity layer that `origin/main` lacks). The DeComFL fix on `temp`
  was applied to `origin/main`'s framework; **`main-clean`'s framework still carries the same
  DeComFL bugs** and would need the fix re-applied if it ever becomes the v2 base.
- The orchestration substrate's foundation gap (§6) stems from this divergence: the v2 design
  assumes `main-clean`'s V5 identity + Postgres, which are not on the `origin/main` lineage.

---

## 8. Suggested next steps

1. **Decide the v2 base** (resolve the `temp` vs `main-clean` divergence) — this gates the
   identity + orchestration work.
2. **Identity/RBAC + Postgres cutover**, then the **orchestration substrate** (the audit's P1
   architectural core).
3. **Finish the mobile build-out** (iOS project, signing, `@fedlearn/tokens`) → flip
   `MOBILE_NATIVE_CI=true` for a green native gate.
4. **Enable branch protection** so `ci.yml`/`security.yml`/`mobile.yml` become required checks
   (see `docs/v2/ci-and-branch-protection.md`).
