# B5 — Desktop FL Orchestrator: Framework Strategy (Electron vs Tauri vs Native vs Thin-Shell-over-C++-Core)

**Audit unit:** `fedlearn-desktop/` (Electron orchestrator) + reuse analysis of `origin/fed-mobile:mobile_client/shared/` (portable C++ FL core)
**Date:** 2026-05-29
**Branch:** `main-clean`
**Builds on:** `docs/audit/2026-05-27/02-frontend-desktop.md` (C5 unsigned auto-update RCE, C3 `'unsafe-eval'`, C1 desktop JWT-in-body, H10 fail-open bridge) and `README.md` P0 #4.

---

## 0. The explicit question, answered up front

> Build separate native Win/Mac/Linux apps, or keep Electron, for the desktop FL orchestrator?

**Recommendation: KEEP a web-tech shell, but MIGRATE Electron → Tauri v2, and DO NOT build per-OS native apps. SEPARATELY, do NOT make the desktop a thin shell over the mobile C++ core in v2 — keep the spawn-a-subprocess orchestration model.** These are two orthogonal decisions and the assignment conflates them; I separate them below.

Net verdicts:

| Decision | Verdict |
|---|---|
| Per-OS native (Swift/WinUI/GTK) | **kill** — 3x the UI surface for a 4-person team, no payoff for an orchestrator UI. |
| Electron (current shell) | **refactor → Tauri** — the shell is salvageable in *function* but Tauri removes the C5 auto-update RCE class, cuts bundle ~90%, and shrinks the attack surface. Migrate, don't rewrite-from-scratch. |
| Thin shell over mobile C++ core (desktop does on-device training in-process) | **kill for v2** (keep as a v3 research spike) — it does NOT remove the libtorch size burden, it *adds* a fragile cross-language RNG-parity invariant, and it collapses the process-isolation boundary that is the desktop's main safety property. |
| The subprocess orchestration model itself (`DockerService`) | **salvage** — it is the correct architecture; only the shell around it should change. |

The rest of this report is the evidence.

---

## 1. What the desktop actually is today (correct the framing)

The assignment frames the desktop as "orchestrating Docker." That is only the **Jetson** path. The dominant path is a **bundled native subprocess**, and Docker is the exception:

- `docker.service.ts:90-96` — `startTraining()` routes `jetson` → Docker, *everything else* (`mps`/`cuda`/`cpu`) → `startNativeProcess()`.
- `docker.service.ts:218-261` — the "native" binary is a **PyInstaller bundle of the Python client** (`fedlearn-client` / `fedlearn-client.exe`), shipped via `electron-builder.yml:40-44 extraResources`. In dev it falls back to system `python3 client-docker/scripts/client.py` with `framework/src` on `PYTHONPATH`.
- So today the desktop ships **CPython + PyTorch frozen by PyInstaller**, not C++ libtorch, and not Docker (except Jetson).

This matters enormously for the decision: **the heavy dependency (PyTorch/libtorch) is already in the bundle regardless of shell.** dockerode is a *minor* dep used on exactly one hardware profile.

**dockerode footprint:** `package.json:38` (`dockerode ^4.0.10`), unpacked from asar (`electron-builder.yml:48`) because it touches the Docker socket. It is the only non-trivial native-ish runtime dep besides `electron-*`. Removing Electron does not remove dockerode — the Jetson path still needs a Docker client. (Tauri would call the Docker socket from Rust via `bollard`, the Rust equivalent — see §6.)

---

## 2. The mobile C++ core — what it really is, and whether desktop can reuse it

`origin/fed-mobile:mobile_client/shared/src/` is a **complete, portable C++17 FL client** — not a stub. Evidence:

- `CMakeLists.txt` — links `libtorch` (`find_package(Torch REQUIRED)`, `LIBTORCH_DIR`), `gRPC::grpc++`, `protobuf::libprotobuf`, `nlohmann_json`. Builds `fedlearn_core` STATIC from 7 `.cpp` files (~1,300 LOC total) + generated protos from the **same `fedlearn.proto`** as the platform.
- `FedLearnClient.h` — dual gRPC channel (`channel_` for training, `heartbeat_channel_` for heartbeat on a background `std::thread`) — i.e. it **already implements the parallel-heartbeat invariant** in C++. `CHUNK_SIZE = 50 * 1024 * 1024` — it **already implements parameter chunking** (50 MB; note: the platform rule says >300 MB triggers chunking, framework default differs — a parity check is needed).
- `DeComFLClient.h` / `ZerothOrderEstimator.{h,cpp}` — full DeComFL: ZO gradient scalar `g = (f(x+mu*z) - f(x)) / mu`, seed-based perturbation replay (`rebuildModel`, Algorithm 2), gradient-scalar upload. This is the paper's core (Yang/RIT, comm O(K×P)).
- `FederatedLoop.h` — both `startFedAvg()` and `startDeComFL()` on a background thread, with a `Status` struct (phase/round/loss/accuracy/step) — exactly the telemetry the orchestrator wants to surface.
- `NativeFedLearnCoreImpl` is a **React-Native TurboModule** bridge (`NativeFedLearnCore.mm` is ObjC++ for iOS; Android JNI under `android/app/src/main/jni/`). It is wired to **React Native**, not to a generic C ABI.

So "could desktop reuse it for real on-device training?" — **mechanically yes** (it's portable C++, same proto, MPS/CUDA/CPU all reachable via libtorch). But four things make it the **wrong v2 move**:

### 2.1 It does NOT remove the libtorch size burden
Web research confirms libtorch is **~267 MB CPU, ~1.2 GB CUDA, ~1.9 GB for CUDA 11** ([PyTorch forums](https://discuss.pytorch.org/t/libtorch-cuda-so-is-too-large-2gb/103155), [pytorch#34058](https://github.com/pytorch/pytorch/issues/34058)). The C++ core *links libtorch* — `CMakeLists.txt` `find_package(Torch REQUIRED)`. So a C++ desktop client ships the *same* multi-GB torch payload as the current PyInstaller bundle. **The shell choice (Electron 80-150 MB vs Tauri <10 MB) is noise next to a 267 MB-1.9 GB torch core.** You do not adopt C++ to shrink the bundle.

### 2.2 It ADDS a fragile cross-language correctness invariant
`ZerothOrderEstimator.cpp` comment: *"C++ `torch::Generator` uses the same Mersenne Twister as Python, producing identical outputs."* DeComFL **requires bit-identical perturbations** between server-issued seeds and client-replayed perturbations (`rebuildModel`), or the global model diverges silently. Today the client is the same Python/PyTorch as the framework — RNG parity is free. Introduce C++ and you create a **second implementation of the most numerically sensitive code path**, validated only by an unstated assumption about libtorch's RNG matching across language bindings *and across libtorch versions*. This is a latent correctness landmine, flagged in 2026-05-27 framework report (the chunked-upload asymmetry was exactly this class of "two paths drift" bug). For a startup, a second FL client implementation is a permanent ~2x maintenance tax on the part of the system that *is the product*.

### 2.3 It collapses the process-isolation boundary
Today, training runs in a **separate OS process** (`spawn()` at `docker.service.ts:292`, or a container). If training segfaults, OOMs, or hangs, the UI survives and `stopTraining()` (`docker.service.ts:102-126`, SIGTERM→SIGKILL escalation) reaps it. libtorch on heterogeneous consumer GPUs **will** crash. Linking the C++ core *in-process* (TurboModule-style, as mobile does) means a torch crash takes the whole app down. The subprocess model is a *feature*, not overhead — it is the desktop's main robustness property. Keep it.

### 2.4 It is bound to React Native, not a clean C ABI
The bridge surface (`NativeFedLearnCoreImpl`, `NativeFedLearnCore.mm`) is RN TurboModule/JSI, not a `extern "C"` library or a CLI. Reusing it on desktop means either (a) re-exposing it as a CLI binary (then it's just *another subprocess* — at which point the existing PyInstaller subprocess already works and is in the same language as the server), or (b) FFI into Tauri's Rust (cxx/bindgen across a libtorch-heavy C++ API — high effort, see §6). Neither yields a "thin shell" simply.

**Conclusion on C++ reuse:** The C++ core's real value is **mobile** (where you cannot ship CPython/PyInstaller and where RN already hosts it). For desktop, building it into the shell trades a clean, language-homogeneous subprocess for a 2x-maintained, in-process, RNG-fragile path with no bundle-size win. Verdict **kill for v2**; revisit only if you later ship a *single* universal C++ client used by *both* mobile and desktop and retire the Python client entirely (a v3 consolidation play — see §8).

---

## 3. Why per-OS native (Swift/WinUI/GTK) is a kill

The desktop is an **orchestrator UI**: auth modal, hardware selector, project list, log drawer, settings, an update banner (`src/renderer/views/*`, `components/*` — ~20 files). It is not a graphics-intensive or OS-deeply-integrated app.

- **3x UI surface.** Three codebases (SwiftUI + WinUI/WPF + GTK/Qt), three a11y stories, three test stacks, three idioms. For a 4-person startup this is a non-starter — the 2026-05-27 report already flags **zero React-frontend tests** (M7); triple that surface and the coverage gap triples.
- **No reuse of the existing React frontend.** The web dashboard (`frontend/`) and desktop renderer already share React + the same API/STOMP contracts. Going native throws that away.
- **The only "native" thing the app needs** — OS keychain for the JWT (`safeStorage` per CLAUDE.md; today `electron-store`, see C1) and a Docker-socket client — is a thin platform shim, not a reason to rewrite the UI three times.
- **Code-signing burden is identical or worse** — you still sign 3 OSes; you just also maintain 3 UIs.

Verdict: **kill.** No production-startup rationale survives the maintenance math.

---

## 4. Electron vs Tauri — the real 2025-2026 trade-offs

Web research (cite inline; treat vendor blogs as directional, not gospel):

| Dimension | Electron (today) | Tauri v2 | Source |
|---|---|---|---|
| Installer size (shell only) | 80-150 MB (bundles Chromium + Node) | <10 MB (system WebView + Rust) | [gethopp](https://www.gethopp.app/blog/tauri-vs-electron), [pkgpulse](https://www.pkgpulse.com/guides/electron-vs-tauri-2026) |
| Idle RAM | ~150-300 MB | ~30-50 MB | same |
| Cold start | ~1-2 s | <0.5 s | [rustify](https://rustify.rs/articles/rust-tauri-vs-electron-2026) |
| Rendering consistency | Identical Chromium everywhere | System WebView (WebView2/WKWebView/WebKitGTK) — *can differ* | [pkgpulse](https://www.pkgpulse.com/guides/electron-vs-tauri-2026) |
| Security default | Must harden (contextIsolation, CSP, nodeIntegration off) | Capability system, deny-by-default | same |
| Backend language | Node/TS | Rust | — |

**Caveat I must flag (no fabrication):** the headline size/RAM numbers are *shell-only*. This app ships a **267 MB-1.9 GB torch payload as extraResources** regardless of shell. So the realistic installer goes from e.g. ~120 MB (Electron shell) + ~300 MB-1.9 GB (torch) to ~10 MB (Tauri shell) + same torch. The **percentage win is real but the absolute win is bounded by torch.** Be honest in the v2 plan: Tauri's bundle headline does not apply to the torch payload.

**Where Tauri genuinely wins for *this* app — and it's about security and the C5 finding, not size:**

1. **Auto-update signing is enforceable.** Tauri's updater **requires** signed update artifacts (minisign keypair) — unsigned updates are rejected by the framework itself. This directly kills the 2026-05-27 **C5 critical** (`updater.ts:13-14` `autoDownload=true` + `autoInstallOnAppQuit=true` over unsigned GitHub releases = supply-chain RCE). With Electron you must *add* checksum/signature verification; with Tauri it is the default contract.
2. **Smaller renderer attack surface.** Removes the `'unsafe-eval'` packaged-CSP issue (C3) — the renderer is system WebView, and the Rust core is not a Node runtime an XSS can pivot into. Addresses the spirit of C4/H10 (fail-open bridge) by making the privileged surface a typed Rust command set, not arbitrary IPC.
3. **No Node in the privileged process.** The whole class of "renderer reaches Node" Electron CVEs disappears.

**Where Electron wins / Tauri costs:**

- **Rust skill requirement.** The backend command layer (spawn subprocess, talk to Docker socket, keychain) moves to Rust. For a Java/Python/TS team this is a real ramp cost. Mitigate: the Rust surface here is *small* (spawn/kill a child, stream stdout, one Docker call, one keychain call) — hundreds of lines, not thousands.
- **WebView fragmentation.** WebView2 on Windows (must be present — Evergreen runtime is on Win11 by default, bootstrappable on Win10), WKWebView on macOS, WebKitGTK on Linux. The dashboard is Tailwind + recharts + framer-motion; verify framer-motion/recharts render acceptably on WebKitGTK before committing (low risk for a dashboard, but **test it** — I cannot guarantee pixel parity).
- **Migration cost.** Renderer (React) ports almost as-is; the `src/main/*` services (`docker.service.ts`, `auth.service.ts`, `updater.ts`, `ipc.handlers.ts`, `hardware.probe.ts`) must be reimplemented as Tauri commands in Rust. ~5 files, well-scoped. Estimate: 2-4 weeks for one engineer comfortable ramping Rust, given the renderer is reusable.

**Decision:** the deciding factor is **not** size — it's that Tauri makes the C5 RCE class *structurally impossible* and shrinks the privileged surface, for a team that is the right size to absorb a small Rust core. Migrate.

---

## 5. Code-signing & notarization — the burden is OS-count, not framework

This is **identical across Electron, Tauri, and native** — you sign per-OS artifacts either way. Costs (web research):

- **macOS:** Apple Developer Program **$99/yr**; notarization itself is **free** and unlimited once enrolled ([developer.apple.com/programs](https://developer.apple.com/programs/)). Requires a Developer ID cert + hardened runtime. Today `electron-builder.yml:67-69` is `identity: null`, `hardenedRuntime: false`, `gatekeeperAssess: false` — i.e. ad-hoc, which is why macOS shows "damaged" and why C5 exists.
- **Windows:** EV/OV Authenticode certs run **~$200-580/yr** (DigiCert EV ~$549-581, Sectigo EV ~$297-453) ([ssl2buy](https://www.ssl2buy.com/ev-code-signing-certificates), [signmycode](https://signmycode.com/digicert-ev-code-signing)). **Azure Trusted Signing ~$9.99/mo (~$120/yr)** is the cheap path but **US/Canada-only and requires an established legal entity** (verify current eligibility — it was restricted as of 2025). For an RIT-affiliated startup, Azure Trusted Signing is likely the best cost/effort if entity requirements are met; otherwise OV Authenticode.
- **Linux:** no mandatory signing (AppImage/deb unsigned is normal); GPG-sign repos optionally.

**Framework-specific signing notes:**
- Tauri documents Windows signing including **Azure Trusted Signing** ([v2.tauri.app/distribute/sign/windows](https://v2.tauri.app/distribute/sign/windows/)) and macOS signing/notarization; its updater signs artifacts with **minisign** independently of OS code-signing — defense in depth.
- **Watch item (do not fabricate certainty):** Tauri sidecar binaries have had signing friction — [tauri#11778](https://github.com/tauri-apps/tauri/issues/11778) (sidecar fails to sign with Azure Trusted Signing) and [tauri#9981](https://github.com/tauri-apps/tauri/issues/9981) (sidecar `NotFound` at spawn). **If you ship the torch client as a Tauri `externalBin` sidecar, the sidecar must itself be signed/notarized**, and these issues must be re-checked against the current Tauri release before committing. This is a concrete migration risk, not a blocker.

**Total annual signing cost, any framework:** ~$99 (Apple) + ~$120-580 (Windows) = **~$220-680/yr**. This is a rounding error for a startup and **must be paid regardless of framework** — so signing burden does **not** differentiate Electron vs Tauri vs native. It *does* argue against per-OS native only insofar as native gives you nothing back for the same spend.

---

## 6. Hardware/CPU/CUDA/Jetson support matrix under each option

The training *engine* is decoupled from the shell — this is the key realization that frees the shell choice.

| Profile | Today (Electron) | Under Tauri |
|---|---|---|
| macOS MPS (arm64) | spawn PyInstaller native bundle | spawn sidecar (same bundle) via Rust `Command` |
| Windows CUDA | spawn PyInstaller CUDA bundle | same |
| Windows CPU | spawn PyInstaller CPU bundle | same |
| Linux/Jetson | dockerode → L4T container w/ `/dev/nvhost-*` mounts (`docker.service.ts:41-48`) | Rust `bollard` (Docker API client) → same container/mounts |

- **Jetson invariant preserved:** the prohibition on `--runtime nvidia` and the explicit `JETSON_DEVICE_MOUNTS` device list (`docker.service.ts:41-48, 344-350`) is *data*, not Electron-specific code. It ports verbatim to a Rust `bollard` `HostConfig`.
- **discrete GPU:** `DeviceRequests: [{ Count: -1, Capabilities: [['gpu']] }]` (`docker.service.ts:352`) maps to bollard's equivalent.
- **dockerode → bollard** is the one non-trivial Rust port. bollard is mature and covers `createContainer`/`start`/`logs(follow)`/`stop`/`remove` and the multiplexed-stream demux that `demuxDockerStream` (`docker.service.ts:474-502`) hand-rolls today (bollard demuxes for you — a *simplification*).

**Recommendation that survives either shell:** keep training as a **subprocess/sidecar**, keep the hardware-profile dispatcher (`docker.service.ts:90-96`) as the single source of truth, and treat the shell as a thin spawner + log streamer + auth/keychain holder. This is already the architecture; preserve it.

---

## 7. Security posture mapped to the prior audit

| 2026-05-27 finding | Status under recommendation |
|---|---|
| **C5** unsigned auto-update RCE (`updater.ts:13-14`, `electron-builder.yml:65-71`) | **Structurally fixed** by Tauri's mandatory signed-updater. Phase-0 interim (before migration): set `autoInstallOnAppQuit=false`, ship signing config. |
| **C3** `'unsafe-eval'` packaged CSP (`renderer/index.html:7`) | Eliminated — Tauri renderer is system WebView; tighten CSP in `tauri.conf.json`. |
| **C1** desktop accepts JWT in response body (`auth.service.ts:122-125`) — defeats HttpOnly cookie contract | **Independent of shell — must fix regardless.** Pin to OS-keychain storage of the cookie/token; renderer never sees it (CLAUDE.md `safeStorage` rule). In Tauri this lives in the Rust command layer. |
| **H10** bridge fail-open `preview-user` (`renderer/App.tsx:82-152`) | Must fail-closed regardless of shell; Tauri's typed command invocation makes "preload missing" non-representable. |
| gRPC plaintext over WAN (audit #37) | Unchanged — it's a framework/transport concern, not the shell's. Out of scope here; tracked elsewhere. |

---

## 8. Concrete recommendation & phasing for v2

**Architecture for v2 desktop:**
- **Shell:** Tauri v2. Reuse the React renderer; reimplement `src/main/*` as a small Rust command layer (spawn/stream/kill subprocess; `bollard` for Jetson; keychain for token).
- **Training engine:** unchanged subprocess/sidecar model. Keep the Python client (PyInstaller) as the desktop engine for v2 — it is language-homogeneous with the server, so DeComFL RNG parity is free.
- **C++ core:** stays mobile-only in v2. Document the long-term option (§8.1).

**8.1 The one scenario where C++-on-desktop becomes right (v3 watch-item):**
If/when you decide to **retire the Python client entirely and ship one C++ `fedlearn_core` everywhere (mobile + desktop)**, then a Tauri desktop FFI-ing (cxx) into `fedlearn_core` — or spawning it as a signed CLI sidecar — becomes coherent: one client implementation, one RNG path to validate, mobile+desktop parity. Gate this on (a) a passing **bit-identical perturbation parity test** between the C++ core and the Python server across libtorch versions, and (b) a CI matrix that builds `fedlearn_core` for mac-arm64/win-cuda/win-cpu/jetson. Until both exist, the dual-implementation tax is unjustified.

**Phasing:**
- **Phase 0 (week 1, in Electron, no migration yet):** `autoInstallOnAppQuit=false` (`updater.ts:14`); fill `identity`/`hardenedRuntime`/notarization in `electron-builder.yml`; fail-closed the bridge (H10); fix C1 token-in-body. These are the bleeding-stoppers and they buy time.
- **Phase 1 (weeks 2-5):** Stand up the Tauri v2 shell; port renderer; reimplement spawn + log-stream + keychain in Rust; wire minisign updater. Keep Python client as sidecar.
- **Phase 2 (weeks 6-8):** Port Jetson path to `bollard`; re-verify the `JETSON_DEVICE_MOUNTS` and discrete-GPU `DeviceRequests`; **re-check Tauri sidecar-signing issues (#11778/#9981) against current release**; sign all 3 OS artifacts (Apple $99 + Azure Trusted Signing or Authenticode).
- **Phase 3 (later / optional):** evaluate §8.1 C++ unification with a perturbation-parity gate.

**Do NOT:** build 3 native UIs; do NOT link the C++ core in-process; do NOT chase Tauri for bundle size alone (torch dominates).

---

## 9. Decision table (verdicts)

| Module / option | Verdict | One-line rationale |
|---|---|---|
| Per-OS native (Swift/WinUI/GTK) | **kill** | 3x UI maintenance for an orchestrator UI; zero payoff for a small team; same signing cost. |
| Electron shell (current) | **refactor → Tauri** | Function is salvageable but Tauri structurally kills the C5 auto-update RCE and shrinks the privileged surface. |
| Tauri v2 shell | **rebuild (target)** | Reuse renderer; reimplement the ~5 main-process services as a small Rust command layer. |
| Thin shell over mobile C++ core (in-process, v2) | **kill** | No bundle win (torch dominates), adds fragile RNG-parity invariant, collapses process isolation, RN-bound bridge. |
| C++ core for mobile | **salvage** | Correct and complete where CPython can't ship; keep it mobile-only. |
| Subprocess/sidecar orchestration model (`DockerService` dispatcher) | **salvage** | Correct architecture; only the host shell changes. |
| dockerode (Jetson) | **refactor → bollard** | Same capability in Rust; bollard also removes the hand-rolled stream demux. |
| Auto-updater (`updater.ts`) | **rebuild** | Move to Tauri signed-updater (minisign); fixes C5 by construction. |
| `electron-builder.yml` signing config | **rebuild** | Replace ad-hoc `identity:null` with real Apple + Windows signing in the Tauri pipeline. |

---

## 10. Open uncertainties (explicitly flagged, not papered over)

1. **WebKitGTK rendering parity** for framer-motion/recharts on Linux is unverified — must be smoke-tested before committing the migration. Low risk for a dashboard; not zero.
2. **Tauri sidecar signing** ([#11778](https://github.com/tauri-apps/tauri/issues/11778), [#9981](https://github.com/tauri-apps/tauri/issues/9981)) — open issues as of the cited dates; re-validate against the Tauri release you target. If unresolved, fall back to spawning the client as a non-sidecar external binary you sign yourself.
3. **Azure Trusted Signing eligibility** — US/Canada + established legal entity as of 2025; confirm current terms for an RIT-affiliated entity before budgeting $120/yr vs ~$300-580/yr Authenticode.
4. **C++/Python RNG parity** for DeComFL perturbations is *asserted* in `ZerothOrderEstimator.cpp` comments but **not proven by any test** in the mobile tree I read. This is the single biggest technical risk in any future C++ unification (§8.1) and the strongest reason to keep desktop on the Python client for v2.
5. Vendor blog size/RAM numbers (gethopp, pkgpulse, rustify, tech-insider) are directional marketing; the *shell-only* deltas are credible and corroborated across sources, but the *absolute* installer size for this app is dominated by torch, not the shell.

---

### Sources
- [Tauri vs Electron — gethopp](https://www.gethopp.app/blog/tauri-vs-electron)
- [Electron vs Tauri 2026 — pkgpulse](https://www.pkgpulse.com/guides/electron-vs-tauri-2026)
- [Tauri vs Electron 2026 — rustify](https://rustify.rs/articles/rust-tauri-vs-electron-2026)
- [Tauri Embedding External Binaries (sidecar) — v2 docs](https://v2.tauri.app/develop/sidecar/)
- [Tauri Windows Code Signing — v2 docs](https://v2.tauri.app/distribute/sign/windows/)
- [tauri#11778 — sidecar Azure Trusted Signing](https://github.com/tauri-apps/tauri/issues/11778)
- [tauri#9981 — sidecar spawn NotFound](https://github.com/tauri-apps/tauri/issues/9981)
- [Apple Developer Program](https://developer.apple.com/programs/)
- [EV code signing certs — ssl2buy](https://www.ssl2buy.com/ev-code-signing-certificates)
- [DigiCert EV code signing — signmycode](https://signmycode.com/digicert-ev-code-signing)
- [libtorch_cuda.so >2GB — PyTorch forums](https://discuss.pytorch.org/t/libtorch-cuda-so-is-too-large-2gb/103155)
- [libtorch size — pytorch#34058](https://github.com/pytorch/pytorch/issues/34058)
