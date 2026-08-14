# FedLearn-Platform — Component Versions

**This file is the single source of truth for the release version of each
deployable unit.** Nothing else in the repo is — a version quoted in a README,
a wiki page or a design doc is a copy, and this table wins when they disagree.
Update the relevant row whenever a component ships a tagged release. Versions
follow [Semantic Versioning](https://semver.org).

## Current versions

Verified against the package files on 2026-08-13.

| Component | Version | Package file |
|-----------|---------|--------------|
| **backend** (Spring Boot API) | `1.4.1-beta` | `backend/fl-platform-api/build.gradle` (`version = '1.4.1-beta'`) |
| **framework** (Python FL library) | `0.1.0` | `framework/setup.py` (`version='0.1.0'`) |
| **frontend** (React web app) | `1.4.1-beta` | `frontend/package.json` |
| **desktop** (Electron app) | `3.2.0-beta` | `fedlearn-desktop/package.json` |
| **mobile** (React Native app) | `2.1.0` | `mobile_client/package.json` |
| **client-docker** (container) | — | `client-docker/Dockerfile` (tag mirrors the framework version it bundles) |

`fl-runtime/` is the seventh deployable unit but carries **no version string of
its own**: it has no package file, and it ships inside the same artifacts as the
framework (the `client-docker` image and the desktop PyInstaller bundle both
package `framework/` **and** `fl-runtime/`). Treat it as versioned with the
framework until it gains a package file of its own.

## Versioning policy

- **backend / frontend**: versioned **together, as a pair**; they share the same
  REST API contract and are always deployed together. A change to one that does
  not touch the other still moves both rows.
- **framework**: versioned **independently**; bumped when the gRPC contract,
  aggregation strategies, or public Python API changes.
- **desktop / mobile**: versioned **independently** of everything else and of
  each other; each follows its own release cadence aligned with platform store
  requirements.
- **client-docker**: **not independently versioned**; the Docker tag mirrors the
  framework version it bundles.

## How to bump a version

1. Update the version string in the package file listed above.
2. Update the table in this file. Never do one without the other — a package
   file and this table that disagree is the failure this file exists to prevent.
3. Tag the commit with the per-unit release prefix: `git tag <component>-v<version>`
   (e.g. `desktop-v3.1.0-beta`, `mobile-v2.1.0`). The prefix scopes the CI
   release workflow (`release-desktop.yml` / `release-mobile.yml`) so only the
   matching unit builds.

Not every shipped change is a release. Several substantial cycles below landed
with **no** version bump in any unit; those are recorded here as history with
the versions they left unchanged, rather than being omitted.

## Release history

### 2026-08-12 → 2026-08-13 — no version bump

| Component | Version | Notes |
|-----------|---------|-------|
| — | unchanged | `flwr` / `flwr-datasets` dropped repo-wide (`65048b6`), clearing the SE-22 `cryptography<45.0.0` cap and an unnoticed `protobuf<5.0.0` one; the CIFAR-10 IID shard is now native in `fl-runtime/recipes.py` and was verified byte-identical per partition before the swap |
| — | unchanged | In-process simulator (`d4e91f3`): `framework/src/fedlearn/simulation/` drives the production `FLCoordinator` and strategies by direct call — no gRPC, no port pool — measured to 5,000 clients |
| — | unchanged | Training arms become a first-class, persisted project property: `FULL` / `FROZEN_HEAD` (`V22`) and then `OVA_LP` (`V23`, `886749d`), where an arm carries an **objective** and not only a trainable subset. `CIFAR_RESNET18` joins the recipe catalog as the first pretrained-backbone recipe |

No unit shipped a tagged release in this cycle; every version above stayed where
2026-06-16 left it. Recorded here because the platform's capabilities moved even
though its version numbers did not.

### 2026-07-17 — Ledger design system — no version bump

| Component | Version | Notes |
|-----------|---------|-------|
| frontend | `1.4.1-beta` (unchanged) | `2c50672` replaced the Ember tokens with **Ledger** in `design/tokens.json` and regenerated every per-platform output; `fdd8a79` restyled the representative frontend views onto Ledger the same day (and fixed a latent class-merge bug) |
| desktop | `3.2.0-beta` (unchanged) | renderer `tokens.css` regenerated from the same source of truth |
| mobile | `2.1.0` (unchanged) | `src/theme/tokens.generated.ts` + `global.css` regenerated; the `2.1.0` row below still reads "Ember theme", which is correct **as history** — that version's tokens were later replaced in place by Ledger without a bump |

Ledger is *navy structural ink on quiet paper surfaces*, light-first: canvas
`#F6F3EE`, surface `#FFFFFF`, ink `#191A1C`, muted `#6B6760`, accent `#1C314D`
(hover `#14243A`); the dark family is navy-dark (canvas `#0B1622`, accent
`#4F8AC9`) and stays wired. Type is Hanken Grotesk for **both** sans and display
plus JetBrains Mono, with `'calt' 1, 'tnum' 1`. It superseded **Ember** (2026-06-10),
which had superseded **Instrument** (2026-06-09) — three cycles, of which only
Ledger is current. Bricolage Grotesque was Ember's display face and survives only
in `design/brand/*.html` comparison assets, not in the token font stack.

`design/tokens.json` is the single source of truth and `design/build-tokens.mjs`
generates the per-platform artifacts; CI has a "Design tokens in sync with source
of truth" step, so a hand-edited generated file fails the build. This is why the
rebrand touched three units and bumped none of them: no unit shipped a tagged
release, so under the policy above no version row moved.

### 2026-06-16

| Component | Version | Notes |
|-----------|---------|-------|
| desktop | `3.2.0-beta` | Model Playground — interactive "Use a model" inference (image + vector), reached over validated IPC with the JWT kept in the keychain |

Backend and web also gained the inference feature this cycle (new `/api/inference` API and `/playground` view) but are versioned as a pair and stay `1.4.1-beta`.

### 2026-06-10

| Component | Version | Notes |
|-----------|---------|-------|
| desktop | `3.1.0-beta` | Ember design-system rebrand (renderer + rebranded icons) |
| mobile | `2.1.0` | Ember theme + brand fonts; buildable iOS/Android app projects with native FL-core wiring |

Frontend also received the Ember rollout this cycle but stays `1.4.1-beta`
(versioned as a pair with the unchanged backend). Repository root was
reorganized (diagrams/VERSIONS into `wikis/`, `renovate.json` into `.github/`).

### 2026-06-09

| Component | Version | Notes |
|-----------|---------|-------|
| backend | `1.4.1-beta` | FoT server scripts; project lifecycle fixes |
| framework | `0.1.0` | FoT module (gRPC contract, agent, distiller, 11 tests); DeComFL correctness fixes |
| frontend | `1.4.1-beta` | Instrument design system; dark-default theme; UI primitives |
| desktop | `3.0.4-beta` | Instrument tokens; emoji → lucide icon migration |
| mobile | `2.0.0` | Instrument design system; useThemeTokens hook |
