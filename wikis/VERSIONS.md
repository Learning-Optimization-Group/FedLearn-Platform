# FedLearn-Platform — Component Versions

This file is the single source of truth for the release version of each
deployable unit. Update the relevant row whenever a component ships a
tagged release. Versions follow [Semantic Versioning](https://semver.org).

## Current versions

| Component | Version | Package file |
|-----------|---------|--------------|
| **backend** (Spring Boot API) | `1.4.1-beta` | `backend/fl-platform-api/build.gradle` |
| **framework** (Python FL core) | `0.1.0` | `framework/setup.py` |
| **frontend** (React web app) | `1.4.1-beta` | `frontend/package.json` |
| **desktop** (Electron app) | `3.1.0-beta` | `fedlearn-desktop/package.json` |
| **mobile** (React Native app) | `2.1.0` | `mobile_client/package.json` |
| **client-docker** (container) | — | `client-docker/Dockerfile` (inherits framework) |

## Versioning policy

- **backend / frontend**: versioned together; they share the same REST API
  contract and are always deployed as a pair.
- **framework**: versioned independently; bumped when the gRPC contract,
  aggregation strategies, or public Python API changes.
- **desktop / mobile**: versioned independently; each follows its own
  release cadence aligned with platform store requirements.
- **client-docker**: not independently versioned; the Docker tag mirrors
  the framework version it bundles.

## How to bump a version

1. Update the version string in the package file listed above.
2. Update the table in this file.
3. Tag the commit with the per-unit release prefix: `git tag <component>-v<version>`
   (e.g. `desktop-v3.1.0-beta`, `mobile-v2.1.0`). The prefix scopes the CI
   release workflow so only the matching unit builds.

## Release history

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
