# CI foundation & branch protection (v2 milestone M0)

This is the PR-time gate the audit named the single highest-leverage gap (Theme 4: *"every other
defect compounds because nothing gates PRs"*; the only prior workflows fired on tag only).

> **CI** = Continuous Integration · **PR** = Pull Request · **SBOM** = Software Bill of Materials.

## What runs

| Workflow | Trigger | Jobs |
|---|---|---|
| [`ci.yml`](../../.github/workflows/ci.yml) | PR + push to `main` | `changes` (path filter) → per-stack: **framework** (pytest), **backend** (`./gradlew test`, `test` profile), **frontend** (lint + `tsc --noEmit` + build), **desktop** (lint + jest). Each runs only when its directory changes. |
| [`security.yml`](../../.github/workflows/security.yml) | PR + push + weekly | **gitleaks** (secret scan), **pip-audit** (framework), **npm audit** (frontend + desktop), **CycloneDX SBOM**. |
| [`mobile.yml`](../../.github/workflows/mobile.yml) | PR + push (mobile paths) | proto-mirror + python/cpp parity gates (the DeComFL golden-vector test). |

`renovate.json` keeps dependencies current (torch is held for manual review — it is pinned to the
DeComFL golden-vector fixture version). `.editorconfig` / `.nvmrc` / `.tool-versions` pin the
cross-editor + toolchain defaults.

## Required: turn on branch protection (one-time, repo admin)

CI only *gates* merges once these checks are **required** on `main` (GitHub → Settings → Branches
→ Branch protection rule for `main`, or via `gh`):

- Require a pull request before merging (≥ 1 approving review).
- Require status checks to pass: add the jobs you want mandatory — at minimum `framework`,
  `backend`, and the `security` secret-scan; add `frontend`/`desktop`/`mobile` as those stacks
  mature. (Path-filtered jobs that are skipped report success, so requiring them is safe.)
- Require branches to be up to date before merging.
- Do not allow force-pushes / deletions on `main`.

```bash
# Example (adjust the contexts to the jobs you want mandatory):
gh api -X PUT repos/:owner/:repo/branches/main/protection \
  -f required_pull_request_reviews.required_approving_review_count=1 \
  -F required_status_checks.strict=true \
  -f 'required_status_checks.contexts[]=framework' \
  -f 'required_status_checks.contexts[]=backend' \
  -f enforce_admins=true
```

## Notes / verify-before-use

- The **framework** job is locally verified (pytest green); the **backend/frontend/desktop** jobs
  are written to each subproject's commands but need a first CI run to confirm (network, lockfiles).
- The two legacy on-tag desktop workflows (`desktop-release.yml`, `release-desktop.yml`) are
  duplicates of the same release; consolidating them is a separate cleanup (audit B7).
- Action + tool versions are pinned but flagged verify-before-use.
