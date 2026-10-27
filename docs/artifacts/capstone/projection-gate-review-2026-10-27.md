# Projection design review and phase status

27 October 2026. Scope: the [projection study design](../../superpowers/specs/2026-10-27-projection-mechanism-design.md), not implementation acceptance.

## Review findings and resolutions

1. **Shape-dependent noise breaks naive padding coupling.** Canonical `ComposedStrategy._pipeline` draws full tensors sequentially; the same seed at width 128 and 512 does not couple both weight rows and biases. Resolution: a study-only adapter reuses parent clipping/reduction with noise disabled, then adds explicitly mapped active/bias draws and independently generated inactive draws. The production strategy remains unchanged. Qualification must check coordinate marginals, covariance, coupling and native no-noise parity before any main fits.
2. **The attack damaging-control budget was ambiguous.** Three effort budgets can yield different training trajectories. Resolution: apply the mean 0.02 Full512 no-noise AUROC-damage check separately to each budget; publish all values and mark the failing budget's robustness interpretation inconclusive. Do not qualify one budget using another budget's damage.

A fresh builder review reported no important remaining questions from these resolutions. This establishes design implementability only; the adapter, runner and training controls have not been implemented or tested.

## Current phase status

| Phase | Status |
| --- | --- |
| User scope | Projection extension authorized; later DP-BREM, backbone/LoRA and FLARE/failure work queued subject to time. |
| Literature gate | Targeted primary-source inspection complete. Existing projection/privacy/clipping theory overlaps substantially; originality remains unestablished. |
| Analytical reconnaissance | 10,000 generated one-release noise draws; zero training fits. Padding preserves coupled scores despite 3.975597× observed total noise energy. [Receipt](projection-gate-analytical-2026-10-27.json). |
| Detailed design | Reviewed; 136 proposed main cells, plus at most twelve qualification cells. Concrete-design approval requested before implementation. |
| Implementation and qualification | Not started. No main outcomes inspected. |
| Medical studies | Frozen; no medical data, holdout, test inputs or models reopened. |
| Later projects | Not started. |

Next after design approval: write the implementation plan with exact qualification-cell allocation and red/green tests, then implement the generated-only helpers and canonical strategy adapter. Measure qualification runtime before freezing or starting main fits. The 48-hour gate deadline is 29 October 2026, 19:38:44 UTC; it is a maximum preparation window, not an instruction to wait.

The [contribution audit](contribution-audit-2026-10-27.md) remains the assessment of the pre-extension capstone. No new study result or novelty claim supersedes it.
