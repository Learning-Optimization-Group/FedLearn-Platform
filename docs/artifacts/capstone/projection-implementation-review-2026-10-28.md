# Projection implementation: premeasurement review

28 October 2026. Code reviewed before any qualification/main cells: `e110ba9..e405aac`. Corrections frozen at `1e44031`; prospective main manifest and qualification receipt frozen at `0ecd8af` before main fitting. This review establishes specified implementation/evidence checks, not methodological novelty.

## Material findings and corrections

| Finding | Reproduction and correction | Verification |
| --- | --- | --- |
| Qualification could accept hash-consistent failed/zero-fit records without attack or independent audit evidence. | A generated non-fitting fixture reproduced admission of twelve failed rows. Validation now requires completed20-round/400-fit records, exact artifact inventory, matching hashed result contents, finite positive runtimes, replaying attack-check evidence and a complete independent audit. Main execution revalidates its prerequisite. | `test_hash_consistent_zero_fit_qualification_is_refused` failed before the correction and passes afterward. |
| The auditor did not bind the frozen manifest or recorded source checkpoint. | A generated fixture with empty settings/data/bases/source/calibration metadata and a false plan digest was accepted. Audit now checks the frozen digest, complete resolved design, generated data/basis/calibration/runtime identities and immutable checkpoint source contents. | `test_hash_consistent_audit_with_invalid_provenance_is_refused` failed before the correction and passes afterward; six field-specific binding checks also pass. |
| Noise and geometric diagnostics were not independently reconciled. | A generated private fixture with zero noise and negative scale/energy was accepted. Audit now checks exact declared seeded noise for both output rows/biases, theoretical and realized energies/score effects, clean aggregate norm and lifted teacher alignment. Saved raw clean uploads support independent clipping calculations. | Missing geometry check reproduced before correction; zero private noise, wrong coordinate mapping and eight individually altered diagnostic scalars are refused afterward. |

Two smaller findings were corrected: independent scoring now casts each output row to float64 before subtraction; distribution checks now cover native/full-width draws and active–inactive, cross-row and bias covariance. No review findings were deferred.

## Verification scope

- Scientific/generated subset: **61 checks passed** after fixes, including existing projection/owned-attack helper checks.
- Combined existing CPU/transport and relevant canonical strategy/GDP/RNG/attack profile: **352 passed**, with151 existing UTC-deprecation warnings. Generated local gRPC socket tests ran with permitted socket access.
- Complete reproduction-directory collection: **six errors**, all legacy scikit-learn imports absent from the intentionally minimal profile. No dependency changes were made and this is not a full-directory green claim.
- The qualified twelve-cell run completed once:240 rounds and4800 client fits. Its numerical audit, runtime and source bindings are in the [qualification receipt](projection-qualification-2026-10-28.json).

No medical data, holdout or test artifacts were reopened. No source code in the historical framework or medical runners was changed. New code and checks live under the capstone reproduction directory.
