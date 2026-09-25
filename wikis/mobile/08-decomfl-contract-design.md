# DeComFL in Execution Contract v1 — Focused Design

**Status:** Done (2026-09-25). All slices landed, and a vivo 1805 and three laptop clients completed a live DeComFL run on a published contract. See [Result](#result).

**Goal:** Put DeComFL back on the phone. It is the last strategy the strict v1 gate refuses (see
[06](06-execution-contract-v1-implementation-plan.md), "Android is v1-dependent"). Unlike FedOpt and Robust
([07](07-fedopt-robust-contract-plan.md)), DeComFL is not first-order training, so a matrix row alone cannot
describe it: v1 has no way to state a zeroth-order update.

## What a DeComFL client executes today (measured in code)

Each round the server's `GetDeComFLConfig` sends the round's seeds (a K × P matrix) and a config map. Both client
types then run the same loop. For each of K local steps, on the next batch of the local loader (cycling), for each
of P seeds:

- draw z from the seed;
- estimate g = (L(x + μz) − L(x)) / μ (forward difference);
- after the P draws, apply x ← x − (η/P) Σ g·z.

They upload only the K × P scalars g.

| Setting | Source today | Laptop (`decomfl_client.py`) | Phone (`FederatedLoop::deComFLRound`) |
| --- | --- | --- | --- |
| K (local steps), P (perturbations) | shape of the server's seed matrix | from the seeds | from the seeds |
| η (learning rate) | server config; `config.py` default 0.001 | applied locally, `x -= (η/P)Σgz` | applied by the native loop |
| μ (smoothing) | server config; default 0.001 | server value wins ("FR-10: μ is server-authoritative") | server value |
| Estimator | server `grad_estimate_method`, "forward" | forward | forward, via `etGScalarForward` (parses "central" but runs forward) |
| z generator | seeded CPU `torch.randn`, float32 | `torch.randn` | `RandnEngine`, byte-identical to `torch.randn` (golden vectors in `randn_parity_test`) |
| Batching | the recipe's loader | batch 8, shuffled, cycled across steps | the staged batch |

So DeComFL clients are entirely **server-driven**. Nothing a client executes is pinned before the run: the
server's config decides it, round by round. That is exactly what the contract exists to prevent (the same
override finding as FedOpt in 07, only here nothing else is involved).

## Proposed schema extension (additive; `buf breaking` must pass)

```proto
message LocalTraining {
  ...
  oneof optimizer {
    Sgd sgd = 4; Adam adam = 5; AdamW adamw = 6; Rmsprop rmsprop = 7;
    ZerothOrderSgd zeroth_order_sgd = 12;   // new
  }
}

// DeComFL's client update. The server issues each round's seeds (num_local_steps x num_perturbations);
// every other number is stated here, and a round whose config disagrees is refused.
message ZerothOrderSgd {
  double learning_rate = 1;              // eta, applied per local step as x -= (eta / P) * sum_p g_p z_p
  double smoothing = 2;                  // mu, the finite-difference step
  uint32 num_local_steps = 3;            // K: one batch of the (cycled) local loader per step
  uint32 num_perturbations = 4;          // P
  GradientEstimator estimator = 5;
  PerturbationRng rng = 6;
}

enum GradientEstimator { ESTIMATOR_UNSPECIFIED = 0; ESTIMATOR_FORWARD = 1; ESTIMATOR_CENTRAL = 2; }
// How z is drawn from a seed. A client whose generator is not byte-identical must refuse.
enum PerturbationRng { RNG_UNSPECIFIED = 0; RNG_TORCH_CPU_RANDN_F32 = 1; }
```

## Decisions (each with a recommendation)

1. **Zeroth-order as an optimizer in the existing oneof, not a separate block.** It is the update rule. It is also
   mutually exclusive with SGD/Adam, which the oneof enforces in every generated reader. *Recommended.*
2. **K lives in `ZerothOrderSgd`, not `local_epochs`.** DeComFL counts steps, cycling the loader, not epochs. With
   `zeroth_order_sgd`, `local_epochs` must be 0 and `max_local_steps` absent. Today 0 epochs is always refused, so
   the rule widens only for this optimizer. The alternative, reusing `local_epochs` as K, would give one field two
   meanings. *Recommended.*
3. **The RNG is part of the contract.** A client whose generator differs computes scalars for different directions,
   and the aggregate is silently wrong. The server's advisory `torch_version` / `golden_vector_sha256` fields
   (unset today) become unnecessary for contracted runs. *Recommended.*
4. **Seeds stay per round, from the server.** They are data, not training settings; the contract fixes their shape
   (K × P). *Recommended.*
5. **Server config is a cross-check, not an override.** η, μ, the estimator and the seed-matrix shape must equal the
   contract's, or the round is refused before upload. That is the rule 07 established for FedOpt, applied to both
   clients. *Recommended.*

## Validation rules (added to the fixtures README; one corpus case per rule, all three readers)

- `zeroth_order_sgd` present: `learningRate` or `smoothing` not finite and positive (a zero is absent) →
  `OUT_OF_RANGE`; plain doubles, like `Sgd.learning_rate`, because zero is never legal. `numLocalSteps` or `numPerturbations` 0 or above the v1 limit → `OUT_OF_RANGE`. `estimator` or
  `rng` not known → `UNKNOWN_ENUM`. `localEpochs` non-zero or `maxLocalSteps` present → `INVALID_STRATEGY_SETTINGS`.
- `updateProtocol` is `UPDATE_DECOMFL_SCALAR` exactly when the optimizer is `zeroth_order_sgd`, and the strategy is
  `STRATEGY_DECOMFL` exactly then; otherwise `INVALID_STRATEGY_SETTINGS`.
- Approved matrix: add TinyNet / DeComFL / FULL / vector classification / cross-entropy / `UPDATE_DECOMFL_SCALAR`.
- LightSecAgg stays valid only for DeComFL. The phone refuses it (`UNSUPPORTED_SECURITY`), as it does today.

## Plan (each slice test-first, its own commit)

1. Schema + mirrors + generated code in all languages. `buf lint`, `buf breaking` against `main`, and the mirror
   check must pass.
2. Rules + corpus + the three readers; each reader must fail the new cases with its change reverted.
3. Resolver plan for TinyNet / DeComFL / FULL, stated from the values `fl_server.py` hands `DeComFL(...)`, from one
   source both import (as `strategy_client_settings` does for FedOpt).
4. Laptop: `check_contract` and a per-round guard in the DeComFL client.
5. Phone: the gate projects `zeroth_order_sgd`, and the native DeComFL round cross-checks the server's config (μ, η,
   K, P, estimator) against the contract.
6. Live: a fresh APK; the vivo and three laptops on one published DeComFL contract, checked numerically. Without
   secure aggregation.

## Not in scope

The central estimator on the phone (it runs forward only, so a contract stating central is refused there),
LightSecAgg on the phone, recipes other than TinyNet, and the rebuild-history protocol for clients that miss rounds
(unchanged).

## Result

| Slice | Commits |
| --- | --- |
| 1. Schema | `2084e7c` |
| 2. Rules, corpus, three readers | `4ab4c6c`: 175 cases; each reader fails exactly the 19 new or changed ones with its change reverted |
| 3. Resolver plan | `db5f24f`, backend publication pinned in `f84df71` |
| 4. Laptop guard | `d17015b` (framework round check, estimator passthrough), `c2eefd3` |
| 5. Phone | `6af4264`, `f0b98d6`, `bb065c8` |
| Found by the live run | `426b8e8`: phone DeComFL started from an all-zero model |

**Two phone bugs the live run exposed**, both fixed:

1. **Wrong config keys.** The native client read the server's DeComFL rate and smoothing under `lr` / `mu`, which
   the server never sends (`learning_rate` / `smoothing_param`), so it always trained on its 0.001 defaults
   (`6af4264`).
2. **The phone computed every DeComFL scalar at an all-zero model.** The DeComFL path never downloaded the global
   model, and the ModelManager zero-initialises its parameters. The first live run showed a constant phone loss
   of 1.1221, exactly TinyNet's loss with its trainable parameters zeroed. The phone now downloads the round-1
   model, proves it is the contract's `initialStateSha256`, refuses to join after round 1, and reports loss on the
   model it trains from (`426b8e8`).

   A replay attributes 1.6×10⁻⁴ of error in the buggy run's final model to it, about 18% of the run's total
   parameter movement. This affected every phone DeComFL run before the fix.

Live, after the fix (run `b3162471…`, contract `1c3fa059…`): COMPLETED, with 4 of 4 clients reporting in each of
3 rounds and no errors. The phone's losses were 1.0973 → 1.0972 → 1.0970. A replay through the real server strategy
matches all three and the saved final model to 7×10⁻⁸. Records are in `research/results/multiplatform/decomfl_*`.

Not exercised live: a server config that disagrees with the contract (unit-tested on both clients), a phone
joining after round 1 (refused), and LightSecAgg (refused on the phone).
