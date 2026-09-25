# DeComFL in Execution Contract v1 — Focused Design

**Status:** Approved (2026-09-25). Being implemented in the slices below.

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
