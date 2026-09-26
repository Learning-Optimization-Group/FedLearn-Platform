# FedProx in Execution Contract v1 — Plan

**Goal:** Put FedProx on the phone by adding it to the approved v1 matrix, so a TinyNet FedProx run publishes a
contract that the laptop and Android clients both execute exactly. It is the last strategy the phone refuses
(see [06](06-execution-contract-v1-implementation-plan.md), "Android is v1-dependent").

**Status:** Slices 1–5 done (2026-09-26); the live run (slice 6) is pending a connected phone. See [Result](#result).

## What the clients actually execute

FedProx aggregates exactly like FedAvg. Its whole difference is client-side: every local step adds the proximal
gradient `mu * (w - w_global)` after backward and before the optimizer step, where `w_global` is the round's
downloaded model (`_apply_proximal_gradient` in `fl-runtime/client.py`, `LocalTrainer.fit` in the framework).

| What the FedProx FL server sends each round | What both clients train |
| --- | --- |
| `proximal_mu=0.1`, `learning_rate=0.01`, `local_epochs=1` (`fl_server.py`'s μ and the `FedProx` strategy's defaults) | SGD lr 0.01, 1 epoch, plus the proximal gradient at μ 0.1 |

The schema already has the field: `ModelTraining.fedprox_mu`, present exactly when the strategy is FedProx, and
all three validators enforce that. The Java assembler keeps the resolver's `ModelTraining` as-is, so μ is owned
by the Python resolver and reaches the published contract without a backend code change.

## The proximal term is zero on the contract's TinyNet round

TinyNet's data is one batch of 8, and the contract batches by 8, so one epoch is **one** SGD step. At the first
step of a round the weights equal `w_global`, so the proximal gradient is exactly zero. A FedProx TinyNet round
at the server's real settings is therefore bitwise identical to FedOpt's client training (SGD lr 0.01 × 1).

Consequences, stated rather than hidden:

- A live FedProx run at these settings shows the phone accepts, projects and completes a FedProx contract. It
  **cannot** show the proximal term is right: a client with no proximal term produces the same bytes.
- The native proximal term is therefore proven against a **multi-step** framework golden (`LocalTrainer.fit` with
  μ > 0 over several steps), where the term is non-zero, and that test also asserts the endpoint differs from
  the μ = 0 endpoint by more than its tolerance.
- The contract still states the settings the server really runs. Changing FedProx's client epochs to make the
  term visible would change every FedProx run, and is left as a separate decision.

## Slices (each test-first, each its own commit)

1. **One source for FedProx's client training, and the resolver plan.** Torch-free constants that `fl_server.py`
   passes to `FedProx(...)` and the resolver reads. The FedProx plan states SGD lr 0.01 × 1 and `fedprox_mu` 0.1;
   `check_contract` compares `fedprox_mu` too. Tests: the plan equals what `client.py` builds under the server's
   config, including μ; the other strategies' plans carry no μ.
2. **Matrix, in all three readers,** with corpus cases for the new row and the combinations that stay refused.
3. **Backend publication.** A FedProx TinyNet run publishes READY with μ in its contract.
4. **Laptop guard.** Under a contract, a server-sent `proximal_mu` that differs from the contract's (absent = 0)
   refuses the round, on every first-order strategy.
5. **Phone.** A multi-step FedProx golden from the framework; the native trainer applies the proximal gradient
   against the round's downloaded weights; the round refuses a server μ that disagrees with the contract;
   `projectContract` accepts FedProx and passes μ to native.
6. **Live run.** Fresh APK; one FedProx run with the vivo and laptop clients, with the limit above stated.

## Not in scope

Any recipe other than TinyNet, changing FedProx's server-side defaults, and describing server-side settings in the
contract.

## Result

| Slice | Commit |
| --- | --- |
| 1. One source for FedProx's client training; resolver plan | `04dd07b` |
| 2. Matrix in all three readers | `9cc468a`: 177-case corpus; each reader fails the 3 changed cases with its matrix reverted |
| 3. Backend publication | `8e7e2a5`: no backend code change was needed; the test fails with FedProx removed from the matrix |
| 4. Laptop guard | `be0b41d` |
| 5. Phone | `f1a78c6`: native endpoint 2.98e-8 from the framework golden; fails with the anchor removed |
| 6. Live run | not run yet: no device attached |

Measured while doing it (`research/notes/on-device/2026-09-26-fedprox-proximal-term-is-invisible-at-production-settings.md`):

- **At one local step FedProx is byte-identical to FedAvg** at every rate and coefficient swept. At the production
  lr 0.01 / μ 0.1, even ten steps separate the endpoints by only 6.4e-5. Raising FedProx's epochs would not make a
  live run discriminate, so the server settings were left alone.
- **The inherited tolerance would have passed a trainer with no proximal term.** At the FedAvg golden's lr 0.1 × 5,
  μ 0.1 moves the endpoint 1.37e-3, inside that golden's 2e-3. The FedProx golden has its own 1e-4.
- The APK builds with the change for arm64-v8a, but it has not been installed on a device.
