# FedOpt and Robust in Execution Contract v1 — Plan

**Goal:** Put FedOpt and Robust back on the phone by adding them to the approved v1 matrix, so a TinyNet FedOpt
or Robust run publishes a contract that the laptop and Android clients both execute exactly.

**Why now:** Android refuses every run without a READY v1 contract (the strict decision recorded in
[06](06-execution-contract-v1-implementation-plan.md)). Stage 1 validated FedOpt and Robust live on the phone,
and the strict gate took them away. This restores them without weakening the gate.

## What the clients actually execute

The [parity design](02-android-federated-learning-parity-design.md) already says both are ordinary first-order
client training. Server-side adaptation and robust aggregation stay on the server. Measured in the code:

| Strategy | What the FL server sends each round | What both clients train |
| --- | --- | --- |
| FedAvg | nothing | the client default: SGD lr 1e-3, 1 epoch |
| Robust | nothing (`RobustAggregator` supplies no client config) | the same as FedAvg |
| FedOpt | `learning_rate=0.01`, `local_epochs=1` (the `FedOpt` strategy's defaults, which `fl_server.py` does not override) | SGD lr 0.01, 1 epoch: the laptop via `learning_rate_override`, the phone via the native server settings |

So the contract needs **no new schema fields.** The strategy enum plus `localTraining` fully state what a
client does, and server-only settings (FedAdam betas, the robust method) are not client behaviour. The
FedOpt contract must state lr 0.01, and it must come from the same value the server sends.

## A behaviour this slice changes

The native first-order round lets server-sent `learning_rate` / `local_epochs` **override** the values it was
given ([05](05-first-order-round-config-plan.md)). Under a contract, that means the server rather than the
contract decides what the phone trains. It is invisible today only because a FedAvg server sends nothing.
From this slice on, the contract is authoritative: a server value must **equal** the contract's, or the round
is refused before any upload. The laptop client gets the same guard, since it also lets the server config
override. FedOpt still requires the server to send both values, as a cross-check that the server really runs
the configured FedOpt.

## Slices (each test-first, each its own commit)

1. **One source for FedOpt's client training.** A torch-free constant module in `fl-runtime/` that
   `fl_server.py` passes to `FedOpt(...)` explicitly and the resolver reads. Test: the config the server's
   FedOpt sends equals the resolved plan.
2. **Resolver plans.** `TINYNET_GOLDEN` / `FedOpt` / `FULL` and `Robust` / `FULL`. Tests: each plan equals the
   optimizer, step budget and batching `client.py` really builds under that strategy's server config; DeComFL
   and FedProx stay not representable.
3. **Matrix, in all three readers.** Add the two rows to the approved matrix (README), the generator's corpus,
   and the Python, Java and TypeScript validators in one commit, with corpus cases for each new row and
   the combinations that must stay refused.
4. **Backend publication.** A FedOpt or Robust TinyNet run publishes READY. The legacy/v1 equivalence
   check maps both strategies.
5. **Laptop guard.** Under a contract, a server-sent learning rate or epoch count that differs from the
   contract refuses the round.
6. **Phone.** `projectContract` accepts both strategies and projects the real strategy. The native round
   refuses a server value that disagrees with the contract instead of adopting it. FedOpt still requires
   the server values. Jest and C++ tests; the C++ endpoint test for a server-chosen rate becomes "the
   contract's rate, confirmed by the server".
7. **Live runs.** Fresh APK; one FedOpt run and one Robust run with the vivo and three laptop clients,
   checked numerically the way Stage 2I was.

## Not in scope

DeComFL (needs a schema extension for the zeroth-order path), FedProx (needs a native proximal term), any
recipe other than TinyNet, and describing server-side aggregation settings in the contract.
