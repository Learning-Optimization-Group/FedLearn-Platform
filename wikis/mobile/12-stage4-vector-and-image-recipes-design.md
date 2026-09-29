# Stage 4 — Vector and Image Recipes on Android: Focused Design

Stage 3 finished the foundation: streamed, content-addressed artifacts; imported dataset snapshots; seeded
minibatching; dynamic-batch programs; qualification; and the capability report. Every piece was proven on the vivo
with TinyNet and a bit-exact replay of the saved model. Stage 4 ([02](02-android-federated-learning-parity-design.md))
brings the platform's real recipes to the phone with the same guarantee: **a phone update that the contract states
and a replay reproduces**.

## Current state (measured, 2026-09-29)

| Recipe | Input | Classes | Parameters | Laptop training | Notes |
|---|---|---|---|---|---|
| `MLP` (ECG) | vector, 140 | 2 | 13,314 | Adam lr 1e-3, batch 8, 3 epochs (`config.py` "ecg") | **two `Dropout(0.3)`** |
| `CNN` (CIFAR-10) | image 3×32×32 | 10 | 62,006 | Adam lr 1e-3, batch 32 | LeNet: Conv→MaxPool, no BatchNorm |
| `PNEUMONIA_CNN` | image 1×224×224 | 2 | 25,783,554 | Adam | ~103 MB per float32 upload |
| `CIFAR_RESNET18` | image 3×32×32 | 10 | 11,181,642 | Adam | catalog marks it `mobile_safe: false`; 44.8 s/step measured on the vivo |

What the phone can execute today: SGD (optionally with FedProx's proximal term), zeroth-order DeComFL, cross-entropy,
`IdentityVector` inputs, and seeded minibatches. Everything below is what stands between that and the table.

- **Optimizer.** Every non-TinyNet laptop client trains with **Adam**, created fresh each round. The contract schema
  already has `Adam`, but the phone gate refuses anything but SGD.
- **Dropout.** MLP drops activations at random during training. ExecuTorch's dropout RNG is not torch's, so a phone
  update would stop being reproducible. That is the property every Stage 2–3 verification relied on.
- **Images.** `Transform` has only `IdentityVector`, and the importer reads only numbers. There is no image decoding,
  resizing or normalisation on the device.
- **Plans.** `execution_plan.py` has plans for TinyNet only. The exporter (`export_model.py`) writes no contract
  fields (`modelFiles`, `requiredOperators`, envelope, `maxBatch`, probe), so a non-fixture recipe can never get a
  contract.

## Scope

In: `MLP` and `CNN`, first-order (FedAvg, FedOpt, Robust, FedProx) and DeComFL, on own-data snapshots, mixed with
laptops where the data allows.
Deferred: `PNEUMONIA_CNN`, until qualification measures a real resource budget (upload size and step time on
a phone). `CIFAR_RESNET18` stays off phones for first-order training (catalog `mobile_safe: false`) and is revisited
with accelerators in Stage 7. Text is Stage 5.

## A. Contracts for every recipe, not just TinyNet

`execution_plan.py` becomes recipe-driven:
- the trainable layout and frozen-state digest come from the recipe's model;
- the data requirement comes from a new recipe `input_spec` (shape, dtype, transforms);
- local training comes from the same settings the laptop client uses (`config.py` / client constants). Those settings
  move into one place both read, so a laptop and a phone are held to the same numbers.

The exporter gains the contract fields the TinyNet stager writes: program digests and sizes, the operator set (read
from the programs with ExecuTorch, as `test_stage_model_bundle` does), a declared envelope, `maxBatch` and the probe.
It records them per recipe at export time. The backend host still needs the ExecuTorch toolchain to export, which is
unchanged.

## B. Adam on the device

A native Adam for the trainable program: per-parameter first and second moments, bias correction, and
`reset_optimizer_each_round` (the laptop creates a fresh Adam every round, so no state crosses rounds and none needs
checkpointing). Pinned like SGD was: a torch golden over several steps, a tolerance derived from the smallest
deviation it must catch, and a mutation test per term (bias correction, epsilon placement, β₂).

## C. Dropout

Options (decision 1):

- **Seeded dropout masks, as inputs.** Mobile programs are exported with dropout replaced by multiplication with a mask
  input. The mask for each step comes from a specified stream, SplitMix64 seeded by (run seed, round, epoch, step,
  layer), the same construction as the batch order. A replay regenerates it exactly. Laptop clients stay free to use
  torch dropout, because their updates are replayed by their own hypothesis. *The phone's update stays bit-exact.*
- **Dropout off in contracted training.** Both laptop and phone train without dropout. Simple, but it changes the
  laptops' training, and so the model the recipe produces.
- **Accept it.** The phone uses ExecuTorch's dropout; verification becomes statistical. Gives up the bit-exact
  replay for MLP.

*Recommendation: seeded masks.* Its cost is one export wrapper and one mask generator in C++ and Python, both
pinnable by a golden, and it keeps the verification that caught every Stage 2–3 bug.

## D. Images

Options (decision 2):

- **Raw pixels in the package.** `records.jsonl` carries each image as its uint8 values (HWC) with its height, width
  and channels. Transforms are exact arithmetic declared in the contract: `ScaleToUnit` (/255), `Normalize(mean, std)`
  and a layout change to CHW. No decoder is involved, so Python and Kotlin produce bit-identical tensors.
- **Image files (PNG/JPEG) decoded on the device.** More convenient for a user, but decoders differ (JPEG especially,
  and resizing filters), so the contract would need a tolerance and the replay would no longer be exact.

*Recommendation: raw pixels first*, with PNG decoding (lossless, so exact at a fixed size) as a follow-up and JPEG
only with a declared tolerance. CIFAR-10's 32×32×3 is 3 KB per example as uint8 and 12 KB as a float snapshot; a
thousand examples is 12 MB, well inside the importer's limits.

## E. Order and proof

1. **MLP** first: no image pipeline, the smallest real recipe. It exercises A, B and C.
2. **CNN** next: exercises D. It is the first real Conv/MaxPool training on a phone under a contract; the Conv→MaxPool
   backward defect was fixed in the ExecuTorch build on 2026-08-06.

Each recipe gets:
- the per-strategy plans;
- host parity against LocalTrainer (Adam, minibatches, and for MLP, masks);
- qualification;
- a live phone-only run with a bit-exact replay and wrong-hypothesis controls;
- then a mixed phone + laptop run.

## Decisions needed before implementation

1. **Dropout** (section C): seeded masks (recommended), dropout off, or accept non-reproducibility.
2. **Image input** (section D): raw pixels first (recommended), or decode image files from the start.
3. **Scope**: MLP and CNN in Stage 4, with pneumonia CNN deferred to measured budgets (recommended), or all four.

## Slices (each test-first, each its own commit)

- **S0** Recipe `input_spec` and recipe-driven plans; the laptop and the phone read one settings source.
- **S1** Exporter writes the contract fields for any recipe (digests, operators, envelope, `maxBatch`, probe).
- **S2** Native Adam, with a torch golden and per-term mutation tests; the phone gate accepts Adam.
- **S3** Seeded dropout masks (if decision 1 picks it): a mask-input export, generators in C++ and Python, and a golden.
- **S4** MLP end to end, live on the vivo with a replay.
- **S5** Image package, `ScaleToUnit`/`Normalize`/`ToChw` transforms, and a Kotlin importer with a cross-language golden.
- **S6** CNN end to end, live on the vivo with a replay; then a mixed run.

## Risks

- **ExecuTorch operator coverage.** Adam is done natively, not in the graph, so the risk is the backward operators of
  Conv/MaxPool/Dropout-as-mask. The qualification probe catches a device where one misbehaves.
- **Laptop drift.** Moving the laptop's training settings into a shared source must not change a laptop run. A test
  pins the resolved numbers before and after.
- **Snapshot size for images.** Float snapshots are 4× the uint8 source. Snapshots could store uint8 and transform
  per batch; that is deferred until a dataset needs it.
