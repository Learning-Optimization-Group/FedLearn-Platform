# TinyNet First-Order Optimizer Parity Implementation Plan

**Goal:** Make the existing TinyNet desktop first-order training step use the same SGD rule as Android, without changing other recipes, and prove the endpoint against a one-step reference.

**Architecture:** The recipe already declares SGD and Android's `TrainableExecutorchModel::trainStep` performs SGD. Resolve the desktop optimizer from the TinyNet recipe before training; leave other recipe branches unchanged. This is the first executable prerequisite for [execution contract v1](03-android-execution-contract-v1-design.md), not full mobile recipe or strategy parity.

**Tech stack:** Python 3.12, PyTorch, pytest; existing React Native/C++ TinyNet golden fixtures remain unchanged.

## Constraints

- Work on `mobSupp`; commit locally and do not push.
- Write and run a failing regression test before changing the training path.
- Preserve Adam/AdamW behavior for every non-TinyNet recipe.
- Do not mark contract v1 ready or claim FedProx/text/image/GPU support in this slice.

## Task 1: Align the effective TinyNet desktop optimizer

**Files:**

- Modify: `fl-runtime/client.py` in `train()`'s optimizer-selection block.
- Test: `fl-runtime/tests/test_client_tinynet_golden_fedavg_wire.py`.

**Interface:** `train(net, trainloader, epochs, dataset_name, ...)` keeps its signature. For `MODEL_TYPE == "TINYNET_GOLDEN"`, it constructs `torch.optim.SGD` on trainable parameters with learning rate `CNN_LEARNING_RATE` (`0.001`), zero momentum and weight decay; all other branches retain their present optimizer.

- [x] Add a deterministic test in `test_client_tinynet_golden_fedavg_wire.py`: build two identical TinyNets; train one through `client.train()` on the fixed 8-row golden batch for one epoch, and one through an explicit single PyTorch `SGD(lr=0.001)` cross-entropy step. Assert all parameters match within `1e-6`, including unchanged frozen `fc2`. Configure the client's module flags through the existing `_as_fedavg_tinynet_client` helper.
- [x] Run `cd fl-runtime && ../.venv/bin/python -m pytest -q tests/test_client_tinynet_golden_fedavg_wire.py -k optimizer` and verify the endpoint test fails because the present code constructs Adam.
- [x] Add a TinyNet-only SGD branch ahead of the current `USE_LLM`/default optimizer choices. Use `p for p in net.parameters() if p.requires_grad`; do not alter the other branches.
- [x] Re-run the targeted test and the complete `fl-runtime` suite. The full run had one unrelated Hugging Face DNS failure; the offline run with only that test excluded passed (296 passed, 29 deselected).
- [x] Run `git diff --check`, review the diff, and commit with `fix(mobile): align TinyNet desktop optimizer with native SGD`.

## Follow-on gates

After this task, separately pin the live round's learning rate and local step budget in the versioned execution contract, and replace Android's hardcoded `roundConfigFor` values with validated contract values. Then test FedAvg, FedOpt, and Robust as first-order strategies; implement FedProx's proximal term before enabling that strategy. DeComFL LightSecAgg, file-picked datasets, non-TinyNet artifacts, text, and GPU qualification remain separate approved stages. Passing Task 1 alone does not satisfy those gates.
