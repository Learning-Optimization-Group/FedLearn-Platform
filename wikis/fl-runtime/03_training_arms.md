# 03 — Training Arms

> **Part of:** [FedLearn Platform Docs](../README.md) → [FL Runtime Wiki](./README.md)

## Table of Contents
- [What an Arm Is](#what-an-arm-is)
- [The Three Arms](#the-three-arms)
- [How an Arm Is Declared](#how-an-arm-is-declared)
- [The Arm Helpers in `recipes.py`](#the-arm-helpers-in-recipespy)
- [End-to-End Flow](#end-to-end-flow)
- [Server Side](#server-side)
- [Client Side](#client-side)
- [`arm_tradeoff.json` — the measured cost of the choice](#arm_tradeoffjson--the-measured-cost-of-the-choice)
- [Provenance: the Arm Rides on Every Result](#provenance-the-arm-rides-on-every-result)
- [Four Defects Only a Live Federation Found](#four-defects-only-a-live-federation-found)
- [Frozen Means Frozen: the BatchNorm Trap](#frozen-means-frozen-the-batchnorm-trap)

---

## What an Arm Is

A **training arm** is a declared property of a project that answers two questions at once:

> **which parameters train, and under what objective.**

It started as only the first half. `OVA_LP` forced the second: it trains exactly the parameters `FROZEN_HEAD` trains, and differs solely in the loss. So an arm that recorded only a parameter subset would make two genuinely different experiments indistinguishable.

Before arms existed, `client.py` inferred the behaviour from the recipe key — a module global `USE_DERIVED = (mt == "FROZEN_DEMO")`. That gave every recipe exactly one hard-coded arm, so `PNEUMONIA_CNN` could not be run frozen *and* full as two arms of one comparison through the product path. It also meant a result could not say which arm produced it, which is the bug class behind commit `21699bc` — *"frozen arm silently mislabelled its backbone, risking cell overwrites"*.

---

## The Three Arms

```python
TRAINING_ARMS  = ("FULL", "FROZEN_HEAD", "OVA_LP")                  # recipes.py:55
ARM_OBJECTIVES = {"FULL": "cross_entropy",
                  "FROZEN_HEAD": "cross_entropy",
                  "OVA_LP": "one_vs_all"}                           # recipes.py:59
DEFAULT_ARM    = "FULL"                                             # recipes.py:60
```

| Arm | Trains | Objective | Notes |
|---|---|---|---|
| `FULL` | everything (`trainable_prefixes` is `None`) | softmax cross-entropy | The default. An omitted arm resolves here, so every project created before arms existed is unchanged |
| `FROZEN_HEAD` | only the head, per the recipe's `trainable_spec` prefixes | softmax cross-entropy | Head-only federation: the frozen backbone never rides the wire |
| `OVA_LP` | the same parameters as `FROZEN_HEAD` | one-vs-all (BCE-with-logits against a one-hot target) | One-vs-all linear probing on a frozen encoder, per arXiv:2511.05028 |

### Why the objective is not cosmetic

Softmax cross-entropy normalises across classes, so raising one logit lowers every other class's contribution — a client holding **none** of class *k* still moves class *k*'s weights through that coupling. Under one-vs-all each class is its own binary problem, so a class absent from a client's shard is simply a negative example like any other. That is the recorded argument for suppressing client drift at its source under extreme non-IID.

`build_criterion` (`recipes.py:144`) constructs the loss. The `one_vs_all` branch uses `BCEWithLogitsLoss(reduction="sum")` divided by the batch size — **sum over classes, mean over the batch**, which is *not* `BCEWithLogitsLoss`'s default. Two reasons the default is wrong here:

1. **Faithfulness.** "C independent binary classifiers" means each classifier gets the gradient it would get trained standalone. Averaging over classes scales every one by 1/C, making a class's update depend on how many other classes exist — a weaker form of exactly the coupling this objective removes.
2. **Comparability.** Softmax cross-entropy averages over the batch only. Under the default, the OvA arm would train at 1/C the effective learning rate, so a `FROZEN_HEAD` vs `OVA_LP` contrast on CIFAR-10 would be confounded by a 10× LR difference rather than measuring the objective.

Inference is unchanged: the head still emits per-class scores and `argmax` remains valid, so accuracy stays comparable across arms.

> **Honest scope.** `OVA_LP` implements the frozen encoder and the one-vs-all objective. It does **not** implement the paper's two-stage schedule — this repository records that a schedule exists but not what it is, and inventing one would misattribute a design to the citation. `CIFAR_RESNET18`'s `arm_notes` block says so in the catalog itself, so the caveat travels with the metadata rather than living only in a doc. Read `OVA_LP` results as *OvA heads on a frozen encoder*, not as a reproduction of the paper.

---

## How an Arm Is Declared

Every recipe declares two things in its `RECIPE_METADATA` entry:

```python
"supported_arms": ["FULL", "FROZEN_HEAD", "OVA_LP"],
"trainable_spec": {"FULL": None, "FROZEN_HEAD": ["fc."], "OVA_LP": ["fc."]},
```

`trainable_spec[arm]` is a list of **parameter-name prefixes** that stay trainable; `None` means everything. Current declarations:

| Recipe | `supported_arms` | `FROZEN_HEAD` / `OVA_LP` prefixes |
|---|---|---|
| `PNEUMONIA_CNN` | `FULL`, `FROZEN_HEAD` | `classifier.` |
| `CNN` | `FULL`, `FROZEN_HEAD` | `fc3.` |
| `CIFAR_RESNET18` | `FULL`, `FROZEN_HEAD`, `OVA_LP` | `fc.` (both) |
| `MLP` | `FULL` | — |
| `TRANSFORMER` | `FULL` | — |
| `LLM_LORA` | `FULL` | — |
| `TINYNET_GOLDEN` | `FULL` | — |
| `BLOOD_CNN` (non-catalog) | `FULL`, `FROZEN_HEAD` | `classifier.` |
| `FROZEN_DEMO` (non-catalog) | `FULL`, `FROZEN_HEAD` | `head.` |

Two invariants are pinned by `tests/test_ova_lp_arm.py`: a recipe offering `OVA_LP` must also offer `FROZEN_HEAD` (OvA-LP *is* linear probing on a frozen encoder, so it needs the same freezing capability), and `OVA_LP` must declare the **same prefixes** as `FROZEN_HEAD` on that recipe — otherwise the comparison between them stops being controlled.

---

## The Arm Helpers in `recipes.py`

| Function | Line | What it does |
|---|---|---|
| `apply_arm(model, arm, prefixes)` | `recipes.py:63` | Sets `requires_grad` across the model in place. **Writes every parameter's flag**, never just clearing some, so applying an arm is idempotent and a process that ran a frozen arm cannot leak frozen state into a later `FULL` run in the same interpreter. **Raises if the prefixes match nothing** — a typo'd prefix would otherwise freeze the whole model and train nothing, which looks like a converged-but-terrible run rather than an error |
| `freeze_untrained_modules(model, prefixes)` | `recipes.py:95` | Puts every module *outside* the trainable set into `eval()` mode. See [the BatchNorm trap](#frozen-means-frozen-the-batchnorm-trap). Prefixes are *parameter*-name prefixes (`"fc."`) while module names are bare (`"fc"`), so it appends the separator before comparing — matching the raw name would put the trainable head itself into eval mode and silently disable its dropout |
| `arm_objective(recipe_key, arm)` | `recipes.py:139` | The objective for this (recipe, arm). Validates first, so it raises on an arm the recipe cannot run |
| `build_criterion(objective)` | `recipes.py:144` | The loss. This is the only function here that imports torch |
| `validate_arm(recipe_key, arm)` | `recipes.py:186` | Resolves `None`/`""` to a runnable arm and rejects unsupported ones. Called at **project creation** so a bad arm fails there rather than at FL-server spawn |
| `trainable_prefixes(recipe_key, arm)` | `recipes.py:217` | The prefixes; `None` = everything |
| `arm_stamp(recipe_key, arm)` | `recipes.py:224` | JSON-serialisable provenance for a result's `meta` block |

### The resolution rule in `validate_arm`

An omitted arm resolves to `FULL` **when the recipe supports it**. When it does not, and the recipe offers exactly one arm, it resolves to that one; when it does not and several arms are offered, it raises and demands the arm be stated explicitly.

That middle case is not hypothetical. `CIFAR_RESNET18` could not run `FULL` at all until 2026-08-13 — every BatchNorm module carries an int64 `num_batches_tracked` and the safetensors wire is float32-only, so a `FULL` run died on the first `GetGlobalModel`. Raising on every creation that omitted the arm would have turned a capability limit into an outage.

---

## End-to-End Flow

```
Frontend  CreateProjectModal
   │  arm selector rendered only when supportedArms.length > 1;
   │  trainingArm is included in the payload only when a choice was offered
   ▼
POST /api/projects   CreateProjectRequest.trainingArm
   │  @Pattern "must be one of: FULL, FROZEN_HEAD, OVA_LP"
   ▼
Java  TrainingArm enum  →  Project.trainingArm  (defaults to FULL)
   │  DB: projects.training_arm VARCHAR(32) NOT NULL DEFAULT 'FULL'
   │      CHECK chk_projects_training_arm IN ('FULL','FROZEN_HEAD','OVA_LP')
   │      (V22__project_training_arm.sql, widened by V23__training_arm_ova_lp.sql)
   ▼
FlServerManager.buildServerCommand
   │  emits --training-arm ONLY when the arm is not FULL, so a pre-arm spawn's
   │  argv is byte-identical to before
   ▼
fl-runtime/fl_server.py  --training-arm      fl-runtime/client.py  --training-arm
                                              ▲
                       Desktop: TRAINING_ARM env → entrypoint.sh (Docker path)
                                or --training-arm directly (native path)
```

The **Python side is the authority on whether a recipe supports an arm** — `recipes.validate_arm()` decides that, because the catalog is where `supported_arms` lives. The Java enum and the V23 `CHECK` bound the *vocabulary* only.

---

## Server Side

`fl_server.py` resolves and applies the arm while loading initial parameters (`fl_server.py:667-687`):

```python
args.training_arm = recipes.validate_arm(args.model_type, args.training_arm)
full_initial_parameters = OrderedDict(initial_parameters)   # keep the complete model
initial_parameters = federable_state(initial_parameters)    # drop non-float32 first
_prefixes = recipes.trainable_prefixes(args.model_type, args.training_arm)
if _prefixes is not None:
    _kept = OrderedDict((k, v) for k, v in initial_parameters.items()
                        if k.startswith(tuple(_prefixes)))
    if not _kept:
        logging.error(...)   # names the arm, the prefixes and the keys it actually saw
        exit(1)              # an empty federated set is a hard failure, not a degenerate run
    initial_parameters = _kept
```

Three properties of that block matter:

- **The `.npz` deliberately keeps the FULL model.** The arm is applied here, at load, not at save — the frozen backbone has to stay recoverable. See [04](04_the_federated_set.md#saving-merge_non_federated).
- **The wire filter runs *before* the arm filter**, using the same helper the client uses, so both sides agree on the federated key set by construction.
- **An arm whose prefixes match no key in the `.npz` exits 1** with the keys it actually saw, rather than federating an empty set.

The server must federate exactly what the clients send. A client on a subset arm returns `trainable_state(net)`; a server holding the full state dict would have `d_server > d_client`, and for DeComFL the shared-seed perturbation `z` would silently misalign.

`fl_server.py` also derives **evaluation strictness** from the arm rather than from a recipe name — see [04](04_the_federated_set.md#evaluation-strictness).

---

## Client Side

`client.py` resolves the arm in `main()` and stores three globals:

```python
MODEL_TYPE   = mt
TRAINING_ARM = _r.validate_arm(mt, requested)
USE_DERIVED  = _r.trainable_prefixes(mt, TRAINING_ARM) is not None
```

`USE_DERIVED` is **derived from whether the arm federates a subset**, not from an arm-name comparison. The name test silently excluded `OVA_LP`, which is also a subset arm — and comparing arm names is the pattern behind three separate defects in this file's history.

Three places consume it:

1. **`apply_declared_arm(net)`** (`client.py:173`), called **after the whole build chain** (`client.py:695`) rather than inside one branch of it. See below.
2. **`get_parameters()`** (`client.py:714`) — under a subset arm it returns `trainable_state(self.net)`, so the frozen backbone never rides the wire. This mirrors the `LLM_LORA` adapter-only upload.
3. **`fit()`** — under a subset arm the aggregated head is loaded with `strict=False`, because the wire carried only the head and the local backbone must survive.

The objective is applied in `train()` (`client.py:405-414`): the criterion comes from `recipes.arm_objective(MODEL_TYPE, TRAINING_ARM)`, defaulting to cross-entropy when no recipe is set, so every legacy caller is byte-identical.

> **A known asymmetry, stated honestly:** `fl_server.py`'s `server_side_evaluate` builds `torch.nn.CrossEntropyLoss()` unconditionally (`fl_server.py:830`). So an `OVA_LP` run trains under one-vs-all but its reported server-side *eval loss* is cross-entropy. Accuracy — which is what the arm trade-off is stated in — is unaffected, because argmax over per-class scores is valid under both objectives. Treat an `OVA_LP` run's loss curve as not directly comparable to its own training loss.

---

## `arm_tradeoff.json` — the measured cost of the choice

`fl-runtime/arm_tradeoff.json` carries the **measured** frozen-vs-full trade-off, generated by `scripts/build_arm_tradeoff.py` and surfaced through `describe()` so the picker can show what the choice costs at the point it is made.

Its structure is `{"$schema_version": 2, "generated_by": …, "note": …, "by_recipe": {…}}`, and `arm_tradeoff(recipe_key)` (`recipes.py:376`) does a **keyed lookup with no fallback**:

> Returning a default here is exactly the bug this replaced: one chest X-ray result was attached to every dual-arm recipe, so a CIFAR-10 recipe advertised a pneumonia figure.

It returns `None` rather than raising when the file is missing or unparseable — a missing trade-off must cost the user an explanation, not the ability to create a project.

Three recipes currently carry a measurement, and the three headlines say very different things — which is the point of keying it per recipe:

| Recipe | Measured verdict |
|---|---|
| `PNEUMONIA_CNN` | Freezing costs **21.8 accuracy points** (92.44 % → 70.62 %) and saves only 1.004× the communication — its classifier is 99.6 % of the model, so there is almost nothing to freeze. The frozen arm is identical every round: the majority-class rate. It never learns |
| `CNN` | The frozen arm reaches **10.0 %** — chance for ten classes — against 38.0 % for `FULL`. The backbone is randomly initialised, so freezing it trains a head on random features |
| `CIFAR_RESNET18` | The frozen arm reaches **80.37 %** for 0.02 MB per download while `FULL` reaches 77.94 % for 42.7 MB — **2,135× less data**, and slightly *better* accuracy at this budget. It starts from ImageNet weights, so freezing keeps a backbone worth keeping |

Every entry records its `source` result file, that file's `source_sha256`, the protocol (rounds, clients, α, seed, resolution) and its own caveats. `tests/test_arm_tradeoff.py` guards two distinct risks: **drift** (the committed numbers are regenerated from the record and compared where the record is present) and **unsupported claims** (a number shown without its caveat is a claim the record does not make — the comm ratio in particular is round-budget dependent).

> The `research/` tree those source files live in is **gitignored**, so the committed JSON is the only copy of these numbers inside the repository.

---

## Provenance: the Arm Rides on Every Result

`build_eval_card` (`fl_server.py:147`) writes the arm onto the eval card attached to the registered model artifact:

```json
{ "recipe_key": "...", "training_arm": "FROZEN_HEAD", "trainable_prefixes": ["fc."], "seed": 1234567, ... }
```

Three deliberate choices:

- **`FULL` is recorded explicitly, not by absence.** Otherwise a reader cannot distinguish a full fine-tune from a card written before arms existed.
- **The prefixes ride along with the name.** Two runs can share an arm name while freezing different modules, and a name alone is not a checkable provenance claim.
- **`arm_stamp()` additionally carries the objective**, because `OVA_LP` and `FROZEN_HEAD` train the same parameters — without the objective their stamps would be identical and two different experiments would be indistinguishable.

A card travels independently of the project, so a reader must be able to answer *"which arm produced this?"* from the card alone. The failure this prevents is downstream, when results are **keyed**: if the identity of a result does not include the arm, two arms of the same recipe produce the same key and the second silently replaces the first. A warning does not fix that; only putting the arm in the key does.

---

## Four Defects Only a Live Federation Found

The arm feature was implemented, unit-tested, and *then* run live — and the live run surfaced four defects that no unit test had caught. Each now has a regression test named after the property rather than the instance.

| # | Symptom | Cause | Guard |
|---|---|---|---|
| 1 | The FL server never started: `UnboundLocalError` on `recipes` | `main()` contains `import recipes` inside later dataset branches. In Python, any binding of a name inside a function makes it local for the *entire* body — so the arm filter's earlier use referred to an unbound local | `test_fl_server_arm_scope.py` |
| 2 | A completed round failed evaluation: `Missing key(s) in state_dict: conv1.weight, …` | `server_side_evaluate` hardcoded `_strict = model_type != 'TINYNET_GOLDEN'` — correct while that was the only subset-federating recipe, wrong the moment any recipe could run `FROZEN_HEAD` | `test_server_eval_strictness.py` |
| 3 | A frozen CNN run died on `Expected 3D or 4D input to conv2d, but got input of size: [32, 256]` | `load_data` branched on `USE_DERIVED`, which used to mean "this is FROZEN_DEMO". Once it meant "this arm federates a subset", it handed FROZEN_DEMO's synthetic 256-dim vector shard to whatever model the selected recipe had built. **The dataset is a property of the recipe; the arm only decides which parameters are trainable** | `test_arm_does_not_choose_the_dataset.py` |
| 4 | A `FROZEN_HEAD` run reported itself frozen while its backbone moved by 0.65 | The arm was applied *inside* the `USE_DERIVED` branch of the build chain, and `USE_PNEUMONIA` is tested first — so a frozen pneumonia run built its model, skipped the arm entirely, and fine-tuned the whole network. **Correctness must not depend on the order of a build chain** | `test_arm_applies_to_every_recipe_branch.py` |

A fifth, found in the same campaign: a `FROZEN_HEAD` run's saved `.npz` contained **two keys**, because the server writes the final *global* model to `--model-path` and under a subset arm the global model *is* the head. That destroyed the only full copy of the model. Fixed by `merge_non_federated` — see [04](04_the_federated_set.md#saving-merge_non_federated), guarded by `test_frozen_save_preserves_backbone.py`.

A sixth, subtler one concerns FedProx. `_apply_proximal_gradient` zips `net.parameters()` against `global_params` **positionally**. Had `global_params` been the *received* global model, that would be the head alone under `FROZEN_HEAD` — two tensors against sixty-two — and `zip` would silently pair the first two backbone parameters with the head's values and stop. `test_fedprox_respects_the_frozen_arm.py` pins that the proximal term regularises the federated subset and leaves the frozen backbone alone.

---

## Frozen Means Frozen: the BatchNorm Trap

`apply_arm` sets `requires_grad=False`, which stops **gradients**. It does not stop BatchNorm.

`running_mean` and `running_var` are **buffers, not parameters**, so `requires_grad` does not touch them: a module left in `train()` mode re-estimates them from local data on every forward pass. That is a second, silent channel through which a "frozen" backbone changes.

Measured on CIFAR-10 (linear probe on frozen features), 2026-08-13:

| Condition | Accuracy |
|---|---|
| Pretrained backbone, **BN held fixed** | **80.37 %** federated / 81.05 % offline probe |
| Pretrained backbone, **BN adapting** | 72.85 % federated |
| Random backbone, BN fixed (offline probe, same arch/resolution/data) | 26.25 % |

> The two probe figures are the **committed** harness's (`research/benchmarks/frozen_feature_linear_probe.py`, seed 1234 / head seed 0). They supersede an earlier inline-script pair — 80.35 % / 25.10 % — which `recipes.py`'s `freeze_untrained_modules` docstring still quotes; the record keeps both (`research/results/pretrained-backbone/pretrained_frozen_arm_2026-08-13.json`) and notes that probe and federation agree to within about a point, not the 0.02 once claimed.

So BN adaptation **degrades** good features by roughly 7.5 points — it re-estimates, from one client's shard, statistics that ImageNet training had already fitted well.

> An earlier reading of this said BN adaptation *lifted a random backbone* to 72 %. That came from a federated "random control" that was invalid: under a subset arm the backbone is never transmitted, so the client always builds it locally from the recipe and a random `.npz` cannot change it. The valid random comparison is the offline probe at 26.25 %.

Three things are wrong with letting it adapt, independent of the numbers: the arm's premise is that the backbone is delivered once and stays fixed; BN statistics are data-dependent and never federated, so clients silently diverge from each other and from the server's copy; and it hides the value of pretraining, which is the only reason to freeze a backbone at all.

The fix is `freeze_untrained_modules`, and its call site is timing-sensitive: `nn.Module.train()` re-enables every child, so it must be re-applied **after** each call to `train()`, not before. `client.py` does exactly that (`client.py:415-424`). For the `FULL` arm it is a deliberate no-op — BatchNorm *should* adapt when the whole model is being trained.

`test_frozen_means_frozen.py` pins this.
