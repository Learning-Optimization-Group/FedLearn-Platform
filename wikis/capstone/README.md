# Capstone plan — current as of 18 October 2026

This is the current schedule and completion checklist. Earlier milestone decks,
reports and design notes remain historical evidence. Phase 2 has not been
pre-registered or run; the Milestone 2 grids are exploratory.

## Objective and boundaries

Study how client-level differential privacy, robust aggregation, adversarial
clients and trainable head size interact in a reproducible federated simulation.
The next dataset is NIH ChestX-ray14 with **0 = No Finding** and **1 = any of the
listed findings**. These are report-derived labels, not verified clinical diagnoses.
Pneumonia-only results from the mobile project answer a different task.

The simulation engine is implemented at `8a9b89d`. Its 1,000-client execution check
does not establish performance or scientific results on NIH. The mobile branch is
published separately at `8467f37`; its Release/TLS work is parked for the capstone.
Do not merge all mobile changes into the research branch just to obtain a loader.

## Verified starting point

- The presented Milestone 2 HTML deck was last edited on 7 October. Its older
  PowerPoint and the first generated HTML deck are not the current presentation.
- The three retained grids contain 420 + 480 + 45 = 945 run JSON files, excluding
  their index files. The research note explicitly calls them exploratory.
- The capstone branch has 17 commits not present on origin before this handoff.
  Its code, ignored results/notes, presentation assets, experiment configurations
  and original Kermany feature cache are included in a private safety backup.
  Git publication alone does not include ignored evidence.
- NIH metadata has 112,120 unique image IDs: 60,361 No Finding and 51,759 with
  findings. Official train/validation has 86,524 images from 28,008 patients;
  official test has 25,596 images from 2,797 patients. The lists cover metadata,
  with zero image-ID or patient overlap. Image-content/corruption checks are still
  required; metadata checks alone do not establish content integrity.
- The existing workload repartitions feature rows and lacks patient ownership.
  The current feature-preparation CLI expects ImageFolder data. Setting its
  dataset label to `nih` does not implement an NIH loader.

## Schedule and completion gates

| When | Priority | Required output |
| --- | --- | --- |
| 18–19 October | NIH data and pilot | Explicit labels, validated patient-disjoint manifests, frozen ResNet-18 feature cache with provenance, patient-preserving simulated clients, clean validation pilot |
| Before confirmatory runs | Phase-2 protocol | Versioned hypotheses, primary endpoint, exact configurations/seeds, tuning/test separation, run budget, inference and failure-handling rules; approved and frozen before examining confirmatory results |
| 20 October–2 November | Experiments | Complete the core matched NIH study first, then separately budgeted scale/head-size studies; verified records, failures and analysis |
| 3–16 November | Simulation Lab | Design review first; minimum GUI loads a resolved config, starts/stops a run, streams results and exports the record |
| 17–30 November | Final deliverables | Report, current presentation and reproducible demonstration; keep writing alongside experiments |
| 1–5 December | Buffer | Reproduction checks, corrections and submission; 5 December is the stated deadline |

October 20 is a target start, not permission to call a retrospective analysis
pre-registered. If the data/pilot/protocol gates are incomplete, move confirmatory
runs rather than mislabeling their status. Advisor title approval remains open;
draft the message for review, without treating a working title as approved.

## Next work: NIH adapter

The [NIH workload design](../../docs/superpowers/specs/2026-10-18-capstone-nih-workload-design.md)
defines the proposed implementation boundary and acceptance checks. Preserve the
old Kermany behavior and exact reproduction anchors. Do not alter the mobile
pneumonia recipe, reuse its class mapping, overwrite an old feature cache or inspect
official-test performance to select settings.

The first useful run is a small, no-DP/no-attack validation pilot with real NIH
features. It must establish that labels, learning, patient ownership and evaluation
work before scaling to 1,000 clients or adding privacy/attacks. The next gate is
a 1,000-client, two-round execution/cost check, explicitly distinct from a utility
claim. Measure elapsed time and memory before expanding the matrix.

## Protocol decisions to freeze

- Primary cohort, rounds, local optimizer/steps, learning rate, clipping norm,
  attacker knowledge/fraction/strength, aggregation rules and DP settings.
- One primary endpoint and contrast, paired seeds, interval method and treatment
  of multiple secondary comparisons. Seed intervals describe simulation variation,
  not uncertainty across clinical populations.
- Training/validation/test identities, patient-to-client allocation and whether
  balancing is applied. Keep primary evaluation at its declared prevalence;
  report majority baseline, balanced accuracy, AUROC and confusion counts.
- Pilot/tuning budget restricted to training/validation. Freeze settings before
  confirmatory test evaluation; never choose a favorable intermediate round.
- Exact cell list and repetitions, resource/time cap, stop/refusal criteria and
  reporting of incomplete/failed cells. Avoid an unrestricted Cartesian product.
- DP unit is a simulated client under replace-one adjacency. Patient-disjoint
  allocation does not by itself turn that into a patient-level DP guarantee.
  Keep the existing conservative no-subsampling-credit accounting explicit.

The recommended head-size comparison holds frozen ResNet-18 features fixed and
uses a linear head plus MLP hidden widths 8, 128 and 512: 1,026, 4,122, 65,922 and
263,682 trainable parameters. This is awaiting the author's decision. A ResNet-50
linear head has 4,098 parameters but also changes feature dimension/backbone;
retain it only as a separately named representation comparison.

## Scope control

LoRA/full image-network training is already outside this capstone phase. Do not
reintroduce it. Keep a matched core comparison of mean/trimmed mean, central
DP/no DP and clean/declared attack conditions. Treat scale, head size and the
broader bad-client suite as separately bounded studies. If the schedule slips,
cut the local-DP extension first, then GUI extras. Preserve the core study,
honest reporting, evidence backup and final reproducibility.

A useful ImageNet-feature baseline is an empirical question. Do not assume that
a domain-trained backbone would necessarily improve it, or transfer the mobile
pneumonia model's AUROC to this different label/task.
