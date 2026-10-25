# Literature and NVIDIA FLARE refresh

25 October 2026. Narrow primary-source refresh after the diagnostic pilots. This note does not amend the frozen finalist protocol or require another implementation.

## Tested mechanisms and comparison boundary

The [qualification](effusion-byzfl-qualification-result-2026-10-25.md) reconciled native and pinned ByzFL 0.0.11 mean exactly on one seed, with and without private noise. The [screen](effusion-diagnostic-screen-result-2026-10-25.md) then tested clipping, trimming, one-step released-history centered clipping, fixed public projections and owned-data PGD; the [bucket pilot](effusion-bucket-pilot-result-2026-10-25.md) tested median/bucketing without DP. These are completed development diagnostics, not untested recommendations.

The adapter uses upstream reducers but its own seeded grouping and released-center policy. Upstream [CenteredClipping](https://byzfl.epfl.ch/aggregators/classes/centered_clipping.html) retains its previous output as the next center; the private adapter instead uses the actual released model delta. It tests neither arbitrary iterative private clipping nor the complete client-momentum algorithm of [Learning from History](https://proceedings.mlr.press/v139/karimireddy21a.html). The conditional one-step sensitivity bound cannot be transferred unchanged to another stateful protocol.

Projection improved private utility in the two-seed screen while reducing clean utility. Bucketing improved some skewed clean points and worsened several high-attacker points. These findings justify the bounded [fresh-seed finalist comparison](effusion-finalist-protocol-2026-10-25.md), not a universal ranking. None establishes external-site performance or privacy of preprocessing, diagnostics, selection or the complete research workflow.

## Recent methods: relevant, not evaluated here

[Byz-Clip21-SGD2M](https://arxiv.org/html/2603.23472v1), a March 2026 preprint, combines client gradient momentum, clipped error feedback and robust aggregation of accumulated messages. Its algorithm adds noise at clients; its privacy discussion uses Gaussian mechanisms and advanced composition for local releases. Its MNIST evidence and smoothness/sub-Gaussian assumptions do not establish performance for this frozen-head workload. Matching it requires aligning the protected unit, client state, transmitted messages and accountant, rather than attaching its name to one-step centered clipping. Code availability was not verified. It remains a candidate for separate reproduction, not an implemented baseline or an established winner here.

[PAC-DP](https://arxiv.org/abs/2603.24003), also posted in March 2026, studies record-level local DP with public-proxy-fitted clipping schedules. This is a different privacy unit and training mechanism from replacing an entire simulated client. Its matched-budget claims cannot be imported directly.

[DP-BREM](https://arxiv.org/html/2306.12608v4), published at USENIX Security 2025, likewise targets record-level cross-silo privacy. Client momentum and centered clipping motivate further work, but its theorem is not the accountant for the present client-private experiment. DP-BREM+ separately verifies bounded inputs and jointly generates noise: secure summation alone supplies neither property. This distinction is relevant to the earlier distributed-DP clip-skipping failure.

## NVIDIA FLARE: documented capability versus measured comparison

The official [release list](https://github.com/NVIDIA/NVFlare/releases), checked today, lists 2.9.0 as a stable feature release, 2.8.3 as a maintenance release, and 2.9.1rc5 as a prerelease. Documentation/examples below were inspected; no NVIDIA FLARE job was executed for this study. Links to `main` describe the inspected development snapshot, not a pinned experiment dependency.

The [2.9.0 recipe catalog](https://nvflare.readthedocs.io/en/2.9.0/user_guide/data_scientist_guide/available_recipes.html) supports FedAvg, FedProx, FedOpt and SCAFFOLD. FedProx requires the proximal client loss; SCAFFOLD requires control-variate updates and extra client messages. Ignoring those contracts does not run the named algorithm. The same catalog documents HE FedAvg with provisioned contexts and production/POC execution, excluding `SimEnv`. This does not reproduce coded LightSecAgg or demonstrate Byzantine input validation.

The actual [custom median example](https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/cifar10/pt/cifar10-sim/cifar10_custom_aggr/custom_aggregators.py) stacks client arrays and computes coordinate medians. It is useful implementation evidence, not an encrypted median or an end-to-end private robust protocol.

[Hello-DP](https://nvflare.readthedocs.io/en/main/hello-world/hello-dp/) uses Opacus per-sample DP-SGD and cumulative accounting across rounds. The [privacy documentation](https://nvflare.readthedocs.io/en/2.9.0/user_guide/admin_guide/security/differential_privacy.html) explicitly distinguishes SVT filter accounting from whole-training accounting. Percentile sparsification alone is not a formal epsilon guarantee. The research [GaussianPrivacy filter](https://github.com/NVIDIA/NVFlare/blob/main/research/quantifying-data-leakage/src/nvflare_gradinv/filters/gaussian_privacy.py) scales noise by an update percentile; that source supplies no fixed client sensitivity or epsilon/delta composition analysis. Its name does not establish DP equivalent to the capstone mechanism.

## Priority after the frozen study

Complete the declared finalist comparison unchanged. If another study is warranted, prioritize a small matched mean-versus-server-momentum comparison on the selected representation: two optimizers, clean/owned-PGD, zero-noise/private, and eight prespecified fresh seeds would be 64 cells. Freeze one momentum policy, common client recipe, threat information and sensitivity/accounting before fitting. Server operations using only private aggregates are postprocessing; new raw client-history releases are not. This proposed comparison is a research priority, not added completion work.

SCAFFOLD or a complete client-momentum protocol follows only after qualifying its messages and privacy analysis. DP-FTRL/matrix mechanisms are secondary: [multi-epoch matrix-factorization work](https://proceedings.mlr.press/v202/choquette-choo23a.html) handles repeated participation explicitly; single-participation accounting cannot be reused when every client participates in all rounds.

There is no state-of-the-art claim: the study uses one repeatedly inspected development cohort, a bounded recipe/attack set and simulated clients, with no matched reproduction of these newer protocols or NVIDIA FLARE. Source inspection verifies published definitions and implementation boundaries; it does not independently reproduce their empirical superiority claims.
