# Matched noise is an experimental coupling, not a joint release guarantee

20 October 2026. Clarification of the frozen protocol's explicit per-run/research-output
scope. This changes no cell, hypothesis, seed, endpoint or calibration. The fixed
study remains a public-data mechanism/utility experiment, not a protected release.

## Source facts

At source 3baea7d, `simulation/engine.py:131,155–157` calibrates z per run and passes
`seeds.noise` into ComposedStrategy. `server/composed_strategy.py:121` resets a
persistent generator from that seed; lines171–175 draw each key's normal noise at
the chosen standard deviation. Matched private cells deliberately reuse the same
noise seed while epsilon/rule can change its amplitude. This is disclosed in the
prospective protocol's matching scheme. Normal draws advance within each run;
the observation here concerns resets across matched runs.

For the same seed/head/cohort/clipping/condition/rule, the first round's local
updates and clipped pre-noise query are common before privacy noise is added.
Later trajectories differ. Final-only checkpoints do not preserve a shared query
for every later round, so do not claim the attack below reconstructs their final
aggregate or demonstrates a patient-data attack on the retained NIH models.

## Elementary ideal-mechanism counterexample

Even grant a truly random, secret shared standard Gaussian Z, rather than the
benchmark's publicly specified seeds. Two first-round releases of the same query
f(D), at distinct public noise amplitudes a and b, are

Y_a = f(D) + a Z;    Y_b = f(D) + b Z;    a != b.

Their joint output determines

(b Y_a - a Y_b)/(b-a) = f(D).

For example, a clipped uniform mean can differ between two one-upload neighbors
whose changing scalar uploads are 0 and C: f(D)=0 versus f(D')=C/n, with all other
uploads 0. The statistic above equals the corresponding query with probability 1.
In the ideal real-valued mechanism, the event that it equals 0 therefore has
probability 1 versus 0. The joint release cannot satisfy any finite-epsilon DP bound
with delta<1, although each marginal Gaussian mechanism can have its own calibrated
bound. This is an upload-domain algebraic counterexample, not an empirical NIH
patient-neighbor reconstruction or a new general theorem.

Floating-point implementations introduce rounding. A post-capture, source-bound
scalar check through the production ComposedStrategy used 80 uploads, C0.05 and
noise calibrated to 20-round targets epsilon 1/4 (only the first round was released).
For eight fixed seeds and two one-upload neighbors, the maximum shared-noise query
reconstruction error was 1.06681513038e-09,
versus the neighbor query gap 0.000625; all 16 preset neighbor classifications matched.
An independent-noise control retained residual error (maximum
0.029968767). The [retained fixture record](matched-noise-production-illustration.json)
binds the exact source and probe hashes. This is a numerical illustration, not a
statistical DP test, formal finite-precision theorem, patient attack or recovery
from final-only NIH models. The ideal algebra is the reason marginal guarantees
alone are insufficient for this specific joint construction.

## Practical consequence

Simply summing marginal epsilons is not justified for this shared-noise construction.
Adaptive composition requires the later mechanism's guarantee conditional on prior
outputs; its premises do not follow from a standalone marginal guarantee. See
[Mironov, Proposition1](https://arxiv.org/html/1702.07476) for the composition premise.
That source is not a proof of our project's specific coupled mechanism.

Use common random numbers to reduce variance in declared public-data comparisons.
For protected deployment releases, use separately secret randomness and verify the
conditional/joint mechanism and release accounting; independent fresh draws are a
standard sufficient design direction, not a substitute for the sensitivity/export
checks. Hiding the benchmark seeds alone would not solve this joint-release example.

Keep the fixed study's epsilon labels as per-run mechanism targets and explicitly
retain its existing no-whole-matrix/no-DP-safe-research-log limitations. A dashboard
research-record export must not present these matched benchmark models/transcripts
as a composed protected deployment. No outcome-driven reruns or retroactive protocol
amendment follows from this clarification.
