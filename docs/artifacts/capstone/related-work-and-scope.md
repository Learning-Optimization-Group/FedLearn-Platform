# Related work and scope of comparison

Checked 20 October 2026 during capture. Selected primary-source checks for the final
report, not a systematic literature review or a verification of every cited proof.
The historical reports remain unchanged. This note changes no frozen study choice.

| Source and inspected material | Finding relevant here | Boundary for this capstone |
| --- | --- | --- |
| Guerraoui et al., [Differential Privacy and Byzantine Resilience in SGD: Do They Add Up?](https://arxiv.org/pdf/2102.08166), PODC 2021; §2.3 and start of §3, especially Eq.8 | Workers perturb gradients before sending them to a curious server. The analysis exposes dimension-dependent pressure on the variance-to-norm resilience condition. | Motivates joint measurement; it is not an impossibility theorem for our trusted-server aggregate-noise mechanism. |
| Allouah et al., [On the Privacy-Robustness-Utility Trilemma in Distributed Learning](https://proceedings.mlr.press/v202/allouah23a/allouah23a.pdf), ICML 2023; §1.1 and §2.2, Definitions2.2–2.3 | Item-level adjacency protects worker-server communication transcripts against the server and other workers; SAFE-DSHB supplies algorithmic upper bounds under declared assumptions. | Different privacy unit, observer and algorithm. Its bounds are not our NIH AUROC predictions or guarantees for local-SGD trimmed deltas. |
| Gu et al., [DP-BREM](https://www.usenix.org/system/files/usenixsecurity25-gu-xiaolan.pdf), USENIX Security 2025; abstract, §2.2 and §4.1–4.2 | Targets record-level privacy. DP-BREM uses client momentum and noisy centered aggregation with a trusted server; DP-BREM+ changes the trust architecture. | Same broad joint-defense problem, different protected unit and mechanism. We do not implement it or establish superiority to it. |
| Allouah et al., [Towards Trustworthy Federated Learning with Untrusted Participants](https://proceedings.mlr.press/v267/allouah25a.html), ICML 2025; proceedings abstract only | CafCor combines robust aggregation with correlated noise and pairwise secret randomness under an untrusted-server setting. | The abstract supports this positioning only; its proof is not checked here. Our cancellation example concerns one specific shared-noise construction, not all correlated-noise designs. |

The capstone's contribution is a reproducible comparison of existing FedLearn
components on a qualified NIH workload, with measured utility/cost, execution
qualification and an inspectable dashboard workflow. It does not propose a new
joint-defense algorithm or test the cited methods head to head. Report any eventual
ranking as conditional on this study's recipes, attacks, representation and trust
assumptions; scientific relevance does not require claiming state-of-the-art utility.

The [matched-noise boundary](matched-noise-release-boundary.md) concerns reuse of one
draw at different amplitudes across matched first-round releases. General statements
that correlated randomness is inherently unsafe would exceed that counterexample.
The [working final report](capstone-final-report-v2-working.md) should retain this
scope when it is converted to its final reference/layout format.
