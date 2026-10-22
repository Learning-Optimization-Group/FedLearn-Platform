# Robust and private FL: literature check against what we tested

22 October 2026. Four parallel research passes covered:
1. Robust + private aggregation.
2. Improving DP utility.
3. Robust secure aggregation.
4. Medical-imaging FL practice and evaluation.

I verified the items marked ✔ myself (paper exists and the stated claim checks out). Items marked ✔ (core) confirm the paper and its main mechanism only; the finer detail is the agent's reading of the full text. Everything else is as reported by the agents with their URLs. Agent notes are kept privately under `.temp/.../literature/`.

## Verdict: did we pick the best things?

Partly. The methods we ran are the standard baselines, and two of our findings match the
literature. But we skipped the methods the literature recommends **for exactly the problems we
hit**, and our evaluation used weak attacks and no empirical privacy check.

| What we found | What the literature says |
| --- | --- |
| Trimmed mean is worse than the mean under central DP (A9) | Known. Median- and trimming-type rules increase DP sensitivity, and coordinate-wise rules degrade with dimension under DP noise. Guerraoui et al., PODC'21 ([2102.08166](https://arxiv.org/abs/2102.08166)); Allouah et al., ICML'23 ([2302.04787](https://arxiv.org/abs/2302.04787)); DP-BREM, USENIX Sec'25 ([2306.12608](https://arxiv.org/abs/2306.12608)) ✔ (core). |
| Our clipped DP mean already resists attacks fairly well | Known. Norm bounding plus weak DP is a recognised defence (Sun et al., [1911.07963](https://arxiv.org/abs/1911.07963)). |
| Client-level DP is costly with 80 clients | Expected. Client-level DP-FedAvg only became nearly free with thousands of clients per round (McMahan et al., ICLR'18, [1710.06963](https://arxiv.org/abs/1710.06963)). The published chest X-ray DP results are **record-level** (Tayebi Arasteh 2024; Ziegler 2022), so our numbers are not comparable to theirs. |
| Local DP is near chance at ε ≤ 8 | Consistent with the √n penalty. The literature's alternative is distributed DP: Skellam ✔ (NeurIPS'21, [2110.04995](https://arxiv.org/abs/2110.04995)) or the discrete Gaussian ([2102.06387](https://arxiv.org/abs/2102.06387)) on top of secure aggregation. These get about central-DP utility without a trusted server. |

## Two corrections to our own work

1. **Our ε was conservative.** ✔ Verified here by exact computation (Gaussian DP; Dong, Roth, Su, [1905.02383](https://arxiv.org/abs/1905.02383)).
   - With every client in every round, the 20 releases compose exactly into one Gaussian mechanism.
   - The runs we labelled ε 8/4/1 actually satisfy **ε 6.76/3.25/0.74** (δ 10⁻⁵).
   - Calibrating exactly would need 13–24% less noise for the same claim (z 2.68/4.84/16.68 instead of 3.09/5.80/21.92).
   - The claims stand; they were simply loose.
2. **Production DP's guarantee is weaker than it reads, not miscalculated.** Resolved in the [adjacency analysis](production-dp-adjacency-analysis-2026-10-22.md).
   - `dp_aggregate` (FedLoRA, `DPFedAvg`) adds noise z·S/n. That is correct for add/remove adjacency with a fixed public denominator, and the live path always runs on the fixed cohort.
   - It is not correct for replace-one, the relation the capstone simulator uses. Read as replace-one, claimed ε 8/4/1 would be about 18/8.6/2.0 under RDP.
   - Only direct framework calls with a varying cohort understate even the add/remove ε; the live server refuses those.
   - My guidance line "z√d/N ≈ 2.3" used the add/remove scale. For the capstone runs the lower bound is about 4.6.
3. **Our secure-aggregation implementation is not communication-light.** ✔ Verified in `security/lightsecagg.py`: each peer receives a full-length per-coordinate Shamir share. LightSecAgg proper sends coded shares of size d/(U−T) (So et al., [2109.14236](https://arxiv.org/abs/2109.14236)). That explains A5's 158× upload. With proper coding the estimate is about 6–10× plain (calculated, not measured).

## Ranked shortlist (value for the platform and the capstone, per unit of effort)

| # | What | Why | Effort |
| --- | --- | --- | --- |
| 1 | **Production DP adjacency.** Done: add/remove is the default, `--dp-adjacency replace-one` doubles the noise, the eval card records the relation, and the strategies enforce a fixed denominator. | Correctness of shipped privacy claims (FedLoRA and DPFedAvg). | S–M |
| 2 | **Exact GDP accountant for q = 1.** Done: `fedlearn.privacy.gdp_accountant`, opt-in through `--dp-accountant gdp` and `plan_central_dp(accountant="gdp")`. RDP stays the default. | 13–24% less noise for the same ε, free, with no change to the unit or trust model. | S |
| 3 | **Sensitivity-matched robust rule under central DP.** Clip each update's deviation from the previous released aggregate, with sensitivity 2τ/n; sweep τ. Also a small-k trimming sweep (k = 1–16). | Directly targets our negative A9 result. | S |
| 4 | **Stronger attacks.** Fang et al.'s attack tailored to trimmed mean and median ✔ ([1911.11815](https://arxiv.org/abs/1911.11815)), IPM ([1903.03936](https://arxiv.org/abs/1903.03936)), min-max/min-sum (NDSS'21), and an attacker-fraction sweep of 1/5/10/20%. | Our trimmed-mean and median robustness claims are untested against attacks built for them. A 20% attacker fraction is a stress test by production standards (Shejwalkar et al., S&P'22, [2108.10241](https://arxiv.org/abs/2108.10241)). | S |
| 5 | **Empirical client-level DP audit.** Canary clients, one run (Andrew et al., ICLR'24 ✔ (core), [2302.03098](https://arxiv.org/abs/2302.03098)), plus LiRA membership inference ([2112.03570](https://arxiv.org/abs/2112.03570)). | Tests whether the DP claims mean anything in practice. Cheap with a linear head. | S–M |
| 6 | **Non-IID clients.** Dirichlet label skew, quantity skew, and view/age site proxies, with bucketing ([2006.09365](https://arxiv.org/abs/2006.09365)). | Our FedAvg = centralised result and the robust-rule results are IID upper bounds. Median degrades under non-IID data even with no attacker. | S |
| 7 | **Fewer rounds or a one-shot head** under DP (z falls with √T; DP from features: Mehta et al., [2211.13403](https://arxiv.org/abs/2211.13403)), and **dimension reduction with a non-private subspace**. | The biggest possible DP utility gains in our regime. | S–M |
| 8 | **Noise-aware norm filtering under local DP** (Xiang et al., SIGMOD'23, [2304.09762](https://arxiv.org/abs/2304.09762)). | Robustness at zero privacy cost under local DP; complements A12. | S |
| 9 | **Distributed DP: Skellam on secure aggregation, plus proper coded LightSecAgg shares.** | The realistic route to "no trusted server" with about central-DP utility, and a fix for the 158× upload. | M–L |
| 10 | **Stronger frozen encoder** (e.g. RAD-DINO). | Better features shrink the DP gap (Tramèr & Boneh, [2011.11660](https://arxiv.org/abs/2011.11660)). **Caution:** the public RAD-DINO checkpoint was reportedly trained on all of NIH CXR14, including our holdout, so using it would leak. | M |

Deliberately not prioritised:
- Client subsampling for amplification (does not pay off at N = 80).
- DP-FTRL / matrix factorisation (unclear gain at T = 20 with full participation).
- Patient-level ISRL-DP (changes the privacy unit; worth mentioning, not swapping in).
- Two-server MPC robust rules and zero-knowledge norm proofs (large cryptographic builds).

## Where our numbers sit

- **Non-private effusion AUROC (0.836 holdout)** is consistent with published ChestX-ray14 effusion results, for example 81.5–82.6 for fine-tuned ResNet50 in multi-site FL (Tayebi Arasteh et al., Sci. Rep. 2023). Splits and labels differ, so this is "in range", not state of the art.
- **DP results** should be presented as client-level DP at 80 clients, which is much stronger and much costlier than the record-level chest X-ray results in print. The **specificity-at-fixed-sensitivity** framing was not found in the checked papers.
