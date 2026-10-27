# Advisor review packet — draft, not sent

26 October 2026. This packet supersedes the old phase-two title draft for current scope discussion. It is not evidence of advisor approval or message delivery.

## Current proposed title

Measuring Privacy and Robustness Trade-offs in Federated Learning

## Contribution to assess

An auditable experimental implementation inside FedLearn, with qualified aggregation/privacy integration, conditional utility evidence and verified execution across heterogeneous hosts. It does not propose a new privacy theorem or a universally superior aggregation algorithm.

The frozen Effusion comparison uses pretrained features and a linear head. Projection improves private-clean AUROC by +0.019176 on a repeatedly used selection cohort, with a clean no-noise utility cost. The three-host CPU extension preserves numerical training semantics but incurs a whole-job timing penalty. Fresh CPU environment and tracked-source reproduction are verified.

Independent-cohort evaluation was stopped at the access gate because no approved unused cohort is available. This limitation remains in the report. GPU/deep-backbone, matched NVIDIA FLARE and protected-production extensions are not substituted for that blocked study.

## Decisions needed from the advisor

1. Is this bounded engineering/empirical contribution appropriate for the capstone, with the independent-cohort limitation stated explicitly?
2. Confirm the title and which contribution should receive the most emphasis.
3. Confirm required report format, length, citation style, assessment rubric and submission route. The current manuscript is a 14-page working reading copy, not a declared RIT-format submission.
4. Confirm presentation duration, demo expectations and the actual due date. The working schedule records 5 December 2026; this is not independently verified institutional guidance.

## Suggested message — not sent

Subject: Capstone scope, report format and defense confirmation

Hi Professor,

I have consolidated the capstone into a reproducible study of privacy, robust aggregation and representation choices within FedLearn. The current working report is 14 pages. It includes the frozen twelve-seed Effusion comparison, a verified three-host CPU execution study and their limitations.

The strongest private candidate improves selection AUROC while sacrificing clean no-noise utility. Distributed execution matches simulator parameters, but the short three-host jobs are slower overall. I have preserved these negative findings. No independent cohort was evaluated because approved access was unavailable; I am keeping that limitation explicit rather than expanding the project into several new studies.

Could you confirm whether this scope and the proposed title are appropriate, and specify the report format, presentation/demo duration and submission requirements? I can provide the current report, evidence map and reproduction instructions for review.

Best regards,
Anurag

## Review materials

- [Report PDF](capstone-final-report-v5-working.pdf) and [editable manuscript](capstone-final-report-v5-working.md).
- [Current claim/evidence map](consolidation-evidence-map-2026-10-26.md).
- [Research-freeze decision](research-freeze-2026-10-26.md).
- [Clean reproduction](clean-reproduction.md).

Do not send this draft, publish data, merge or push on the basis of this packet alone.
