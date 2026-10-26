# Pre-benchmark implementation review

26 October 2026. Verdict after correction: no remaining material finding from the source review; full gates and final frozen-snapshot qualification remain separate evidence requirements. Medical fitting was not used to debug or tune these corrections.

| Finding | Failure mechanism | Correction and regression |
| --- | --- | --- |
| Admission acknowledgement | Stale/duplicate updates were not counted, but RPC returned `received=True`; host journal interpreted that as admission. | Fixed-cohort RPCs now return actual admission for unary and streamed updates. Generic receipt semantics remain unchanged. Real generated gRPC test covers stale/ahead/duplicate and missing-client failure; full-cohort success remains covered. |
| Public input seal | Training data and private hash fields could be changed together while leaving the public protocol digest unchanged. | Public manifest now commits to every host manifest and server plan; these transitively bind payloads and selection. A generated reseal test failed before the correction and passes afterward. |
| Disconnected remote cleanup | Local SSH exit was treated as proof that remote Python had exited. | Always attempt exact-plan/executable ownership verification; use Linux process descriptors for bounded TERM, then KILL only if that same owned process persists. Regression covers an already exited SSH process. Connection loss is still bounded by the remote timeout. |
| Actual interpreter binding | Supervisor runtime was pinned while separately configured interpreters could execute cells. | Probe the selected Mac/G14/Orin interpreters, publish their intended heterogeneous package records, recheck at execution and reconcile every host summary. |
| Final artifact reconciliation | A valid head could be altered without changing AUROC, so AUROC-only rescoring was insufficient. | Validate float32 head layout and finite values, canonical hash against final receipt/summary, and independently recompute loss and AUROC. Regression modifies a saved head and requires rejection. |
| Queue reporting | Fit events wrote `queue_s` while the exporter read another field, yielding zero queue totals. | Export the actual field; regression requires a nonzero delay. |

The four twenty-round anchors now compare separately against pinned historical selection-stage simulator heads and against paired transport placement heads. Reusing the historical seed is labelled a replay. Source inventory includes the parity helper, protocol and new tests.

Timing and communication claims remain narrower than the original aspirational measurement list: actual protobuf message counts exclude wire framing/SSH, and serialization time is not isolated. CPU load and available-memory snapshots constrain interpretation; they do not remove background-workload confounding. Prospective shuffled execution order avoids always placing Mac-only first. Three hosts remain three failure domains, regardless of logical-client count.


## Postprocessing review and outcome

The separate audit initially trusted exported comparison booleans and did not reconcile all public values. Four injected cases—changed AUROC, changed speedup, contradictory difference/acceptance, duplicated placement pair—failed to be rejected. The correction regenerates the complete analysis from bound artifacts and strictly reconciles canonical JSON before issuing a receipt. All seven audit regressions then passed, as did the actual generated and medical audits. Source fix inspection reported no remaining material blocker.

Aggregate-only results review independently recomputed budgets, medians, numerical maxima, RSS and communication values. It found no material misstatement in the results note or reportv5 AppendixD, within public-evidence limits. All52cells completed and passed artifact reconciliation; the result remains a CPU execution qualification with an observed whole-cell slowdown, not a general scaling or security claim.
