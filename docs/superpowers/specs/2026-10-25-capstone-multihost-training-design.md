# Three-host federated training and client-count benchmark

25 October 2026. Proposed design grounded in source inspection. Remote execution is not qualified: read-only SSH checks to Orin, G14 and the alternate G14 LAN route timed out. No remote installation, training or service change has occurred.

## Intent and scope

Run one federated training job whose logical clients execute on the Mac, G14 and Orin concurrently. Measure low and high logical-client counts and compare the same job against a Mac-only reference. Extend the existing FedLearn transport and coordinator rather than replacing them with a different federation framework.

Use the capstone's frozen Effusion features and linear heads first. This directly tests whether its training semantics survive real network transport and heterogeneous hosts. A full-image, LoRA or hospital deployment study is separate. The approved contribution-first report revision can proceed independently of hardware readiness and must describe this experiment as proposed until actual evidence exists.

Three hosts are three physical failure domains even when they run eighty logical clients. A host outage can remove many logical clients together. Repartitioning the same prepared NIH training patients is not external or institutional validation.

## Observed starting point

- `framework/src/fedlearn/server/server.py` starts a gRPC server around `FLCoordinator` and an existing strategy; TLS and client authentication have existing configuration seams.
- `fl-runtime/client.py` is the canonical executable client. It has remote-server addressing, project/partition identity, execution contracts and CPU/CUDA/MPS selection. Do not introduce a second FL client implementation.
- Its current parser limits partition IDs to zero through nine. Other recipes also carry default client counts. A validated run-wide client count and unique identities must replace those assumptions for this workload.
- The qualified Effusion bundle loader and patient-whole workload currently belong to the simulator path. The image-level NIH runtime recipe targets Pneumonia, so reusing its name or loader would silently change the task.
- The capstone uses uniform client averaging. The generic `FedAvgAggregator` uses example counts; transport reuse must not silently substitute this weighted objective.
- The ByzFL adapter and `ComposedStrategy` share the strategy interface, but the gRPC coordinator accesses strategy initialization and configuration details that require an actual compatibility test. Similar interfaces alone are not proof of integration.
- The device helper can fall back from an unavailable requested accelerator to CPU. Benchmark preflight must record and verify the actual device; a fallback is not a CUDA/MPS result.

## Architecture and data boundary

Mac hosts the coordinator, authoritative run manifest and selection-only evaluator. It also runs its assigned clients. G14 and Orin run their assigned clients against that coordinator over the existing private network route. Verify the chosen route and peer access before binding the training service; do not expose a public endpoint or disable existing production security gates.

Every logical client has a unique project/run/client identity, a fixed whole-patient training shard, a local optimizer, independent client RNG state and separate lifecycle/metrics. Host processes may share a read-only local feature cache to avoid duplicating arrays in memory. Start with independently supervised clients and bounded concurrent fits per host. If a scheduler is needed, it must preserve separate per-client state and prevent duplicate or stale submissions; it must use the canonical client path.

Transfer only assigned training shards and necessary immutable model/projection metadata to each remote host. Keep selection features and labels on the Mac. No holdout/test payloads, saved holdout heads or old holdout predictions are read or transferred. Pin shard contents and ownership before launch. Public exports contain counts, aggregates and nonidentifying hashes; host endpoints, credentials and patient identifiers stay local.

Clients receive a round's head, train locally, and submit their update. The coordinator completes the declared cohort, aggregates exactly once and broadcasts the next global head. Validate transport serialization, client identity, sample count semantics, round numbers and cancellation through the actual gRPC path.

## Host and client-count choices

These are proposed defaults, not measured capacity promises:

| Logical clients | Mac / G14 / Orin assignment | Purpose |
| --- | --- | --- |
| 3 | 1 / 1 / 1 | Prove every physical host contributes to one job |
| 6 | 2 / 2 / 2 | Low-count concurrent-client behavior |
| 24 | 8 / 8 / 8 | Moderate multiplexing |
| 80 | 27 / 27 / 26 | Match the earlier logical cohort size |

Use all CPUs first, one training thread per active client and a fixed initial concurrency limit of two fitting clients per host after the three-client smoke. Keep idle clients registered/heartbeating while queued. Record queue wait separately from training and transport time. Measure memory before increasing concurrency; do not launch eighty independent Torch imports without checking resource capacity. The client count is the cohort cardinality, not the concurrency limit.

The initial assignment is balanced to keep the design interpretable. A later capacity-weighted assignment needs a separate comparison because it changes the measured host/straggler mix. Do not stop the Orin inference service or reclaim its GPU without specific need and authorization. CPU qualification can be attempted alongside existing workloads, whose interference must be recorded.

After CPU correctness is established, a separate capability-gated run can use Mac MPS and available NVIDIA CUDA devices. Install architecture-appropriate dependencies in isolated environments. Record real versions, capabilities and memory; do not assume the G14 GPU type or assume generic CUDA wheels work on Jetson. Do not require bitwise cross-platform replay: use a generated-data float64 reference with declared float32 parameter tolerances and report measured differences. GPU throughput is a different comparison from CPU transport correctness.

## Ordered qualification and bounded benchmark

1. **Readiness:** establish SSH and direct gRPC reachability, hardware/runtime versions, writable isolated run directories, available memory and actual device. No model fitting. Record failures; do not install into shared environments.
2. **Generated-data qualification:** one client per physical host, five rounds, no DP or attacks. Prove three distinct host contributions per round, model change, valid submissions and clean shutdown. Compare the same shards/recipe against a single-host reference. Exercise missing, duplicate and stale submissions plus one controlled client exit on generated data before medical fitting.
3. **Frozen transport benchmark:** four client counts, mean512/projected128, three new paired seeds, Mac-only versus three-host placement: 48 selection-only cells, five rounds each, no DP/attacks. Fixed total training patients and local recipe make this a strong-scaling/system-overhead study. The five-round utility is an execution diagnostic, not a reproduction of the historical twenty-round primary endpoint. Freeze seed IDs, mappings, source/runtime hashes and tolerances in a prospective protocol before these fits.
4. **Twenty-round qualification anchor:** at eighty clients, the two methods and two placements, one additional frozen seed: four nonprivate cells. Reconcile complete cohorts, update/model differences and selection metrics against an equivalent single-host transport run and the existing simulator semantics. Keep simulator-versus-transport and placement-versus-placement differences separate.
5. **Optional private/accelerator extension:** separate amendment after qualification. Not required to establish that the hosts train together. Recalibrate privacy for its actual cohort, release count, clipping and reducer. No new holdout evaluation.

The proposed medical budget is 52 cells and 320 rounds. It is provisional until readiness and the generated-data qualification establish feasible deadlines and memory. A changed budget must be declared before medical results are viewed; failed or unrun declared cells remain visible. Do not tune the original capstone's policies using this systems comparison.

## Semantics and acceptance criteria

- Both methods use the existing Effusion label, prepared features, zero head and public projection basis. Preserve the capstone's local-training order, optimizer and uniform-client objective; do not silently use a demo task or example-weighted mean.
- Medical cells have full participation. An incomplete cohort fails the cell and suppresses a successful-model claim. Diagnostic dropout recovery is tested separately without assigning the original fixed-cohort privacy guarantee to a partial result.
- Before medical fitting, require five complete generated-data rounds, exactly three distinct physical contributors, no duplicate aggregation and max absolute head error at most 1e-5 against a deterministic reference. This is a proposed engineering tolerance, not a statistical equivalence margin; record the observed error and investigate failures without loosening it after viewing medical outcomes.
- For CPU placement comparisons, require the same logical shards, fixed data order, float32 update representation, client RNG streams and a stable aggregation order. Same-host exact replay and cross-host numerical tolerances are different checks. For every medical result independently rescore the saved final head on selection and reconcile exported metrics within 1e-10 using the same scoring kernel.
- A server must reject unknown identities, stale rounds and incompatible head layouts. Logs and aggregate receipts must identify actual admitted/aggregated clients and hosts, not just submitted updates.

## Measurements and claims

Report run startup separately from steady-state rounds: per-client queue/train/serialization/transfer time, aggregate round duration, bytes sent/received and their application-versus-network scope, completion/failure counts, host process memory, and available CPU/GPU utilization. Do not sum monotonic clock timestamps from different machines. Use client-local elapsed intervals and server-local receipt/barrier times; synchronized wall clocks are not assumed to have submillisecond accuracy.

A matched three-seed timing difference is descriptive. Background workloads, shared LAN/Tailscale transport, scheduler limits and three physical hosts limit extrapolation. Large logical-client counts test multiplexing and coordination overhead, not eighty independent network links or hospitals.

Privacy remains a separate measurement: at fixed C, changing eighty privacy clients to three changes mean sensitivity from 2C/80 to 2C/3. Keeping eighty logical clients on three physical hosts preserves that logical adjacency only under the stated trust model; it does not establish privacy against host compromise or correlated malicious ownership. Fixed-cohort GDP accounting cannot be carried unchanged to silently dropped clients.

## Decisions left to implementation

Choose environment locations, host-local cache format and supervised process mechanics after readiness. Preserve canonical runtime boundaries, per-client isolation, resource caps and the source-frozen protocol. Add a dedicated Effusion vector recipe/loader rather than changing the Pneumonia task. Prefer reuse of the existing coordinator/transport/strategy seams; add narrowly scoped adapters only where verified contracts differ. All new protected/private transport claims require their own qualification.

## Outstanding user-dependent condition

The remote machines must be powered on and reachable from the Mac's homelab network. Their connectivity and free resources have not been verified. The submitted question asks whether both are online; elapsed time is not evidence of readiness or approval to interrupt existing services.
