# Prospective three-host CPU protocol

26 October 2026. This protocol operationalizes the approved 25 October design. It freezes the benchmark before medical fitting; generated-data qualification is separate. The public preparation manifest binds this document, executable source, tests, actual selected interpreter versions, execution order and every server/host input commitment. A local Git checkpoint must preserve that public manifest before fitting.

## Budget and semantics

Exactly 52 new nonprivate transport fits, totaling 320 rounds:

- 48 systems cells: 3, 6, 24 or 80 logical clients × mean512/projected128 × training seeds 14001, 14002, 14003 × Mac-only/three-host placement, five rounds each.
- Four anchors: 80 clients × both methods × both placements, twenty rounds, historical training seed 13001. These are historical-seed replays, not fresh confirmation. Reusing existing simulator heads avoids two undeclared reference fits.

Mac coordinates and evaluates selection. Mac, G14 and Orin train their assigned whole-patient shards. Three-host IDs are assigned round-robin; Mac-only retains every logical ID. There are three physical hosts, never eighty independent devices or hospitals. Each host uses one interpreter, shared training arrays, independent per-client state and RNG, one Torch CPU thread and at most two simultaneous fits. Total allowed fits are two for Mac-only and six for three-host execution: this compares adding host resources rather than fixed-total-concurrency network overhead.

Both methods use zero binary linear heads, SGD learning rate 0.03, seven local epochs, batch size 256, final partial batches and uniform client weighting. Server clips each client delta to L2 norm 0.05 before mean aggregation in ascending logical-ID order. Original feature width is 512; projected width is 128 using the existing public projection seed 20261025. Projection occurs once on the Mac. No DP noise, attacks, GPU execution, new policy selection or holdout evaluation is included.

Canonical training/selection bundle report SHA-256: `dec54dfb580232ed6bf8da16ea9466248db58de4925285c8fb0d07a381562ffa`. Only assigned training arrays and immutable metadata go to remote hosts. Selection stays on the Mac. Endpoints, private identities, payloads and heads remain private.

## Qualification and failures

Before medical fitting require both methods to complete five generated-data rounds with one client per physical host and with Mac-only placement. Compare final heads against the real simulator implementation at maximum absolute float32 parameter error ≤ 1e-5. Do not loosen the tolerance after viewing outcomes. Private SSH process provenance and machine identities establish distinct physical contributors; source-declared host labels alone do not.

Exercise unknown registration, stale and ahead-round submissions, duplicate uploads and a registered client exiting before its upload through real generated-data gRPC. A bounded six-second deadline is used only for this failure test; medical rounds retain the public 600-second deadline. Incomplete cohorts never produce a successful head. Fixed-cohort upload acknowledgements indicate actual admission; generic transport acknowledgement behavior is preserved.

One attempt per declared cell, fresh output directories, no replacement fitting. Stop on a failed cell and retain its logs and remaining unrun cells. The supervisor owns its local processes; remote cleanup verifies exact executable and plan arguments, then uses a Linux process descriptor before signalling. A remote timeout bounds orphan lifetime if connectivity prevents cleanup. Existing inference services are not interrupted.

## Runtime, execution order and measurements

Probe the actual selected local and remote interpreters before preparation. Freeze Python, Torch, NumPy, gRPC, safetensors, operating system, architecture and Torch thread count. Verify again at execution startup and reconcile every host's startup runtime against those records. CPU Torch wheels may have different architecture/build suffixes. Source snapshots and training packs are verified on transfer.

Prospective order is the ascending SHA-256 of `20261026:` plus each cell name, published explicitly in the manifest. This avoids always running Mac-only before three-host placement. Do not reorder based on observed performance.

Supervisor elapsed time includes server/client startup, per-cell data staging and shutdown. First-round time starts at strategy initialization and includes readiness/client connection. Steady-round mean is the mean of adjacent round-completion differences on the server's monotonic clock, excluding the first round. Client fit and queue times are host-local elapsed intervals; do not subtract clocks across machines. Record host process peak RSS, process CPU time, available memory and start/end system load. Background inference and other host activity remain uncontrolled and must constrain interpretation.

Communication counters cover actual training-RPC protobuf requests/responses, including repeated model downloads. They exclude registration, heartbeats, framing, TLS and SSH. They are application message counts, not network-interface traffic. Serialization time is not separately instrumented; download/upload RPC durations do not isolate codec cost. No direct-TCP, end-to-end network throughput or GPU utilization claim follows.

## Verification and interpretation

Independently validate every saved final head's keys, float32 shapes, finite values and canonical hash against the final aggregation receipt and server summary. Recompute selection AUROC and cross-entropy loss with the same float32 linear scoring kernel; require absolute metric difference ≤ 1e-10. Compare each placement pair's final parameters to the prospective 1e-5 engineering tolerance, reporting actual differences even on failure.

Anchor simulator reference is the completed nonprivate ByzFL study summary SHA-256 `83f38253f61c5c674d408f4baac141ed7aff2661d304383aa2f8ba8c647f0409`. Its selection-stage final-head records are:

- mean512, seed13001: `d56b38c74c83ca9db769abe16e2a31ae203836639fb250e4b1500c4766019793`.
- projected128, seed13001: `740f12ba11e2a970a54b354f6279b7e02e04d3a0d4309b7321c0889d3c509dad`.

Read only these pinned selection-stage records. Compare the four anchors with their stored simulator final heads, separately from placement parity. No new simulator fit is needed. A mismatch is evidence to investigate and report, not permission to change the comparator or tolerance.

Five-round utility is a systems diagnostic, not the earlier twenty-round primary endpoint. Three paired training seeds support descriptive timing comparisons, not broad hardware rankings. Three shared, heterogeneous machines and background workloads limit external validity. This benchmark adds distributed execution evidence; it does not validate patient-level privacy, host-compromise resistance, clinical utility or a hospital deployment.
