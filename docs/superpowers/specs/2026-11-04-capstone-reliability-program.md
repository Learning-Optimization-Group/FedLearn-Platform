# Capstone integration, recovery and qualification

## Objective

Complete the seven improvement areas requested after milestone three. Deliver working software and direct execution evidence. Preserve completed studies, raw results, the medical evaluation freeze and existing local files. The implementation starts from the capstone branch because it contains the current simulator and the mobile history.

## Critical assessment and sequence

1. **Research-to-device integration.** Publish and execute one real frozen-backbone/head recipe through the existing dashboard, backend contract, desktop client and Android native trainer. Reuse the established trainable-state wire and export machinery. Bind the frozen state, preprocessing and objective; do not enable an arm merely because a graph can be exported. Start with a generated-data conformance fixture, then a supported public-data recipe. Keep OVA_LP separate because its objective differs.
2. **Durable recovery.** Save complete state at a completed round boundary before publishing the next model. A checkpoint needs configuration/run identity, tensor layout, model, strategy state, round history and necessary randomness. Restore optimizer moments and privacy release/accounting state where applicable. Refuse unsupported strategies and incompatible checkpoints. Never resume a half-aggregated round. Qualify storage, coordinator integration, process restart and the API workflow separately.
3. **Signed release and TLS.** Use a persistent release identity and the existing signing workflow. Verify release installation, HTTPS enrollment, authenticated TLS training, final-model reload and same-key update with dataset preservation. Credentials and trust material stay private. Local fixtures do not establish deployed qualification. Preserve the installed app/data; do not uninstall to evade a signer mismatch.
4. **Failure acceptance suite.** Exercise actual processes/transports for interrupted upload, lost response, client exit, API restart, training-server exit, deadline and screen-off/Wi-Fi operation. Assert unique accepted contributions, truthful terminal states, retained committed work and cleanup. Keep browser E2E under homelab/fedlearn. Automate a generated-data real-training smoke gate independently of physical-device qualification.
5. **Measured performance.** Instrument startup, data staging, serialization, download/polling and training separately. Reproduce a generated-data baseline, then evaluate persistent workers/cache reuse against it. Preserve the frozen three-host measurements. Report whole-run and steady-round costs; no acceleration claim from added machines alone.
6. **Security composition.** Implement the Android scalar secure-aggregation protocol with authenticated identity/key exchange, share sealing/masking, threshold/dropout handling and interoperability checks. Keep weight-update secure aggregation a separately specified protocol. Distributed DP needs enforced or verified clipping and independently qualified accounting; do not offer the simulator result as a deployed untrusted-client defense.
7. **Research evidence.** Preserve the medical freeze and closed holdout. Define a matched established-baseline protocol on generated/public workloads, matching privacy unit, adjacency, released transcript, clipping, attacker access, cohort, training budget and uncertainty target. A baseline may be refused as incomparable; extra seeds are not independent-cohort validation. Independently verify outputs and retain failures.

Recovery infrastructure and generated-data fault tests begin first because they provide shared acceptance machinery. Frozen-head publication and native conformance follow. Release qualification, performance and security each require their own acceptance gates; none is satisfied by a unit-test count. All seven remain in scope, with staged implementation rather than a claim that one month guarantees all outcomes.

## Initial recovery design

Create private, immutable checkpoint generations using JSON plus safetensors, without pickle. Publish each generation atomically after flushing file contents; synchronize the directory entry. Retain prior committed generations. Validate checksums, schema, run/configuration identity, strategy configuration, original-model layout and round bounds before changing live state. Reject a corrupt newest generation instead of silently rewinding. A conflicting generation must never overwrite prior evidence.

The strategy adapter explicitly enumerates supported state. Start with FedAvg, FedProx and FedOpt and extend private/robust/DeComFL state only with trajectory and privacy tests. Unsupported checkpointing fails before the server listens. Production checkpoint configuration is opt-in until the backend/API continuation workflow is qualified. Checkpoint files are private server state, never research-record downloads or model-registry artifacts.

At a network round boundary, aggregation/evaluation and disk writing run outside the coordinator lock while submissions remain closed. Heartbeats and Stop stay responsive. A checkpoint-write failure stops the run before the next round is served. A restore starts at the next round and rejects old-round retries. Interrupted uncommitted rounds are retrained from the last committed model.

## Completion evidence

Each numbered area needs reviewed source changes plus its stated execution acceptance. Track implementation and tests in the companion plan. Physical-device, deployment, cryptographic and empirical claims require their actual runtime evidence. No push, external publication, destructive environment change or medical-data experiment is implied by local implementation.
