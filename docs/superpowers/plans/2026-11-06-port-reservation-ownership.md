# Bind each training port to its reservation

A delayed old-child exit can release the replacement child's reservation when a
stop/start reuses the same numeric port. Current project ownership compares only
project UUID and port; the global reserved set also releases by number. This is a
current restart defect, independent of future logical-run continuation.

Allocate a unique reservation for every successful port allocation. Carry that
reservation through project publication, exit callbacks, startup failure cleanup
and explicit stop. Both project and global ownership remove only the expected
reservation. Do not hold a lifecycle lock during database work or process waits.
Keep process termination, token policy and persisted run transitions scoped as they
are; execution-attempt fencing remains a separate required continuation slice.

- [x] Reproduce a delayed old exit after same-project/same-port replacement; verify
  the replacement stays tracked and the one-port pool remains exhausted.
- [x] Reproduce replay of an old global release after reallocation; a stale cleanup
  cannot free the replacement reservation. Preserve normal stop/exit/failure controls.
- [x] Replace numeric ownership with a unique reservation in both ownership maps;
  update every allocation, publication and release path, including startup failures.
- [x] Run lifecycle/orchestration controls and the full backend coverage gate;
  review ownership and interleavings. Do not overwrite the running API's jar.
- [x] Commit the verified correction and retain the broader attempt-identity,
  immutable history, private-checkpoint and API/browser continuation requirements.


## Verification and boundaries

Both added cases failed against the original manager: the replacement's reservation
became free after a delayed exit or replayed release. All 85 manager/orchestration
checks then passed. The complete backend gate passed 1,346 tests with two opt-in
checks skipped; both actual-worker controls passed separately. Line coverage was
86.55%. All 1,193 staged source hashes matched, and the separate deployable jar
built successfully. Review found no introduced issues in this bounded correction.

The running qualification API jar was preserved during these tests. This change
protects in-memory port reservation identity. Startup re-adoption still needs to
restore a known reservation before a warming child binds; persisted attempt identity,
retained attempt history and public continuation remain separate unfinished work.
