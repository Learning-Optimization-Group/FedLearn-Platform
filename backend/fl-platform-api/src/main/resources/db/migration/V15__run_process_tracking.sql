-- BA-3: record the OS identity of a run's FL-server child process so a StartupReconciler can reap
-- orphans after a backend crash. FL servers are otherwise tracked only in a JVM in-memory map, so a
-- crash leaves runs stuck RUNNING with leaked ports and no way to find the orphaned children.
--
-- server_host + server_port already exist on runs (V8). This adds:
--   server_pid           -- the child's OS process id, checked for liveness via ProcessHandle.of(pid)
--   process_started_at   -- the child's OS start instant; paired with the PID it defends against PID
--                           reuse (a recycled PID on an unrelated process won't share this start time)
-- Both are nullable: historical runs and runs with no spawned process simply carry NULL.
ALTER TABLE runs ADD COLUMN server_pid BIGINT;
ALTER TABLE runs ADD COLUMN process_started_at TIMESTAMP WITH TIME ZONE;
