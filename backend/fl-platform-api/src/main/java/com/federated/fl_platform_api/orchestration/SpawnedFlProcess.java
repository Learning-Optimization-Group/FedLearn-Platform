package com.federated.fl_platform_api.orchestration;

import java.io.InputStream;
import java.time.Instant;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

/**
 * DA-8: a handle to a spawned FL-server process, abstracting away {@code java.lang.Process} so the
 * orchestration in {@link FlServerManager} no longer touches raw process mechanics directly.
 *
 * <p>Exposes exactly the operations the manager needs during the startup window — identity (for BA-3
 * reconciliation), merged stdout for log broadcasting, the startup-probe wait, and forced teardown —
 * plus a {@link ProcessHandle} for the long-lived tracking map. The default implementation
 * ({@link LocalProcessFlServerRunner.LocalSpawnedFlProcess}) wraps a local child process; a future
 * managed-task runner (ECS) would implement this same seam without the manager changing.</p>
 */
public interface SpawnedFlProcess {

    /** The OS process id (BA-3 anti-orphan record). */
    long pid();

    /** The process start instant, if the OS exposes it — the anti-PID-reuse guard for BA-3. */
    Optional<Instant> startInstant();

    /** A restart-survivable handle for the tracking map (a re-adopted orphan yields only a handle). */
    ProcessHandle toHandle();

    /** Merged stdout+stderr stream (the runner redirects stderr into stdout at spawn). */
    InputStream getInputStream();

    /** Block up to {@code timeout} for the process to exit; true if it exited within the window. */
    boolean waitFor(long timeout, TimeUnit unit) throws InterruptedException;

    /** The exit code — only valid once the process has exited. */
    int exitValue();

    /** Whether the process is still running. */
    boolean isAlive();

    /** Forcibly terminate the process (SIGKILL semantics on POSIX). */
    void destroyForcibly();
}
