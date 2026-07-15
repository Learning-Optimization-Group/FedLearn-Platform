package com.federated.fl_platform_api.orchestration;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * DA-8: the FL-server orchestration seam. {@link FlServerManager} builds the command + policy and
 * delegates the actual process launch to a runner, so the raw {@code ProcessBuilder} mechanics live
 * behind an injectable interface rather than inline in the JVM orchestration class.
 *
 * <p>The default is {@link LocalProcessFlServerRunner} (a local child process). The seam is what makes
 * the spawn orchestration unit-testable with a fake runner (no real process), and is the extension
 * point for a future managed-task runner (ECS/Fargate — currently fail-closed in the manager).</p>
 *
 * <p>Contract: the runner applies {@code envCustomizer} to the child environment, merges stderr into
 * stdout, sets the working directory, and starts the process — nothing else. All policy (argv building,
 * secret/token scrubbing via the customizer, port reservation, run-state persistence, the startup probe
 * and log broadcasting) stays in the manager.</p>
 */
public interface FlServerProcessRunner {

    /**
     * Start a process for {@code command}. The runner MUST: apply {@code envCustomizer} to the child's
     * environment map, redirect stderr into stdout, set the working directory to {@code workingDir}, and
     * launch — returning a {@link SpawnedFlProcess} handle. It must NOT mutate {@code command}.
     *
     * @throws IOException if the process cannot be started (surfaced by the manager as a spawn failure).
     */
    SpawnedFlProcess start(List<String> command, Consumer<Map<String, String>> envCustomizer,
                           File workingDir) throws IOException;
}
