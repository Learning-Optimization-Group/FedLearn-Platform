package com.federated.fl_platform_api.service;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import jakarta.annotation.PreDestroy;

/**
 * MO-15: default {@link ModelBundleStager} — stages a per-run on-device bundle by invoking
 * {@code scripts/export_model.py <runId> --recipe <recipeKey> --out <dir>} into the model-bundle dir the
 * backend serves ({@code RunService}). Flag-gated ({@code feature.model-bundle-autostage.enabled}, default
 * off), idempotent (skips if {@code <dir>/<runId>/manifest.json} exists), and best-effort — every failure
 * path is logged and swallowed.
 *
 * <p><b>Off the request thread.</b> The export can be slow (ExecuTorch lowering) or hang, so it runs on a
 * bounded background executor rather than on the caller's thread. {@link #stageForRun} only schedules and
 * returns immediately — it never blocks a project start, defers the FL-server spawn, or holds the BA-2
 * per-project start lock. The phone tolerates a brief 404 window right after a run goes RUNNING (its
 * provisioning treats a missing bundle as a graceful "not staged yet"), which async staging closes once
 * the export lands. On a host without the ExecuTorch toolchain the export simply exits non-zero and the
 * phone keeps 404-ing — same as before this hook existed.</p>
 */
@Component
public class ScriptModelBundleStager implements ModelBundleStager {

    private static final Logger log = LoggerFactory.getLogger(ScriptModelBundleStager.class);

    /** The process launch, behind a seam so the stager is unit-testable without a real child process. */
    @FunctionalInterface
    interface ProcessInvoker {
        /** Run {@code command}, waiting up to {@code timeoutSeconds}; return the exit code (or throw). */
        int run(List<String> command, long timeoutSeconds) throws Exception;
    }

    @Value("${feature.model-bundle-autostage.enabled:false}")
    private boolean enabled;
    @Value("${app.model-bundle.dir:/var/models}")
    private String modelBundleDir;
    @Value("${python.script.export-model.path:scripts/export_model.py}")
    private String exportScript;
    @Value("${app.model-bundle.autostage.python:python3}")
    private String pythonExecutable;
    @Value("${app.model-bundle.autostage.timeout-seconds:120}")
    private long timeoutSeconds;

    private ProcessInvoker invoker = ScriptModelBundleStager::runLocalProcess;

    // Staging runs here, NOT on the request thread — one bounded daemon worker so a slow/hung export can
    // never delay a start or hold the start lock. A full backlog (rapid starts) drops extra tasks
    // (best-effort: the phone 404s, an operator can re-stage). Overridable in tests with a same-thread executor.
    private Executor executor = defaultExecutor();
    private boolean ownsExecutor = true;

    private static ExecutorService defaultExecutor() {
        ThreadPoolExecutor ex = new ThreadPoolExecutor(
                1, 1, 30L, TimeUnit.SECONDS, new LinkedBlockingQueue<>(64),
                r -> {
                    Thread t = new Thread(r, "model-bundle-stager");
                    t.setDaemon(true);
                    return t;
                },
                (r, pool) -> log.warn("model-bundle auto-stage dropped: staging backlog is full"));
        ex.allowCoreThreadTimeOut(true);
        return ex;
    }

    /** Seam override for tests (package-visible): production uses the local {@link ProcessBuilder} invoker. */
    void setInvoker(ProcessInvoker invoker) {
        if (invoker != null) {
            this.invoker = invoker;
        }
    }

    /** Seam override for tests: inject a same-thread executor ({@code Runnable::run}) so staging runs
     *  synchronously and assertions don't race the background worker. */
    void setExecutor(Executor executor) {
        if (executor != null) {
            if (ownsExecutor && this.executor instanceof ExecutorService es) {
                es.shutdownNow();
            }
            this.executor = executor;
            this.ownsExecutor = false;
        }
    }

    @PreDestroy
    void shutdown() {
        if (ownsExecutor && executor instanceof ExecutorService es) {
            es.shutdownNow();
        }
    }

    @Override
    public void stageForRun(UUID runId, String recipeKey) {
        if (!enabled) {
            return;  // feature off — no-op, don't even schedule
        }
        try {
            executor.execute(() -> doStage(runId, recipeKey));
        } catch (RejectedExecutionException e) {
            // Scheduling itself failed (shutting down); still never fail the caller's start.
            log.warn("model-bundle auto-stage not scheduled for run {} (rejected): {}", runId, e.toString());
        }
    }

    /** The staging work, run on the background executor. Never throws (best-effort). */
    private void doStage(UUID runId, String recipeKey) {
        try {
            if (runId == null || recipeKey == null || recipeKey.isBlank()) {
                log.warn("model-bundle auto-stage skipped: missing runId or recipeKey (runId={}, recipe={})",
                        runId, recipeKey);
                return;
            }
            Path runDir = Path.of(modelBundleDir, runId.toString());
            if (Files.exists(runDir.resolve("manifest.json"))) {
                log.debug("model-bundle already staged for run {} — skipping auto-stage", runId);
                return;
            }
            List<String> command = List.of(
                    pythonExecutable, exportScript, runId.toString(),
                    "--recipe", recipeKey, "--out", modelBundleDir);
            int exit = invoker.run(command, timeoutSeconds);
            if (exit == 0) {
                log.info("auto-staged model bundle for run {} (recipe {})", runId, recipeKey);
            } else {
                log.warn("model-bundle auto-stage for run {} exited {} (recipe {}); a mobile client will "
                        + "get 404 until it is staged", runId, exit, recipeKey);
            }
        } catch (Exception e) {
            // Never propagate (best-effort): a missing bundle is a graceful 404, not a failure. Restore the
            // interrupt flag if the wait was interrupted, so shutdown/cancellation still propagates upward.
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            log.warn("model-bundle auto-stage failed for run {} (recipe {}): {}", runId, recipeKey, e.toString());
        }
    }

    /**
     * Launch the export as a child process. Output is redirected to a temp file (never a pipe) so the
     * child can't deadlock on a full stdout buffer, and {@code waitFor} enforces the timeout; on failure
     * the captured output is logged for diagnostics.
     */
    private static int runLocalProcess(List<String> command, long timeoutSeconds) throws Exception {
        File out = File.createTempFile("mo15-stage-", ".log");
        try {
            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectErrorStream(true);
            pb.redirectOutput(out);
            Process p = pb.start();
            boolean finished = p.waitFor(timeoutSeconds, TimeUnit.SECONDS);
            if (!finished) {
                p.destroyForcibly();
                throw new TimeoutException("export_model timed out after " + timeoutSeconds + "s");
            }
            int exit = p.exitValue();
            if (exit != 0) {
                String tail = Files.readString(out.toPath(), StandardCharsets.UTF_8).strip();
                if (!tail.isBlank()) {
                    log.warn("export_model output (exit {}): {}", exit, tail);
                }
            }
            return exit;
        } finally {
            if (!out.delete()) {
                out.deleteOnExit();
            }
        }
    }
}
