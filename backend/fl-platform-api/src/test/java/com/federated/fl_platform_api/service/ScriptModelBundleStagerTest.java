package com.federated.fl_platform_api.service;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.test.util.ReflectionTestUtils;

/**
 * MO-15: the auto-stage impl is best-effort + flag-gated + idempotent, and builds the export command
 * from the runId + recipe. These pin each of those against a fake {@link
 * ScriptModelBundleStager.ProcessInvoker} (no real child process).
 */
class ScriptModelBundleStagerTest {

    private ScriptModelBundleStager stager(String dir, boolean enabled, AtomicReference<List<String>> captured,
                                           AtomicInteger calls, int exitOrThrow) {
        ScriptModelBundleStager s = new ScriptModelBundleStager();
        ReflectionTestUtils.setField(s, "enabled", enabled);
        ReflectionTestUtils.setField(s, "modelBundleDir", dir);
        ReflectionTestUtils.setField(s, "exportScript", "scripts/export_model.py");
        ReflectionTestUtils.setField(s, "stageScript", "scripts/stage_model_bundle.py");
        ReflectionTestUtils.setField(s, "fixtureRecipesCsv", "TINYNET_GOLDEN");
        ReflectionTestUtils.setField(s, "pythonExecutable", "python3");
        ReflectionTestUtils.setField(s, "timeoutSeconds", 5L);
        s.setExecutor(Runnable::run);  // run staging synchronously so assertions don't race the worker
        s.setInvoker((cmd, timeout) -> {
            calls.incrementAndGet();
            captured.set(cmd);
            if (exitOrThrow < 0) {
                throw new IllegalStateException("invoker blew up");
            }
            return exitOrThrow;
        });
        return s;
    }

    /** Arranges a stager that exercises the REAL runLocalProcess/ProcessBuilder path (no setInvoker),
     *  running staging synchronously (same-thread executor) so the process outcome is assertable. */
    private ScriptModelBundleStager realStager(String dir, String pythonExe, String script, long timeout) {
        ScriptModelBundleStager s = new ScriptModelBundleStager();
        ReflectionTestUtils.setField(s, "enabled", true);
        ReflectionTestUtils.setField(s, "modelBundleDir", dir);
        ReflectionTestUtils.setField(s, "exportScript", script);
        ReflectionTestUtils.setField(s, "pythonExecutable", pythonExe);
        ReflectionTestUtils.setField(s, "timeoutSeconds", timeout);
        s.setExecutor(Runnable::run);
        return s;  // default invoker (runLocalProcess) stays in place
    }

    @Test
    void disabled_neverInvokesTheExport(@TempDir Path dir) {
        AtomicInteger calls = new AtomicInteger();
        ScriptModelBundleStager s = stager(dir.toString(), false, new AtomicReference<>(), calls, 0);
        s.stageForRun(UUID.randomUUID(), "CNN");
        assertEquals(0, calls.get(), "disabled stager must not launch the export");
    }

    @Test
    void enabled_buildsExportCommandFromRunIdAndRecipe(@TempDir Path dir) {
        AtomicReference<List<String>> cmd = new AtomicReference<>();
        AtomicInteger calls = new AtomicInteger();
        ScriptModelBundleStager s = stager(dir.toString(), true, cmd, calls, 0);
        UUID runId = UUID.randomUUID();

        s.stageForRun(runId, "PNEUMONIA_CNN");

        assertEquals(1, calls.get());
        // The script is resolved to an absolute path (found by walking up from the CWD); assert the
        // structure + tail rather than the brittle relative literal.
        List<String> c = cmd.get();
        assertEquals("python3", c.get(0));
        assertTrue(c.get(1).endsWith("export_model.py"), "export path: " + c.get(1));
        assertEquals(List.of(runId.toString(), "--recipe", "PNEUMONIA_CNN", "--out", dir.toString()),
                c.subList(2, c.size()));
    }

    @Test
    void enabled_fixtureRecipe_buildsStageCommandFromRunId_withNoRecipeFlag(@TempDir Path dir) {
        AtomicReference<List<String>> cmd = new AtomicReference<>();
        AtomicInteger calls = new AtomicInteger();
        ScriptModelBundleStager s = stager(dir.toString(), true, cmd, calls, 0);
        UUID runId = UUID.randomUUID();

        s.stageForRun(runId, "TINYNET_GOLDEN");

        assertEquals(1, calls.get());
        // Fixture-backed recipe -> the stdlib-only stage_model_bundle.py path (resolved to absolute),
        // WITHOUT --recipe (that script copies the committed golden fixture and takes no recipe key).
        List<String> c = cmd.get();
        assertEquals("python3", c.get(0));
        assertTrue(c.get(1).endsWith("stage_model_bundle.py"), "stage path: " + c.get(1));
        assertEquals(List.of(runId.toString(), "--out", dir.toString()), c.subList(2, c.size()));
        assertFalse(c.contains("--recipe"), "fixture path must not pass --recipe");
    }

    @Test
    void enabled_fixtureRecipe_isCaseInsensitive_stillUsesFixturePath(@TempDir Path dir) {
        AtomicReference<List<String>> cmd = new AtomicReference<>();
        ScriptModelBundleStager s = stager(dir.toString(), true, cmd, new AtomicInteger(), 0);
        UUID runId = UUID.randomUUID();

        s.stageForRun(runId, "tinynet_golden");   // lower-case must match the fixture-recipe set

        List<String> c = cmd.get();
        assertEquals("python3", c.get(0));
        assertTrue(c.get(1).endsWith("stage_model_bundle.py"), "stage path: " + c.get(1));
        assertEquals(List.of(runId.toString(), "--out", dir.toString()), c.subList(2, c.size()));
    }

    // ── BA-16 path-resolution: the script is resolved to an absolute path by walking up from the JVM
    //    working dir, so a relative default (scripts/…) is found even when the backend runs from a module
    //    subdir (backend/fl-platform-api) under bootRun — the live bug the fake-invoker tests missed. ──

    @Test
    void resolveScriptPath_walksUpFromModuleSubdir_toRepoRootScript(@TempDir Path root) throws Exception {
        Files.createDirectories(root.resolve("scripts"));
        Path script = root.resolve("scripts/stage_model_bundle.py");
        Files.writeString(script, "# fixture stager");
        Path moduleCwd = root.resolve("backend/fl-platform-api");   // where bootRun actually runs
        Files.createDirectories(moduleCwd);

        String resolved = ScriptModelBundleStager.resolveScriptPath(
                "scripts/stage_model_bundle.py", moduleCwd.toString());
        assertEquals(script.toAbsolutePath().normalize().toString(), resolved,
                "must resolve the repo-root script from a module subdir CWD");
    }

    @Test
    void resolveScriptPath_absolutePassesThrough_andMissingReturnedUnchanged(@TempDir Path root) {
        assertEquals("/abs/x.py", ScriptModelBundleStager.resolveScriptPath("/abs/x.py", root.toString()),
                "an absolute path is used verbatim");
        assertEquals("scripts/nope.py", ScriptModelBundleStager.resolveScriptPath("scripts/nope.py", root.toString()),
                "a not-found script returns unchanged so the miss surfaces as a logged failure");
    }

    @Test
    void alreadyStaged_isIdempotent_skipsTheExport(@TempDir Path dir) throws Exception {
        AtomicInteger calls = new AtomicInteger();
        UUID runId = UUID.randomUUID();
        // Pre-create <dir>/<runId>/manifest.json to simulate an already-staged run.
        Path runDir = dir.resolve(runId.toString());
        Files.createDirectories(runDir);
        Files.writeString(runDir.resolve("manifest.json"), "{}");

        ScriptModelBundleStager s = stager(dir.toString(), true, new AtomicReference<>(), calls, 0);
        s.stageForRun(runId, "CNN");
        assertEquals(0, calls.get(), "an already-staged run must not be re-exported");
    }

    @Test
    void blankOrNullRecipe_skips(@TempDir Path dir) {
        AtomicInteger calls = new AtomicInteger();
        ScriptModelBundleStager s = stager(dir.toString(), true, new AtomicReference<>(), calls, 0);
        s.stageForRun(UUID.randomUUID(), "   ");
        s.stageForRun(UUID.randomUUID(), null);
        s.stageForRun(null, "CNN");
        assertEquals(0, calls.get(), "missing runId/recipe must skip, not launch a broken command");
    }

    @Test
    void nonZeroExit_doesNotThrow(@TempDir Path dir) {
        ScriptModelBundleStager s = stager(dir.toString(), true, new AtomicReference<>(), new AtomicInteger(), 3);
        assertDoesNotThrow(() -> s.stageForRun(UUID.randomUUID(), "CNN"));
    }

    @Test
    void invokerThrows_isSwallowed_neverPropagates(@TempDir Path dir) {
        ScriptModelBundleStager s = stager(dir.toString(), true, new AtomicReference<>(), new AtomicInteger(), -1);
        assertDoesNotThrow(() -> s.stageForRun(UUID.randomUUID(), "CNN"),
                "best-effort contract: an invoker failure must never propagate");
    }

    // ── Real ProcessBuilder invoker (runLocalProcess): the production path the fake invoker bypasses. ──
    // Uses /bin/sh scripts (the command's extra args are ignored by the script). Unix-only.

    @Test
    void realInvoker_fastExitZero_succeeds(@TempDir Path dir) throws Exception {
        assumeTrue(!isWindows(), "uses a /bin/sh script");
        Path script = dir.resolve("ok.sh");
        Files.writeString(script, "#!/bin/sh\nexit 0\n");
        ScriptModelBundleStager s = realStager(dir.toString(), "sh", script.toString(), 5);
        assertDoesNotThrow(() -> s.stageForRun(UUID.randomUUID(), "CNN"));
    }

    @Test
    void realInvoker_nonZeroExit_isSwallowed(@TempDir Path dir) throws Exception {
        assumeTrue(!isWindows(), "uses a /bin/sh script");
        Path script = dir.resolve("fail.sh");
        Files.writeString(script, "#!/bin/sh\necho boom\nexit 3\n");
        ScriptModelBundleStager s = realStager(dir.toString(), "sh", script.toString(), 5);
        assertDoesNotThrow(() -> s.stageForRun(UUID.randomUUID(), "CNN"));
    }

    @Test
    void realInvoker_hangingExport_isKilledByTimeout_withinBound(@TempDir Path dir) throws Exception {
        assumeTrue(!isWindows(), "uses a /bin/sh script");
        Path script = dir.resolve("hang.sh");
        Files.writeString(script, "#!/bin/sh\nsleep 30\n");
        ScriptModelBundleStager s = realStager(dir.toString(), "sh", script.toString(), 1);  // 1s timeout
        // Must return far inside the 30s sleep — proving waitFor(timeout) fires, destroyForcibly runs, and
        // the TimeoutException is swallowed (the whole point of the timeout branch, previously untested).
        assertTimeoutPreemptively(Duration.ofSeconds(15),
                () -> s.stageForRun(UUID.randomUUID(), "CNN"));
    }

    private static boolean isWindows() {
        return System.getProperty("os.name", "").toLowerCase().startsWith("win");
    }
}
