package com.federated.fl_platform_api.orchestration;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * DA-8: the default {@link FlServerProcessRunner} — spawns the FL server as a local child process via
 * {@link ProcessBuilder}, exactly as {@link FlServerManager} did inline before the seam was
 * extracted. This is a pure move: the byte-for-byte spawn behaviour (env customization → redirect
 * stderr → working dir → start) is unchanged, which is what keeps the real-spawn integration test green.
 */
public final class LocalProcessFlServerRunner implements FlServerProcessRunner {

    @Override
    public SpawnedFlProcess start(List<String> command, Consumer<Map<String, String>> envCustomizer,
                                  File workingDir) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(command);
        // The manager scrubs the web-auth secret and injects the per-run token here (SE-1/SE-7).
        envCustomizer.accept(pb.environment());
        pb.redirectErrorStream(true);
        pb.directory(workingDir);
        return new LocalSpawnedFlProcess(pb.start());
    }

    /** Thin adapter over {@link Process} exposing only the operations {@link FlServerManager} needs. */
    static final class LocalSpawnedFlProcess implements SpawnedFlProcess {
        private final Process process;

        LocalSpawnedFlProcess(Process process) {
            this.process = process;
        }

        @Override
        public long pid() {
            return process.pid();
        }

        @Override
        public Optional<Instant> startInstant() {
            return process.info().startInstant();
        }

        @Override
        public ProcessHandle toHandle() {
            return process.toHandle();
        }

        @Override
        public InputStream getInputStream() {
            return process.getInputStream();
        }

        @Override
        public boolean waitFor(long timeout, TimeUnit unit) throws InterruptedException {
            return process.waitFor(timeout, unit);
        }

        @Override
        public int exitValue() {
            return process.exitValue();
        }

        @Override
        public boolean isAlive() {
            return process.isAlive();
        }

        @Override
        public void destroyForcibly() {
            process.destroyForcibly();
        }
    }
}
