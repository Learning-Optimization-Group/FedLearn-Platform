package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import com.federated.fl_platform_api.orchestration.ProcessTrees;
import org.springframework.stereotype.Component;

import java.io.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;

/** Argument-list launches with a parent pipe, bounded output and an external deadline. */
@Component
public class SimulationProcessRunner implements AutoCloseable {
    private static final int OUTPUT_LIMIT = 65536;
    private static final String PROBE = """
            import json,sys
            from pathlib import Path
            from fedlearn.simulation.nih_cache import load_nih_cache
            cache=load_nih_cache(Path(sys.argv[1]),expected_report_sha256=sys.argv[2])
            report=cache['provenance']
            if report['pretrained'] is not True: raise ValueError('pretrained cache required')
            print(json.dumps({'splits':report['splits'],'dataset':report['dataset'],'pretrained':True}))
            """;
    private final SimulationProperties properties;
    private final ExecutorService threads = Executors.newVirtualThreadPerTaskExecutor();
    private final Set<OwnedProcess> children = ConcurrentHashMap.newKeySet();

    public SimulationProcessRunner(SimulationProperties properties) { this.properties = properties; }

    record Resolution(JsonNode config, String hash, JsonNode recipe) { }

    boolean configured() {
        return properties.isEnabled() && properties.getInterpreter() != null && properties.getInterpreter().isAbsolute()
                && Files.isExecutable(properties.getInterpreter()) && properties.getSourceRoot() != null
                && Files.isDirectory(properties.getSourceRoot()) && properties.getCache() != null
                && properties.getStorage() != null && properties.getReportSha256() != null
                && properties.getReportSha256().matches("[0-9a-f]{64}")
                && properties.getDeadlineSeconds() >= 1 && properties.getDeadlineSeconds() <= 180;
    }

    JsonNode inspect() {
        if (!configured()) throw SimulationException.unavailable();
        try {
            JsonNode result = command(List.of("-c", PROBE, properties.getCache().toAbsolutePath().normalize().toString(), properties.getReportSha256()));
            if (!result.path("splits").isObject() || !result.path("pretrained").asBoolean()) throw new IOException("Invalid probe");
            return result;
        } catch (IOException | InterruptedException error) {
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
            throw SimulationException.unavailable();
        }
    }

    Resolution resolve(JsonNode choices) {
        if (!configured()) throw SimulationException.unavailable();
        Path input = null;
        try {
            input = Files.createTempFile(storageRoot(), ".choices-", ".json");
            Files.writeString(input, choices.toString());
            JsonNode resolved = command(List.of("-m", "fedlearn.lab_worker", "resolve", "--choices", input.toString(),
                    "--cache", properties.getCache().toAbsolutePath().normalize().toString(), "--report-sha256", properties.getReportSha256()));
            String hash = resolved.path("config_hash").asText();
            if (!hash.matches("[0-9a-f]{64}") || !resolved.path("config").isObject() || !resolved.path("display_recipe").isObject()
                    || !properties.getReportSha256().equals(resolved.path("display_recipe").path("featureReportSha256").asText())) {
                throw new IOException("Invalid resolver contract");
            }
            return new Resolution(resolved.path("config"), hash, resolved.path("display_recipe"));
        } catch (IOException | InterruptedException error) {
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
            // Includes every nonzero resolver exit, including private Python tracebacks.
            throw SimulationException.invalid();
        } finally {
            if (input != null) try { Files.deleteIfExists(input); } catch (IOException ignored) { }
        }
    }

    private ProcessBuilder builder(List<String> arguments) {
        List<String> command = new ArrayList<>(); command.add(properties.getInterpreter().toString()); command.addAll(arguments);
        ProcessBuilder builder = new ProcessBuilder(command).redirectErrorStream(true);
        Map<String, String> environment = builder.environment(); environment.clear();
        environment.put("PATH", "/usr/bin:/bin");
        environment.put("PYTHONPATH", properties.getSourceRoot().toAbsolutePath().normalize().toString());
        environment.put("PYTHONNOUSERSITE", "1"); environment.put("PYTHONUNBUFFERED", "1");
        environment.put("HF_HUB_OFFLINE", "1"); environment.put("HF_DATASETS_OFFLINE", "1");
        environment.put("OMP_NUM_THREADS", "4"); environment.put("MKL_NUM_THREADS", "4");
        return builder;
    }

    private JsonNode command(List<String> arguments) throws IOException, InterruptedException {
        Process process = builder(arguments).redirectErrorStream(false).start();
        var bytes = threads.submit(() -> capture(process.getInputStream(), null));
        var diagnostics = threads.submit(() -> capture(process.getErrorStream(), null));
        try {
            process.getOutputStream().close();
            if (!process.waitFor(15, TimeUnit.SECONDS)) throw new IOException("Worker control timeout");
            byte[] output = bytes.get(2, TimeUnit.SECONDS);
            diagnostics.get(2, TimeUnit.SECONDS);
            if (process.exitValue() != 0 || output.length >= OUTPUT_LIMIT) throw new IOException("Worker control refused");
            return SimulationJson.parse(output, OUTPUT_LIMIT);
        } catch (ExecutionException | TimeoutException error) {
            throw new IOException("Worker control failed", error);
        } finally {
            if (process.isAlive()) {
                var tree = ProcessTrees.destroyForcibly(process.toHandle());
                try { ProcessTrees.awaitExit(tree, 3); } catch (ExecutionException | TimeoutException ignored) { }
            }
            process.getInputStream().close(); process.getErrorStream().close(); bytes.cancel(true); diagnostics.cancel(true);
        }
    }

    private static byte[] capture(InputStream source, Path log) throws IOException {
        try (source; var buffer = new ByteArrayOutputStream();
             OutputStream destination = log == null ? OutputStream.nullOutputStream()
                     : Files.newOutputStream(log, StandardOpenOption.CREATE_NEW)) {
            byte[] block = new byte[4096]; int count;
            while ((count = source.read(block)) != -1) {
                int retain = Math.min(count, OUTPUT_LIMIT - buffer.size());
                if (retain > 0) { buffer.write(block, 0, retain); destination.write(block, 0, retain); }
            }
            return buffer.toByteArray();
        }
    }

    private Path storageRoot() throws IOException {
        Path root = properties.getStorage().toAbsolutePath().normalize();
        if (Files.isSymbolicLink(root)) throw new IOException("Storage symlink refused");
        Files.createDirectories(root);
        return root;
    }

    Path jobDirectory(UUID id) { return properties.getStorage().toAbsolutePath().normalize().resolve(id.toString()); }

    public OwnedProcess launch(UUID id, JsonNode config) throws IOException {
        storageRoot(); Path job = jobDirectory(id); Files.createDirectory(job);
        try { Files.setPosixFilePermissions(job, java.nio.file.attribute.PosixFilePermissions.fromString("rwx------")); }
        catch (UnsupportedOperationException ignored) { }
        Path input = job.resolve("config.json"); Files.writeString(input, config.toString(), StandardOpenOption.CREATE_NEW);
        List<String> arguments = runArguments(id);
        long launched = System.nanoTime();
        Process process = builder(arguments).start();
        try {
            OwnedProcess owned = new OwnedProcess(process, job, launched);
            children.add(owned); owned.start(); return owned;
        } catch (RuntimeException error) {
            ProcessTrees.destroyForcibly(process); throw error;
        }
    }

    private List<String> runArguments(UUID id) {
        Path job = jobDirectory(id);
        return List.of("-m", "fedlearn.lab_worker", "run", "--config", job.resolve("config.json").toString(),
                "--output", job.resolve("execution").toString(), "--report-sha256", properties.getReportSha256(),
                "--run-id", id.toString(), "--deadline-seconds", Integer.toString(properties.getDeadlineSeconds()));
    }

    boolean matches(ProcessHandle handle, UUID id, String startIdentity, String executableIdentity) {
        var info = handle.info();
        if (startIdentity == null || !info.startInstant().map(Object::toString).filter(startIdentity::equals).isPresent()) return false;
        try {
            return executableIdentity != null && info.command().isPresent() && Files.isSameFile(Path.of(info.command().get()), Path.of(executableIdentity))
                    && info.arguments().map(args -> Arrays.asList(args).equals(runArguments(id))).orElse(false);
        } catch (IOException error) { return false; }
    }

    public final class OwnedProcess {
        private final Process process;
        private final Path job;
        private final long launched;
        private final String startIdentity;
        private volatile String executableIdentity;
        private final CompletableFuture<Boolean> drained = new CompletableFuture<>();
        private final Map<ProcessHandle, String> descendants = new HashMap<>();
        private volatile long stoppedAt;
        private volatile boolean timedOut;

        private OwnedProcess(Process process, Path job, long launched) {
            this.process = process; this.job = job; this.launched = launched;
            this.startIdentity = process.info().startInstant().orElseThrow(() -> new IllegalStateException("Missing process identity")).toString();
            this.executableIdentity = process.info().command().orElseThrow(() -> new IllegalStateException("Missing executable identity"));
        }

        private void start() {
            threads.submit(() -> capture(process.getInputStream(), job.resolve("process.log")));
            threads.submit(() -> {
                try {
                    while (true) {
                        if (process.isAlive()) process.descendants().forEach(child -> child.info().startInstant()
                                .ifPresent(start -> descendants.put(child, start.toString())));
                        long now = System.nanoTime();
                        if (process.isAlive() && now - launched >= Duration.ofSeconds(properties.getDeadlineSeconds()).toNanos()) {
                            requestStop(true);
                        }
                        boolean rootAlive = process.isAlive();
                        if (!rootAlive || (stoppedAt != 0 && now - stoppedAt >= Duration.ofSeconds(2).toNanos())) {
                            if (rootAlive) ProcessTrees.destroyForcibly(process);
                            descendants.forEach((child, start) -> {
                                if (child.isAlive() && child.info().startInstant().map(Object::toString).filter(start::equals).isPresent()) child.destroyForcibly();
                            });
                        }
                        if (!process.isAlive() && descendants.keySet().stream().noneMatch(ProcessHandle::isAlive)) break;
                        Thread.sleep(20);
                    }
                    process.waitFor(); process.getOutputStream().close();
                    drained.complete(true); children.remove(this);
                } catch (Exception error) {
                    drained.completeExceptionally(error);
                }
            });
        }

        public void stop() { requestStop(false); }

        private synchronized void requestStop(boolean deadline) {
            if (stoppedAt == 0) {
                timedOut = deadline;
                stoppedAt = System.nanoTime();
                if (process.isAlive()) process.toHandle().destroy();
            }
        }
        public Process process() { return process; }
        public String startIdentity() { return startIdentity; }
        public String executableIdentity() {
            var info = process.info();
            if (process.isAlive() && info.startInstant().map(Object::toString).filter(startIdentity::equals).isPresent()) {
                info.command().ifPresent(command -> executableIdentity = command);
            }
            return executableIdentity;
        }
        public boolean timedOut() { return timedOut; }
        public boolean drained() { return drained.isDone() && !drained.isCompletedExceptionally(); }
        public boolean awaitDrain(Duration timeout) throws InterruptedException {
            try { return drained.get(timeout.toMillis(), TimeUnit.MILLISECONDS); }
            catch (ExecutionException | TimeoutException error) { return false; }
        }
    }

    @Override public void close() {
        List<OwnedProcess> snapshot = List.copyOf(children); snapshot.forEach(OwnedProcess::stop);
        for (OwnedProcess child : snapshot) {
            try { child.awaitDrain(Duration.ofSeconds(6)); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        }
        threads.shutdownNow();
    }
}
