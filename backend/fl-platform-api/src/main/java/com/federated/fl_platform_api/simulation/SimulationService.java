package com.federated.fl_platform_api.simulation;

import com.fasterxml.jackson.databind.JsonNode;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.service.AuthorizationService;
import com.federated.fl_platform_api.orchestration.ProcessTrees;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.scheduling.annotation.Scheduled;
import jakarta.annotation.PreDestroy;

import java.io.IOException;
import java.nio.file.Files;
import java.time.Duration;
import java.time.Instant;

import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.federated.fl_platform_api.simulation.SimulationDtos.*;

@Service
public class SimulationService {
    private final SimulationRunRepository repository;
    private final AuthorizationService authorization;
    private final SimulationProcessRunner runner;
    private final SimulationProperties properties;
    private final Object admission = new Object();
    private Active active;
    private boolean closing;
    static final List<SimulationStatus> ACTIVE = List.of(SimulationStatus.ACCEPTED, SimulationStatus.STARTING,
            SimulationStatus.RUNNING, SimulationStatus.STOPPING);
    private static final org.slf4j.Logger LOG = org.slf4j.LoggerFactory.getLogger(SimulationService.class);

    private static final class Active {
        final UUID id;
        final SimulationProcessRunner.OwnedProcess child;
        boolean stopRequested;
        SimulationStatus forced;
        String failureCode;
        Active(UUID id, SimulationProcessRunner.OwnedProcess child) { this.id = id; this.child = child; }
    }

    public SimulationService(SimulationRunRepository repository, AuthorizationService authorization,
                             SimulationProcessRunner runner, SimulationProperties properties) {
        this.repository = repository;
        this.authorization = authorization;
        this.runner = runner;
        this.properties = properties;
    }

    Limits limits() {
        int deadline = properties.getDeadlineSeconds();
        return new Limits(deadline >= 1 && deadline <= 180 ? deadline : 180, 4, 2147483648L, 1, SCOPE);
    }

    public Workloads workloads() {
        JsonNode metadata = SimulationJson.MAPPER.nullNode();
        boolean ready = false;
        try { metadata = runner.inspect(); ready = true; } catch (SimulationException ignored) { }
        return new Workloads(List.of(new Workload("nih-qualified", ready, ready ? null : "Qualified workload unavailable or not configured.",
                "No Finding (0) versus any finding (1)", "Frozen pretrained ResNet-18 features; validation only",
                ready ? properties.getReportSha256() : null, metadata.path("splits"), Map.of(
                    "head", List.of("linear", "mlp8", "mlp128", "mlp512"), "numClients", List.of(10, 80, 1000),
                    "rounds", List.of(1, 20), "localEpochs", List.of(1, 7), "aggregation", List.of("mean", "trimmed_mean"),
                    "clipNorm", List.of(.05, .4), "privacy", List.of("none", "central"), "epsilon", List.of(1, 4, 8),
                    "behavior", List.of("clean", "sign_flip", "label_flip"), "seed", List.of(0, 1000000000)), limits())), SCOPE);
    }

    public Resolved resolve(JsonNode choices) {
        runner.inspect();
        var resolution = runner.resolve(choices);
        return new Resolved("nih-qualified", resolution.hash(), resolution.recipe(), limits(), SCOPE);
    }

    public Accepted start(JsonNode choices) {
        Long owner = authorization.currentUser().getId();
        runner.inspect(); var resolution = runner.resolve(choices);
        synchronized (admission) {
            refresh();
            if (closing) throw SimulationException.unavailable();
            if (active != null || !repository.findByStatusIn(ACTIVE).isEmpty()) {
                throw new SimulationException(HttpStatus.CONFLICT, "WORKER_BUSY", "A simulation worker is already active or awaiting recovery.");
            }
            SimulationRun run = new SimulationRun();
            run.setId(UUID.randomUUID()); run.setCreatedBy(owner); run.setCreatedAt(Instant.now());
            run.setStatus(SimulationStatus.ACCEPTED); run.setConfigHash(resolution.hash());
            run.setRecipeJson(resolution.recipe().toString());
            run.setRequestedRounds(resolution.config().path("federation").path("rounds").asInt());
            run = repository.saveAndFlush(run);
            run.setStatus(SimulationStatus.STARTING); run = repository.saveAndFlush(run);
            try {
                var child = runner.launch(run.getId(), resolution.config());
                active = new Active(run.getId(), child);
                run.setStartedAt(Instant.now()); run.setWorkerPid(child.process().pid());
                run.setWorkerStartIdentity(child.startIdentity()); run.setWorkerExecutableIdentity(child.executableIdentity());
                repository.saveAndFlush(run);
            } catch (IOException | RuntimeException error) {
                if (active != null) {
                    active.forced = SimulationStatus.FAILED; active.failureCode = "LAUNCH_FAILED"; active.child.stop();
                    run.setStatus(SimulationStatus.STOPPING);
                } else {
                    run.setStatus(SimulationStatus.FAILED); run.setEndedAt(Instant.now());
                }
                run.setErrorCode("LAUNCH_FAILED"); run.setErrorMessage("The simulation worker could not be launched.");
                repository.saveAndFlush(run);
            }
            return new Accepted(run.getId(), run.getStatus(), run.getConfigHash(), resolution.recipe(), run.getCreatedAt(), SCOPE);
        }
    }

    @Scheduled(fixedDelay = 100)
    public void poll() {
        synchronized (admission) {
            try { refresh(); } catch (RuntimeException error) {
                // Admission stays occupied; the independent process watchdog still runs.
                LOG.error("Could not retain simulation process state", error);
            }
        }
    }

    private void refresh() {
        if (active == null) return;
        SimulationRun run = repository.findById(active.id).orElseThrow();
        boolean drained = active.child.drained();
        Integer exit = drained ? active.child.process().exitValue() : null;
        SimulationEvidence.Observation evidence = null;
        var directory = runner.jobDirectory(active.id).resolve("execution");
        if (Files.exists(directory.resolve("progress.json"))) {
            try {
                evidence = SimulationEvidence.read(run, directory, exit);
                run.setProgressJson(evidence.progress().toString()); run.setCompletedRounds(evidence.completedRounds());
            } catch (IOException | RuntimeException error) {
                active.forced = SimulationStatus.FAILED; active.failureCode = "INVALID_EVIDENCE"; active.child.stop();
            }
        }
        run.setWorkerExecutableIdentity(active.child.executableIdentity());
        if (!drained) {
            if (active.stopRequested || active.forced != null || active.child.timedOut()) run.setStatus(SimulationStatus.STOPPING);
            else if (evidence != null && evidence.progress().path("state").asText().equals("running")) run.setStatus(SimulationStatus.RUNNING);
            repository.saveAndFlush(run); return;
        }
        SimulationStatus status;
        if (active.forced != null) status = active.forced;
        else if (active.child.timedOut() || exit == 124) status = SimulationStatus.TIMED_OUT;
        else if (evidence != null && evidence.qualified() && exit == 0) status = SimulationStatus.COMPLETED;
        else if (active.stopRequested) status = SimulationStatus.STOPPED;
        else if (exit == 3 && evidence != null && evidence.progress().path("state").asText().equals("interrupted")) status = SimulationStatus.INTERRUPTED;
        else status = SimulationStatus.FAILED;
        run.setStatus(status); run.setExitCode(exit); run.setEndedAt(Instant.now());
        if (evidence != null && evidence.record().isObject()) {
            var record = (com.fasterxml.jackson.databind.node.ObjectNode) evidence.record();
            if (status != SimulationStatus.COMPLETED) record.putNull("qualification");
            record.put("status", status.name());
            record.withObject("terminal").put("status", status.name());
            run.setRecordJson(record.toString());
        }
        if (status == SimulationStatus.FAILED) {
            run.setErrorCode(active.failureCode == null ? "EXECUTION_UNQUALIFIED" : active.failureCode);
            run.setErrorMessage("The worker did not produce consistent, qualified completion evidence.");
        } else if (status == SimulationStatus.TIMED_OUT) {
            run.setErrorCode("DEADLINE_EXCEEDED"); run.setErrorMessage("The whole-process time budget was reached.");
        } else if (status == SimulationStatus.INTERRUPTED) {
            run.setErrorCode("OWNER_INTERRUPTED"); run.setErrorMessage("The worker lost its owning backend process.");
        }
        repository.saveAndFlush(run);
        active = null;
    }

    public Page list(int page, int size, String scope) {
        if (page < 0 || size < 1 || size > 100 || !(scope.equals("mine") || scope.equals("platform"))) {
            throw SimulationException.invalid();
        }
        if (scope.equals("platform") && !authorization.isPlatformAdmin()) throw new AccessDeniedException("Administrator scope required");
        var pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "createdAt").and(Sort.by("id")));
        var result = scope.equals("platform") ? repository.findAll(pageable)
                : repository.findByCreatedBy(authorization.currentUser().getId(), pageable);
        return new Page(result.stream().map(this::detail).toList(), result.getTotalElements(), page, size, SCOPE);
    }

    SimulationRun owned(UUID id) {
        SimulationRun run = repository.findById(id).orElseThrow(() -> ResourceNotFoundException.forEntity("Simulation", id));
        if (!authorization.isPlatformAdmin() && !run.getCreatedBy().equals(authorization.currentUser().getId())) {
            throw ResourceNotFoundException.forEntity("Simulation", id);
        }
        return run;
    }

    Detail detail(SimulationRun run) {
        JsonNode progress = SimulationJson.stored(run.getProgressJson());
        JsonNode record = SimulationJson.stored(run.getRecordJson());
        return new Detail(run.getId(), run.getStatus(), run.getConfigHash(), SimulationJson.stored(run.getRecipeJson()),
                run.getCreatedAt(), run.getStartedAt(), run.getEndedAt(), run.getCompletedRounds(), run.getRequestedRounds(),
                progress.path("rounds").isArray() ? progress.path("rounds") : SimulationJson.MAPPER.createArrayNode(),
                record.path("finalSummary"), record.path("qualification"), run.getRecordJson() != null,
                progress.path("record_encoding").isTextual() ? progress.path("record_encoding").asText() : null,
                run.getErrorCode(), run.getErrorMessage(), SCOPE);
    }

    public Detail get(UUID id) { return detail(owned(id)); }

    public Detail stop(UUID id) {
        synchronized (admission) {
            SimulationRun run = owned(id);
            refresh(); run = owned(id);
            if (run.getStatus().terminal()) return detail(run);
            if (active == null || !active.id.equals(id)) {
                throw new SimulationException(HttpStatus.CONFLICT, "OWNERSHIP_UNRESOLVED", "Process ownership is unresolved.");
            }
            active.stopRequested = true; active.child.stop();
            run.setStatus(SimulationStatus.STOPPING); repository.saveAndFlush(run);
            return detail(run);
        }
    }

    public JsonNode download(UUID id) {
        SimulationRun run = owned(id);
        if (run.getRecordJson() == null) throw new SimulationException(HttpStatus.CONFLICT, "RECORD_NOT_READY", "No retained research record is available.");
        return SimulationJson.stored(run.getRecordJson());
    }

    public void reconcile() {
        synchronized (admission) {
            if (active != null) return;
            for (SimulationRun run : repository.findByStatusIn(ACTIVE)) {
                if (run.getWorkerPid() == null) {
                    if (run.getStatus() == SimulationStatus.ACCEPTED) interrupted(run);
                    else unresolved(run);
                    continue;
                }
                var handle = ProcessHandle.of(run.getWorkerPid());
                if (handle.isEmpty() || !handle.get().isAlive()) { interrupted(run); continue; }
                var actualStart = handle.get().info().startInstant();
                if (run.getWorkerStartIdentity() == null || actualStart.isEmpty()) { unresolved(run); continue; }
                if (!actualStart.get().toString().equals(run.getWorkerStartIdentity())) {
                    // This PID now belongs to another process. Leave that process alone.
                    interrupted(run); continue;
                }
                if (!runner.matches(handle.get(), run.getId(), run.getWorkerStartIdentity(), run.getWorkerExecutableIdentity())) {
                    unresolved(run); continue;
                }
                try {
                    var tree = ProcessTrees.destroyForcibly(handle.get());
                    ProcessTrees.awaitExit(tree, 4);
                    interrupted(run);
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt(); unresolved(run);
                } catch (java.util.concurrent.ExecutionException | java.util.concurrent.TimeoutException error) {
                    unresolved(run);
                }
            }
        }
    }

    private void unresolved(SimulationRun run) {
        run.setErrorCode("OWNERSHIP_UNRESOLVED");
        run.setErrorMessage("Previous process ownership could not be established. New launches are blocked.");
        repository.saveAndFlush(run);
    }

    private void interrupted(SimulationRun run) {
        try {
            var evidence = SimulationEvidence.read(run, runner.jobDirectory(run.getId()).resolve("execution"), null);
            run.setProgressJson(evidence.progress().toString()); run.setCompletedRounds(evidence.completedRounds());
            if (evidence.record().isObject()) {
                var record = (com.fasterxml.jackson.databind.node.ObjectNode) evidence.record();
                record.putNull("qualification").put("status", "INTERRUPTED");
                record.withObject("terminal").put("status", "INTERRUPTED"); run.setRecordJson(record.toString());
            }
        } catch (IOException | RuntimeException ignored) {
            // Retain the previous accepted snapshot. Restart is never proof of exit-zero completion.
        }
        run.setStatus(SimulationStatus.INTERRUPTED); run.setEndedAt(Instant.now());
        run.setErrorCode("BACKEND_RESTART"); run.setErrorMessage("The previous backend stopped before observing completion.");
        repository.saveAndFlush(run);
    }

    @PreDestroy
    public void shutdown() {
        SimulationProcessRunner.OwnedProcess child;
        synchronized (admission) {
            closing = true;
            if (active == null) return;
            active.forced = SimulationStatus.INTERRUPTED; active.child.stop(); child = active.child;
        }
        try { child.awaitDrain(Duration.ofSeconds(6)); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        synchronized (admission) { refresh(); }
    }
}
