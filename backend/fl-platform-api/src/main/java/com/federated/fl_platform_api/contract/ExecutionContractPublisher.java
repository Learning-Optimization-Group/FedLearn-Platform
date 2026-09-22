package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunIntent;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.service.ModelBundleStagingListener;
import com.federated.fl_platform_api.service.RegistryModelResolver;
import com.federated.fl_platform_api.service.RunService;
import com.fedlearn.contract.v1.ExecutionContract;
import com.fedlearn.contract.v1.ModelTraining;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

/**
 * Decides each run's execution contract once its bundle staging ends (Stage 2E).
 *
 * <p>When a run with an intent snapshot is staged, the publisher reads the staged bundle, resolves the plan for the
 * initial model file the FL server loaded (the registry head on a continued run, otherwise the project's model file),
 * assembles the contract, checks it against the legacy manifest fields, and stores it, where it is validated. The
 * first source that fails or disagrees decides the run UNAVAILABLE with its reason instead. Runs without an intent snapshot are legacy and are left alone.</p>
 */
@Component
public class ExecutionContractPublisher implements ModelBundleStagingListener {

    private static final Logger log = LoggerFactory.getLogger(ExecutionContractPublisher.class);

    private final RunRepository runs;
    private final ProjectRepository projects;
    private final RegistryModelResolver registry;
    private final StagedBundleReader bundles;
    private final ExecutionPlanResolver plans;
    private final ExecutionContractStore store;
    private final RunService runService;

    @Value("${app.contract.max-transient-retries:3}")
    private int maxTransientRetries;

    @Value("${app.contract.retry-backoff-ms:1000}")
    private long retryBackoffMs;

    public ExecutionContractPublisher(RunRepository runs, ProjectRepository projects, RegistryModelResolver registry,
                                      StagedBundleReader bundles, ExecutionPlanResolver plans,
                                      ExecutionContractStore store, RunService runService) {
        this.runs = runs;
        this.projects = projects;
        this.registry = registry;
        this.bundles = bundles;
        this.plans = plans;
        this.store = store;
        this.runService = runService;
    }

    @Override
    public void onStaged(UUID runId) {
        Optional<Run> found = runs.findById(runId);
        if (found.isEmpty() || found.get().getIntent().isEmpty()) {
            return;
        }
        Run run = found.get();
        RunIntent intent = run.getIntent().get();
        Project project = projects.findById(run.getProjectId()).orElse(null);
        if (project == null) {
            decide(run, ContractUnavailableReason.NOT_REPRESENTABLE, "the run's project no longer exists");
            return;
        }
        StagedBundle bundle;
        try {
            bundle = bundles.read(runId);
        } catch (IOException e) {
            decide(run, ContractUnavailableReason.STAGING_FAILED, e.getMessage());
            return;
        }
        String initialModelPath = registry.resolveModelPath(project).orElse(project.getModelPath());
        if (initialModelPath == null) {
            decide(run, ContractUnavailableReason.NOT_REPRESENTABLE, "the project has no initial model file");
            return;
        }
        Path initialModel = Path.of(initialModelPath);
        try {
            if (changedSince(initialModel, run.getCreatedAt())) {
                decide(run, ContractUnavailableReason.NOT_REPRESENTABLE,
                        "the initial model file changed after the run started");
                return;
            }
            ModelTraining plan = plans.resolve(run.getRecipeKey(), run.getStrategy(), intent.trainingArm().name(),
                    initialModel);
            ExecutionContract contract = ExecutionContractAssembler.assemble(run, plan, bundle, new ExecutionContractAssembler.Policy(
                    ExecutionContractStore.SERVER_PROTOCOL_VERSION, maxTransientRetries, retryBackoffMs));
            // Old clients act on the legacy manifest fields and updated clients on the contract, so the two must
            // describe the same run before the contract is published.
            List<String> disagreements = LegacyManifestEquivalence.disagreements(runService.legacyManifest(run),
                    contract);
            if (!disagreements.isEmpty()) {
                decide(run, ContractUnavailableReason.INVALID_CONTRACT,
                        "the contract disagrees with the legacy manifest on " + disagreements);
                return;
            }
            PublicationOutcome outcome = store.publish(run, contract);
            log.info("execution contract for run {}: {}", runId, outcome);
        } catch (NotRepresentableException e) {
            decide(run, ContractUnavailableReason.NOT_REPRESENTABLE, e.getMessage());
        } catch (IOException | IllegalArgumentException e) {
            decide(run, ContractUnavailableReason.NOT_REPRESENTABLE,
                    "could not resolve the execution plan: " + e.getMessage());
        }
    }

    @Override
    public void onStagingFailed(UUID runId, boolean timedOut, String detail) {
        runs.findById(runId).filter(run -> run.getIntent().isPresent()).ifPresent(run -> decide(run,
                timedOut ? ContractUnavailableReason.STAGING_TIMED_OUT : ContractUnavailableReason.STAGING_FAILED,
                detail));
    }

    private void decide(Run run, ContractUnavailableReason reason, String detail) {
        PublicationOutcome outcome = store.markUnavailable(run, reason, detail);
        log.warn("no execution contract for run {} ({}: {}): {}", run.getId(), reason, detail, outcome);
    }

    /** True when the file was written after the run was created, so it may no longer be the server's initial model. */
    private static boolean changedSince(Path file, Instant runCreatedAt) throws IOException {
        return runCreatedAt != null && Files.getLastModifiedTime(file).toInstant().isAfter(runCreatedAt);
    }
}
