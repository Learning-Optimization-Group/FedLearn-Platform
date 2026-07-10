package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.EnrollmentDto;
import com.federated.fl_platform_api.dto.ModelBundleDto;
import com.federated.fl_platform_api.dto.RunManifestDto;
import com.federated.fl_platform_api.dto.RunStatusDto;
import com.federated.fl_platform_api.exception.ProjectStateException;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.RunEnrollmentRepository;
import com.federated.fl_platform_api.repository.RunRepository;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import com.federated.fl_platform_api.security.OrgScope;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.PathResource;
import org.springframework.core.io.Resource;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.core.env.Environment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import jakarta.annotation.PostConstruct;
import java.util.Arrays;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;

@Service
public class RunService {

    @Autowired private RunRepository runRepository;
    @Autowired private RunEnrollmentRepository enrollmentRepository;
    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;
    @Autowired private AuthorizationService authz;
    @Autowired private OrgScope orgScope;
    @Autowired private ConnectionTokenService tokenService;
    @Autowired private ObjectMapper objectMapper;

    private static final Logger log = LoggerFactory.getLogger(RunService.class);

    @Value("${app.fl-server.grpc-host:localhost}")
    private String grpcHost;

    @Autowired private Environment environment;

    // OP-15: the host actually ADVERTISED to clients. In dev, a default 'localhost' is upgraded to the
    // detected LAN IP so a same-LAN client (phone) can reach the FL server; an explicit FL_SERVER_GRPC_HOST
    // and every non-dev profile are used verbatim. Resolved once at startup.
    private String effectiveGrpcHost;

    @PostConstruct
    void resolveGrpcHost() {
        boolean isDev = Arrays.asList(environment.getActiveProfiles()).contains("dev");
        effectiveGrpcHost = FlGrpcHostResolver.resolve(grpcHost, isDev, LanAddressDetector::primarySiteLocalIPv4);
        if (!effectiveGrpcHost.equals(grpcHost)) {
            log.info("OP-15: dev profile — advertising FL gRPC host as {} (detected LAN IP) instead of the "
                    + "default '{}' so same-LAN clients (e.g. a phone) can connect; set FL_SERVER_GRPC_HOST "
                    + "to override.", effectiveGrpcHost, grpcHost);
        }
    }

    // Root of the per-run on-device training bundles staged by scripts/stage_model_bundle.py.
    @Value("${app.model-bundle.dir:/var/models}")
    private String modelBundleDir;

    @Value("${feature.model-bundle-delivery.enabled:true}")
    private boolean bundleDeliveryEnabled;

    // The only filenames the bundle file endpoint will serve (blocks path traversal / arbitrary reads).
    private static final Set<String> ALLOWED_BUNDLE_FILES =
            Set.of("loss.pte", "infer.pte", "inputs.f32", "targets.i64");

    private static final SecureRandom RANDOM = new SecureRandom();

    public Run createForStart(Project project, String strategy, int numRounds,
                              int minClients, int clientsPerRound) {
        Run run = new Run();
        run.setProjectId(project.getId());
        run.setStrategy(strategy);
        run.setNumRounds(numRounds);
        run.setMinClients(minClients);
        run.setClientsPerRound(clientsPerRound);
        run.setPartitioningMode(PartitioningMode.SHARDED);
        run.setStatus(RunStatus.STARTING);
        run.setSeed(RANDOM.nextLong());
        run.setRecipeKey(project.getModelType());
        run.setCreatedBy(project.getUser() != null ? project.getUser().getId() : null);
        run.setCreatedAt(Instant.now());
        return runRepository.save(run);
    }

    @Transactional
    public void markRunning(UUID runId, Integer port) {
        Run run = runRepository.findById(runId)
                .orElseThrow(() -> ResourceNotFoundException.run(runId));
        run.setServerHost(advertisedHost());
        run.setServerPort(port);
        run.setStatus(RunStatus.RUNNING);
        run.setStartedAt(Instant.now());
        runRepository.save(run);
    }

    @Transactional
    public void markFailed(UUID runId) {
        endRun(runId, RunStatus.FAILED);
    }

    @Transactional
    public void markStopped(UUID runId) {
        endRun(runId, RunStatus.STOPPED);
    }

    @Transactional
    public void markCompleted(UUID runId) {
        endRun(runId, RunStatus.COMPLETED);
    }

    private void endRun(UUID runId, RunStatus status) {
        if (runId == null) return;
        Run run = runRepository.findById(runId).orElse(null);
        if (run == null) return;
        run.setStatus(status);
        run.setServerPort(null);
        run.setEndedAt(Instant.now());
        runRepository.save(run);
    }

    @Transactional(readOnly = true)
    public RunStatusDto getStatus(UUID runId) {
        Run run = requireParticipantRun(runId);
        RunStatusDto dto = new RunStatusDto();
        dto.setRunId(runId);
        dto.setStatus(run.getStatus().name());
        if (run.getStatus() == RunStatus.RUNNING && run.getServerPort() != null) {
            dto.setGrpcEndpoint(endpoint(run));
            dto.setCaFingerprint(run.getGrpcCaFingerprint());
        }
        return dto;
    }

    @Transactional(readOnly = true)
    public RunManifestDto getManifest(UUID runId) {
        Run run = requireParticipantRun(runId);
        return toManifest(run);
    }

    RunManifestDto toManifest(Run run) {
        RunManifestDto m = new RunManifestDto();
        m.setRunId(run.getId());
        m.setProjectId(run.getProjectId());
        m.setRecipeKey(run.getRecipeKey());
        m.setStrategy(run.getStrategy());
        m.setNumRounds(run.getNumRounds());
        m.setClientsPerRound(run.getClientsPerRound());
        m.setPartitioningMode(run.getPartitioningMode().name());
        m.setSeed(run.getSeed());
        m.setTorchVersion(run.getTorchVersion());
        return m;
    }

    String endpoint(Run run) {
        String host = run.getServerHost() != null ? run.getServerHost() : advertisedHost();
        return host + ":" + run.getServerPort();
    }

    /** The advertised gRPC host: the OP-15-resolved effective host once startup has run, else the raw
     *  configured host (so Mockito unit tests without @PostConstruct keep their configured value). */
    private String advertisedHost() {
        return effectiveGrpcHost != null ? effectiveGrpcHost : grpcHost;
    }

    @Transactional
    public EnrollmentDto enroll(UUID runId) {
        Run run = runRepository.lockById(runId)
                .orElseThrow(() -> new ResourceNotFoundException("Run not found: " + runId));
        Project project = projectRepository.findById(run.getProjectId())
                .orElseThrow(() -> ResourceNotFoundException.project(run.getProjectId()));
        authz.requireOrgScope(project.getOrgId());
        User self = authz.currentUser();
        requireOwnerOrClient(project, self);

        if (run.getStatus() != RunStatus.RUNNING || run.getServerPort() == null) {
            throw new ProjectStateException(
                    "Run is not currently running (status=" + run.getStatus() + ")");
        }

        RunEnrollment enrollment = enrollmentRepository
                .findByIdRunIdAndIdUserId(runId, self.getId())
                .orElse(null);
        if (enrollment == null) {
            int next = enrollmentRepository.maxPartitionIdForRun(runId) + 1;
            if (run.getPartitioningMode() == PartitioningMode.SHARDED
                    && next >= run.getClientsPerRound()) {
                throw new ProjectStateException(
                        "Run is full (K=" + run.getClientsPerRound() + ")");
            }
            ClientKind kind = run.getPartitioningMode() == PartitioningMode.LOCAL
                    ? ClientKind.LOCAL : ClientKind.SHARD;
            enrollment = new RunEnrollment(
                    new RunEnrollmentId(runId, self.getId()), next, kind, Instant.now());
        }
        enrollment.setTokenIssuedAt(Instant.now());
        enrollment = enrollmentRepository.save(enrollment);

        String grpcEndpoint = endpoint(run);
        ConnectionTokenService.Minted minted = tokenService.mint(new ConnectionTokenService.Claims(
                self.getId(), runId, project.getId(), enrollment.getPartitionId(),
                grpcEndpoint, run.getGrpcCaFingerprint(), enrollment.getClientKind().name()));

        EnrollmentDto dto = new EnrollmentDto();
        dto.setRunId(runId);
        dto.setProjectId(project.getId());
        dto.setGrpcEndpoint(grpcEndpoint);
        dto.setPartitionId(enrollment.getPartitionId());
        dto.setClientKind(enrollment.getClientKind().name());
        dto.setCaFingerprint(run.getGrpcCaFingerprint());
        dto.setConnectionToken(minted.token());
        dto.setExpiresAt(minted.expiresAt());
        dto.setManifest(toManifest(run));
        return dto;
    }

    /** Serve the staged on-device training bundle metadata for a run (P2). Same org-scope +
     *  owner-or-CLIENT gate as status/enroll; file URLs point at {@code /api/runs/{id}/files/...}. */
    @Transactional(readOnly = true)
    public ModelBundleDto getModelBundle(UUID runId) {
        if (!bundleDeliveryEnabled) {
            throw new ProjectStateException("Model bundle delivery is disabled");
        }
        requireParticipantRun(runId);
        Path manifestPath = Path.of(modelBundleDir, runId.toString(), "manifest.json");
        if (!Files.isRegularFile(manifestPath)) {
            throw new ResourceNotFoundException("No model bundle staged for run " + runId);
        }
        JsonNode m;
        try {
            m = objectMapper.readTree(Files.readString(manifestPath));
        } catch (IOException e) {
            throw new ProjectStateException("Failed to read model bundle for run " + runId);
        }
        JsonNode mm = m.path("modelManifest");
        List<ModelBundleDto.ParamSpec> layout = new ArrayList<>();
        for (JsonNode p : mm.path("paramLayout")) {
            List<Integer> shape = new ArrayList<>();
            p.path("shape").forEach(s -> shape.add(s.asInt()));
            layout.add(new ModelBundleDto.ParamSpec(p.path("name").asText(), shape));
        }
        JsonNode ds = m.path("dataset");
        List<Integer> inputShape = new ArrayList<>();
        ds.path("inputShape").forEach(s -> inputShape.add(s.asInt()));
        String base = "/api/runs/" + runId + "/files/";
        return new ModelBundleDto(
                runId, layout, mm.path("totalParamCount").asLong(),
                base + "loss.pte", m.path("lossPte").path("sha256").asText(),
                base + "infer.pte", mm.path("inferSha256").asText(),
                base + ds.path("inputsFile").asText("inputs.f32"), ds.path("inputsSha256").asText(), inputShape,
                base + ds.path("targetsFile").asText("targets.i64"), ds.path("targetsSha256").asText());
    }

    /** Stream one whitelisted bundle binary. Same auth gate; the whitelist + a startsWith check block
     *  path traversal (404 on anything not in {@link #ALLOWED_BUNDLE_FILES} or not present). */
    @Transactional(readOnly = true)
    public Resource getModelFile(UUID runId, String filename) {
        if (!bundleDeliveryEnabled) {
            throw new ProjectStateException("Model bundle delivery is disabled");
        }
        requireParticipantRun(runId);
        if (!ALLOWED_BUNDLE_FILES.contains(filename)) {
            throw new ResourceNotFoundException("Unknown bundle file: " + filename);
        }
        Path base = Path.of(modelBundleDir, runId.toString()).normalize();
        Path file = base.resolve(filename).normalize();
        if (!file.startsWith(base) || !Files.isRegularFile(file)) {
            throw new ResourceNotFoundException("Bundle file not found: " + filename);
        }
        return new PathResource(file);
    }

    /** Loads a run and enforces org-scope + owner-or-CLIENT participation. */
    Run requireParticipantRun(UUID runId) {
        Run run = runRepository.findById(runId)
                .orElseThrow(() -> ResourceNotFoundException.run(runId));
        Project project = projectRepository.findById(run.getProjectId())
                .orElseThrow(() -> ResourceNotFoundException.project(run.getProjectId()));
        authz.requireOrgScope(project.getOrgId());
        requireOwnerOrClient(project, authz.currentUser());
        return run;
    }

    void requireOwnerOrClient(Project project, User self) {
        boolean isOwner = project.getUser() != null
                && project.getUser().getId().equals(self.getId());
        boolean isClient = membershipRepository
                .findByIdProjectIdAndIdUserId(project.getId(), self.getId())
                .map(m -> m.getRole() == MembershipRole.CLIENT).orElse(false);
        if (!isOwner && !isClient) {
            throw new AccessDeniedException("You are not a participant of this run's project");
        }
    }
}
