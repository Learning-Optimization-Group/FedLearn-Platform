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
import com.federated.fl_platform_api.security.FlClientCertificateAuthority;
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
    @Autowired private FlClientCertificateAuthority clientCa;
    @Autowired private ObjectMapper objectMapper;

    private static final Logger log = LoggerFactory.getLogger(RunService.class);

    @Value("${app.fl-server.grpc-host:localhost}")
    private String grpcHost;

    // BA-16: when auto-detecting (dev + default localhost), prefer a Tailscale/CGNAT 100.64.0.0/10 address
    // over a site-local LAN IP. Default true because our cross-network demo devices reach the server via the
    // tailnet; set FL_SERVER_PREFER_CGNAT=false to prefer the LAN IP, or FL_SERVER_GRPC_HOST to pin a host.
    @Value("${app.fl-server.prefer-cgnat:true}")
    private boolean preferCgnat;

    @Autowired private Environment environment;

    // OP-15 / BA-16: the host actually ADVERTISED to clients. In dev, a default 'localhost' is upgraded to
    // the detected client-reachable IP (Tailscale/CGNAT-preferred, then site-local LAN) so a remote client
    // (a phone on the tailnet) can reach the FL server; an explicit FL_SERVER_GRPC_HOST and every non-dev
    // profile are used verbatim. Resolved once at startup.
    private String effectiveGrpcHost;

    @PostConstruct
    void resolveGrpcHost() {
        boolean isDev = Arrays.asList(environment.getActiveProfiles()).contains("dev");
        effectiveGrpcHost = FlGrpcHostResolver.resolve(
                grpcHost, isDev, () -> LanAddressDetector.primaryReachableIPv4(preferCgnat));
        if (!effectiveGrpcHost.equals(grpcHost)) {
            log.info("OP-15/BA-16: dev profile — advertising FL gRPC host as {} (detected {} IP) instead of "
                    + "the default '{}' so remote clients (e.g. a phone on the tailnet) can connect; set "
                    + "FL_SERVER_GRPC_HOST to override or FL_SERVER_PREFER_CGNAT=false to prefer the LAN IP.",
                    effectiveGrpcHost, preferCgnat ? "Tailscale/CGNAT-preferred" : "LAN-preferred", grpcHost);
        }
    }

    // Root of the per-run on-device training bundles staged by scripts/stage_model_bundle.py.
    @Value("${app.model-bundle.dir:/var/models}")
    private String modelBundleDir;

    @Value("${feature.model-bundle-delivery.enabled:true}")
    private boolean bundleDeliveryEnabled;

    // The only filenames the bundle file endpoint will serve (blocks path traversal / arbitrary reads).
    private static final Set<String> ALLOWED_BUNDLE_FILES =
            Set.of("loss.pte", "infer.pte", "inputs.f32", "targets.i64", "trainable.pte");

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
        if (isTerminal(run.getStatus())) {
            // A fast run can reach a terminal state (its /finished callback wrote COMPLETED) before the
            // start thread's markRunning executes — do not revert a finished run to RUNNING.
            return;
        }
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
        if (isTerminal(run.getStatus())) {
            // Terminal status is write-once: a completed run must not be flipped to FAILED by the
            // start-probe catch or the reconciler sweep (the FAILED-clobbers-COMPLETED race the
            // StartupReconciler comment flags). The first terminal write wins.
            return;
        }
        run.setStatus(status);
        run.setServerPort(null);
        run.setEndedAt(Instant.now());
        runRepository.save(run);
    }

    /** COMPLETED / FAILED / STOPPED are terminal — the first terminal write wins and no later eager
     *  writer (a racing markRunning from the start thread, a late markFailed from the start-probe catch
     *  or the reconciler sweep) may revert it. */
    private static boolean isTerminal(RunStatus status) {
        return status == RunStatus.COMPLETED
                || status == RunStatus.FAILED
                || status == RunStatus.STOPPED;
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
        m.setFirstOrderSupported(hasStagedTrainableBundle(run.getId()));
        return m;
    }

    /** True iff a trainable .pte was actually staged for this run (the served bundle manifest carries a
     *  non-blank {@code modelManifest.trainablePtePath}). Tying the mobile first-order capability flag to
     *  the artifact's real presence — not the recipe alone — means a run whose trainable export failed or
     *  hasn't staged yet reports false, so the phone fail-closes to DeComFL rather than fetching a 404. */
    private boolean hasStagedTrainableBundle(UUID runId) {
        if (!bundleDeliveryEnabled) {
            return false;
        }
        Path manifestPath = Path.of(modelBundleDir, runId.toString(), "manifest.json");
        if (!Files.isRegularFile(manifestPath)) {
            return false;
        }
        try {
            JsonNode m = objectMapper.readTree(Files.readString(manifestPath));
            return !m.path("modelManifest").path("trainablePtePath").asText("").isBlank();
        } catch (IOException e) {
            return false;
        }
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
        // SE-14: size the token to the whole run (numRounds), not a fixed TTL — otherwise a long run
        // outlives its token and the client is rejected mid-training once require-client-auth is on.
        ConnectionTokenService.Minted minted = tokenService.mint(new ConnectionTokenService.Claims(
                self.getId(), runId, project.getId(), enrollment.getPartitionId(),
                grpcEndpoint, run.getGrpcCaFingerprint(), enrollment.getClientKind().name()),
                tokenService.ttlForRun(run.getNumRounds()));

        EnrollmentDto dto = new EnrollmentDto();
        dto.setRunId(runId);
        dto.setProjectId(project.getId());
        dto.setGrpcEndpoint(grpcEndpoint);
        dto.setPartitionId(enrollment.getPartitionId());
        dto.setClientKind(enrollment.getClientKind().name());
        dto.setCaFingerprint(run.getGrpcCaFingerprint());
        dto.setConnectionToken(minted.token());
        // SE-12: when client-cert issuance is enabled, mint a short-lived per-client mTLS cert bound to THIS
        // identity (CN=userId) + run. Off by default, so enrollment is unchanged until the operator turns it on
        // and points the FL server's FEDLEARN_GRPC_ROOT_CERT at the issuing CA.
        if (clientCa.isEnabled()) {
            FlClientCertificateAuthority.IssuedClientCert issued =
                    clientCa.issueClientCert(String.valueOf(self.getId()), runId);
            dto.setClientCertPem(issued.clientCertPem());
            dto.setClientKeyPem(issued.clientKeyPem());
            dto.setCaFingerprint(issued.caFingerprint());
        }
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
        // First-order trainable graph — present only when the staged bundle carries a trainablePtePath.
        // Absent => null url/sha + empty names, which the mobile client reads as "DeComFL-only".
        String trainablePte = mm.path("trainablePtePath").asText("");
        String trainablePteUrl = trainablePte.isBlank() ? null : base + "trainable.pte";
        String trainableSha256 = trainablePte.isBlank() ? null : mm.path("trainableSha256").asText();
        List<String> trainableParamNames = new ArrayList<>();
        mm.path("trainableParamNames").forEach(n -> trainableParamNames.add(n.asText()));
        return new ModelBundleDto(
                runId, layout, mm.path("totalParamCount").asLong(),
                base + "loss.pte", m.path("lossPte").path("sha256").asText(),
                base + "infer.pte", mm.path("inferSha256").asText(),
                base + ds.path("inputsFile").asText("inputs.f32"), ds.path("inputsSha256").asText(), inputShape,
                base + ds.path("targetsFile").asText("targets.i64"), ds.path("targetsSha256").asText(),
                trainablePteUrl, trainableSha256, trainableParamNames);
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
