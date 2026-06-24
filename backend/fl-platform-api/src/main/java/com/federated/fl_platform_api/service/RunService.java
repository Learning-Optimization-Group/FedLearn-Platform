package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.EnrollmentDto;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.security.SecureRandom;
import java.time.Instant;
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

    @Value("${app.fl-server.grpc-host:localhost}")
    private String grpcHost;

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
        run.setServerHost(grpcHost);
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
        String host = run.getServerHost() != null ? run.getServerHost() : grpcHost;
        return host + ":" + run.getServerPort();
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
