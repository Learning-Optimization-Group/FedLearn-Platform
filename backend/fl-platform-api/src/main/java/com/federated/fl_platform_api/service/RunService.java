package com.federated.fl_platform_api.service;

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
}
