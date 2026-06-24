package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import com.federated.fl_platform_api.service.AuthorizationService;
import com.federated.fl_platform_api.security.OrgScope;
import com.federated.fl_platform_api.service.RunService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RunServiceTest {

    @Mock RunRepository runRepository;
    @Mock RunEnrollmentRepository enrollmentRepository;
    @Mock ProjectRepository projectRepository;
    @Mock ProjectMembershipRepository membershipRepository;
    @Mock AuthorizationService authz;
    @Mock OrgScope orgScope;
    @Mock ConnectionTokenService tokenService;

    @InjectMocks RunService runService;

    @BeforeEach
    void injectValues() {
        ReflectionTestUtils.setField(runService, "grpcHost", "localhost");
    }

    private Project project(UUID id) {
        Project p = new Project();
        p.setId(id);
        p.setModelType("CNN");
        return p;
    }

    @Test
    void createForStart_buildsStartingRunWithSeedAndRecipe() {
        UUID pid = UUID.randomUUID();
        Project p = project(pid);
        when(runRepository.save(any(Run.class))).thenAnswer(inv -> {
            Run r = inv.getArgument(0);
            if (r.getId() == null) r.setId(UUID.randomUUID());
            return r;
        });

        Run run = runService.createForStart(p, "FedAvg", 5, 2, 4);

        assertEquals(RunStatus.STARTING, run.getStatus());
        assertEquals(4, run.getClientsPerRound());
        assertEquals("CNN", run.getRecipeKey());
        assertEquals(PartitioningMode.SHARDED, run.getPartitioningMode());
        assertNotNull(run.getSeed());
        assertNotNull(run.getCreatedAt());
    }

    @Test
    void markRunning_setsRunningHostPortAndStartedAt() {
        UUID rid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid);
        r.setStatus(RunStatus.STARTING);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(runRepository.save(any(Run.class))).thenAnswer(i -> i.getArgument(0));

        runService.markRunning(rid, 50001);

        assertEquals(RunStatus.RUNNING, r.getStatus());
        assertEquals(50001, r.getServerPort());
        assertNotNull(r.getServerHost());
        assertNotNull(r.getStartedAt());
    }

    @Test
    void markFailed_setsFailedAndEndedAt() {
        UUID rid = UUID.randomUUID();
        Run r = new Run();
        r.setId(rid);
        r.setStatus(RunStatus.STARTING);
        when(runRepository.findById(rid)).thenReturn(java.util.Optional.of(r));
        when(runRepository.save(any(Run.class))).thenAnswer(i -> i.getArgument(0));

        runService.markFailed(rid);

        assertEquals(RunStatus.FAILED, r.getStatus());
        assertNotNull(r.getEndedAt());
    }
}
