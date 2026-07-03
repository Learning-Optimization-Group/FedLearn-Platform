package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.model.Project;
import com.federated.fl_platform_api.model.ProjectStatus;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.RunRepository;
import org.junit.jupiter.api.Test;

import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

/**
 * BA-4: project status is derived from the active run. Plain Mockito (no Spring context), so this
 * never spins up the shared Testcontainers DB.
 */
class ProjectStatusServiceTest {

    private final RunRepository runRepository = mock(RunRepository.class);
    private final ProjectStatusService service = new ProjectStatusService(runRepository);

    private Project projectWithActiveRun(UUID runId) {
        Project p = new Project();
        p.setActiveRunId(runId);
        return p;
    }

    private Run runWith(RunStatus status) {
        Run r = new Run();
        r.setStatus(status);
        return r;
    }

    @Test
    void noActiveRun_isCreated() {
        assertEquals(ProjectStatus.CREATED, service.currentStatus(new Project()));
        verifyNoInteractions(runRepository);           // no lookup when there is nothing to look up
    }

    @Test
    void activeRunRunning_isRunning() {
        UUID rid = UUID.randomUUID();
        when(runRepository.findById(rid)).thenReturn(Optional.of(runWith(RunStatus.RUNNING)));
        assertEquals(ProjectStatus.RUNNING, service.currentStatus(projectWithActiveRun(rid)));
    }

    @Test
    void activeRunFailed_isFailed() {
        // The bug this fixes: today a failed run leaves projects.status stuck at RUNNING.
        UUID rid = UUID.randomUUID();
        when(runRepository.findById(rid)).thenReturn(Optional.of(runWith(RunStatus.FAILED)));
        assertEquals(ProjectStatus.FAILED, service.currentStatus(projectWithActiveRun(rid)));
    }

    @Test
    void activeRunIdButRunDeleted_isCreated() {
        // active_run_id is ON DELETE SET NULL, but guard the race where the row is already gone.
        UUID rid = UUID.randomUUID();
        when(runRepository.findById(rid)).thenReturn(Optional.empty());
        assertEquals(ProjectStatus.CREATED, service.currentStatus(projectWithActiveRun(rid)));
    }
}
