package com.federated.fl_platform_api.model;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * BA-4: the single source of truth mapping a run's status to the project's derived status.
 * Pure function — no Spring context, so it never touches the shared Testcontainers DB.
 */
class ProjectStatusTest {

    @Test
    void noActiveRun_isCreated() {
        assertEquals(ProjectStatus.CREATED, ProjectStatus.fromActiveRunStatus(null));
    }

    @Test
    void runningRun_isRunning() {
        assertEquals(ProjectStatus.RUNNING, ProjectStatus.fromActiveRunStatus(RunStatus.RUNNING));
    }

    @Test
    void startingRun_isRunning() {
        // A run is created in STARTING and activeRunId is set immediately, so a starting project must
        // read as RUNNING (today's UX shows RUNNING the instant Start is clicked), NOT CREATED/idle.
        assertEquals(ProjectStatus.RUNNING, ProjectStatus.fromActiveRunStatus(RunStatus.STARTING));
    }

    @Test
    void pendingRun_isRunning() {
        // PENDING is currently dead (runs are created in STARTING) but map it defensively: a pending
        // run means the project is being spun up, i.e. active — never leave it as idle CREATED.
        assertEquals(ProjectStatus.RUNNING, ProjectStatus.fromActiveRunStatus(RunStatus.PENDING));
    }

    @Test
    void completedRun_isCompleted() {
        assertEquals(ProjectStatus.COMPLETED, ProjectStatus.fromActiveRunStatus(RunStatus.COMPLETED));
    }

    @Test
    void stoppedRun_isStopped() {
        assertEquals(ProjectStatus.STOPPED, ProjectStatus.fromActiveRunStatus(RunStatus.STOPPED));
    }

    @Test
    void failedRun_isFailed() {
        // The improvement: a failed run now surfaces as FAILED instead of leaving a stale RUNNING.
        assertEquals(ProjectStatus.FAILED, ProjectStatus.fromActiveRunStatus(RunStatus.FAILED));
    }

    @Test
    void everyRunStatusMapsToSomething() {
        // Guard: if a new RunStatus is added later, this forces an explicit mapping decision.
        for (RunStatus rs : RunStatus.values()) {
            assertEquals(ProjectStatus.class, ProjectStatus.fromActiveRunStatus(rs).getClass());
        }
    }
}
