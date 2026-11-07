package com.federated.fl_platform_api.run;

import java.util.UUID;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.junit.jupiter.api.Assertions.*;

@Testcontainers
class V36ExecutionAttemptMigrationTest {
    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>("postgres:16.6-alpine");

    @Test
    void upgradeAddsExecutionAttemptsTableAndCascadeDeletes() {
        var source = new DriverManagerDataSource(POSTGRES.getJdbcUrl(), POSTGRES.getUsername(), POSTGRES.getPassword());
        Flyway.configure().dataSource(source).target("35").load().migrate();
        var jdbc = new JdbcTemplate(source);
        UUID project = UUID.randomUUID(), runId = UUID.randomUUID(), attemptId1 = UUID.randomUUID(), attemptId2 = UUID.randomUUID();

        jdbc.update("INSERT INTO projects (id,name,model_type,model_name,status,org_id,visibility) "
                + "VALUES (?,?,'CNN','net','RUNNING','00000000-0000-0000-0000-000000000001','PRIVATE')",
                project, "attempt-migration-" + project);
        jdbc.update("INSERT INTO runs (id,project_id,strategy,num_rounds,min_clients,clients_per_round,status,recipe_key,secure_aggregation,created_at) "
                + "VALUES (?,?,'FedAvg',5,1,1,'RUNNING','CNN',false,CURRENT_TIMESTAMP)", runId, project);

        // Run V36 migration
        Flyway.configure().dataSource(source).load().migrate();

        // Check columns on runs table
        Integer count = jdbc.queryForObject("SELECT attempt_count FROM runs WHERE id = ?", Integer.class, runId);
        assertEquals(1, count);
        UUID currentAttempt = jdbc.queryForObject("SELECT current_attempt_id FROM runs WHERE id = ?", UUID.class, runId);
        assertNull(currentAttempt);

        // Insert attempt 1
        jdbc.update("INSERT INTO run_execution_attempts (id,run_id,attempt_number,status,server_pid,server_port,created_at) "
                + "VALUES (?,?,1,'RUNNING',12345,50001,CURRENT_TIMESTAMP)", attemptId1, runId);
        jdbc.update("UPDATE runs SET current_attempt_id = ? WHERE id = ?", attemptId1, runId);

        // Attempt duplicate attempt_number 1 -> must fail unique constraint
        assertThrows(DataAccessException.class, () -> jdbc.update(
                "INSERT INTO run_execution_attempts (id,run_id,attempt_number,status,created_at) "
                        + "VALUES (?,?,1,'FAILED',CURRENT_TIMESTAMP)", attemptId2, runId));

        // Insert attempt 2 -> succeeds
        jdbc.update("INSERT INTO run_execution_attempts (id,run_id,attempt_number,status,created_at) "
                + "VALUES (?,?,2,'STARTING',CURRENT_TIMESTAMP)", attemptId2, runId);
        assertEquals(2, jdbc.queryForObject("SELECT count(*) FROM run_execution_attempts WHERE run_id = ?", Integer.class, runId));

        // Cascade delete on run removes its attempts
        jdbc.update("DELETE FROM runs WHERE id = ?", runId);
        assertEquals(0, jdbc.queryForObject("SELECT count(*) FROM run_execution_attempts WHERE run_id = ?", Integer.class, runId));
    }
}
