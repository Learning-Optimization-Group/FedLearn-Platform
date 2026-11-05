package com.federated.fl_platform_api;

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

/** Exercise the upgrade with ambiguous pre-existing project-only history. */
@Testcontainers
class V34RoundResultMigrationTest {
    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>("postgres:16.6-alpine");

    @Test
    void upgradePreservesLegacyRowsAndEnforcesNewRunRoundIdentity() {
        var source = new DriverManagerDataSource(POSTGRES.getJdbcUrl(), POSTGRES.getUsername(), POSTGRES.getPassword());
        Flyway.configure().dataSource(source).target("33").load().migrate();
        var jdbc = new JdbcTemplate(source);
        UUID project = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, status, org_id, visibility) "
                + "VALUES (?, ?, 'CNN', 'net', 'RUNNING', '00000000-0000-0000-0000-000000000001', 'PRIVATE')",
                project, "migration-" + project);
        for (int index = 0; index < 2; index++) {
            jdbc.update("INSERT INTO round_result (id, project_id, server_round, loss) VALUES (?, ?, 1, 0.5)",
                    UUID.randomUUID(), project);
        }
        Flyway.configure().dataSource(source).load().migrate();
        assertEquals(2, jdbc.queryForObject("SELECT count(*) FROM round_result WHERE run_id IS NULL", Integer.class));
        UUID run = UUID.randomUUID();
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, "
                + "status, recipe_key, created_at) VALUES (?, ?, 'FedAvg', 3, 1, 1, 'RUNNING', 'CNN', CURRENT_TIMESTAMP)",
                run, project);
        jdbc.update("INSERT INTO round_result (id, project_id, run_id, server_round) VALUES (?, ?, ?, 1)",
                UUID.randomUUID(), project, run);
        assertThrows(DataAccessException.class, () -> jdbc.update(
                "INSERT INTO round_result (id, project_id, run_id, server_round) VALUES (?, ?, ?, 1)",
                UUID.randomUUID(), project, run));
        assertThrows(DataAccessException.class, () -> jdbc.update(
                "INSERT INTO round_result (id, project_id, run_id, server_round) VALUES (?, ?, ?, 2)",
                UUID.randomUUID(), project, UUID.randomUUID()));
        jdbc.update("DELETE FROM runs WHERE id = ?", run);
        assertEquals(2, jdbc.queryForObject("SELECT count(*) FROM round_result", Integer.class));
    }
}
