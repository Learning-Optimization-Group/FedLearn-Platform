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

/** Preserve reported legacy provenance while enforcing identity for new run-bound records. */
@Testcontainers
class V35BenchmarkIdentityMigrationTest {
    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>("postgres:16.6-alpine");

    @Test
    void upgradeRetainsLegacyRowsAndAllowsDifferentRunsAtTheSameRound() {
        var source = new DriverManagerDataSource(POSTGRES.getJdbcUrl(), POSTGRES.getUsername(), POSTGRES.getPassword());
        Flyway.configure().dataSource(source).target("34").load().migrate();
        var jdbc = new JdbcTemplate(source);
        UUID project = UUID.randomUUID(), oldRun = UUID.randomUUID(), newRun = UUID.randomUUID(), legacy = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id,name,model_type,model_name,status,org_id,visibility) "
                + "VALUES (?,?,'CNN','net','RUNNING','00000000-0000-0000-0000-000000000001','PRIVATE')",
                project, "benchmark-migration-" + project);
        jdbc.update("INSERT INTO benchmark_rounds (id,project_id,run_id,server_round,loss,recorded_at) "
                + "VALUES (?,?,?,1,0.5,CURRENT_TIMESTAMP)", legacy, project, oldRun);
        jdbc.update("INSERT INTO benchmark_runs (id,project_id,run_id,rounds_completed) VALUES (?,?,?,1)",
                UUID.randomUUID(), project, oldRun);
        Flyway.configure().dataSource(source).load().migrate();
        var preserved = jdbc.queryForMap("SELECT run_id,loss,payload_sha256 FROM benchmark_rounds WHERE id=?", legacy);
        assertEquals(oldRun, preserved.get("run_id"));
        assertEquals(.5, preserved.get("loss"));
        assertNull(preserved.get("payload_sha256"));
        assertEquals(false, jdbc.queryForObject("SELECT run_identity_verified FROM benchmark_runs WHERE project_id=?", Boolean.class, project));
        jdbc.update("INSERT INTO benchmark_rounds (id,project_id,run_id,server_round,payload_sha256,recorded_at) "
                + "VALUES (?,?,?,1,?,CURRENT_TIMESTAMP)", UUID.randomUUID(), project, newRun, "a".repeat(64));
        assertThrows(DataAccessException.class, () -> jdbc.update(
                "INSERT INTO benchmark_rounds (id,project_id,run_id,server_round,payload_sha256,recorded_at) "
                        + "VALUES (?,?,?,1,?,CURRENT_TIMESTAMP)", UUID.randomUUID(), project, newRun, "b".repeat(64)));
        assertEquals(2, jdbc.queryForObject("SELECT count(*) FROM benchmark_rounds WHERE project_id=?", Integer.class, project));
        assertThrows(DataAccessException.class, () -> jdbc.update(
                "INSERT INTO benchmark_rounds (id,project_id,run_id,server_round,payload_sha256,recorded_at) "
                        + "VALUES (?,?,?,2,'not-a-digest',CURRENT_TIMESTAMP)", UUID.randomUUID(), project, newRun));
    }
}
