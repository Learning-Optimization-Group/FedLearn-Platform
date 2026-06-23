package com.federated.fl_platform_api.run;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Validates the V8 run-lifecycle migration.
 *
 * <p>Runs every migration through V8 against a real PostgreSQL instance
 * (Testcontainers, {@code jdbc:tc:}). Asserts:
 * <ul>
 *   <li>{@code runs} table is created with all required columns.</li>
 *   <li>{@code run_enrollments} table is created with partition-uniqueness enforced.</li>
 *   <li>{@code projects.active_run_id} column is added.</li>
 * </ul>
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v8",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=none",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V8MigrationTest {

    @Autowired
    JdbcTemplate jdbc;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    private void assertColumnExists(String table, String column) {
        Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM information_schema.columns " +
                        "WHERE table_name = ? AND column_name = ?",
                Integer.class, table, column);
        assertThat(count)
                .as("Column %s.%s must exist after V8 migration", table, column)
                .isEqualTo(1);
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    @Test
    void runsTableExists_withClientsPerRoundAndStatus() {
        assertColumnExists("runs", "clients_per_round");
        assertColumnExists("runs", "partitioning_mode");
        assertColumnExists("runs", "status");
        assertColumnExists("runs", "seed");
        assertColumnExists("runs", "recipe_key");
    }

    @Test
    void runEnrollmentsTableExists_withUniquePartition() {
        assertColumnExists("run_enrollments", "partition_id");
        assertColumnExists("run_enrollments", "client_kind");

        // Assert the uq_run_partition unique constraint exists in the catalog.
        Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM information_schema.table_constraints " +
                        "WHERE constraint_name = ?",
                Integer.class, "uq_run_partition");
        assertThat(count)
                .as("Unique constraint uq_run_partition must exist after V8 migration")
                .isEqualTo(1);
    }

    @Test
    void projectsHasActiveRunIdColumn() {
        assertColumnExists("projects", "active_run_id");
    }
}
