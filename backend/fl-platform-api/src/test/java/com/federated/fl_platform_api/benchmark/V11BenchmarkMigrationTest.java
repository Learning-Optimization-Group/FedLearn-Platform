package com.federated.fl_platform_api.benchmark;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Validates the V11 benchmarking migration end-to-end.
 *
 * <p>Runs every migration through V11 against a real PostgreSQL (Testcontainers,
 * {@code jdbc:tc:}) with {@code ddl-auto=validate} — so the Spring context only
 * loads if EVERY JPA entity (including {@code BenchmarkRound} / {@code BenchmarkRun})
 * matches the Flyway-created schema. This is the exact dev/prod path the
 * {@code test} profile (create-drop, Flyway off) does not exercise.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v11",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V11BenchmarkMigrationTest {

    private static final Path V11 = Path.of(
            "src/main/resources/db/migration/V11__benchmarks.sql");

    @Autowired
    JdbcTemplate jdbc;

    /** If this loads, Flyway V1→V11 ran and Hibernate validated all entities against it. */
    @Test
    void context_loads_with_flyway_and_validate() {
        assertThat(jdbc).isNotNull();
    }

    @Test
    void benchmark_tables_and_key_columns_exist() {
        assertThat(tableExists("benchmark_rounds")).as("benchmark_rounds table").isTrue();
        assertThat(tableExists("benchmark_runs")).as("benchmark_runs table").isTrue();

        // Spot-check the research-driven additions are really columns (not just
        // in the entity): calibration (ece/brier) and time-to-target-accuracy.
        assertThat(columnType("benchmark_rounds", "ece")).isEqualTo("double precision");
        assertThat(columnType("benchmark_rounds", "brier")).isEqualTo("double precision");
        assertThat(columnType("benchmark_rounds", "per_class_json")).isEqualTo("text");
        assertThat(columnType("benchmark_runs", "rounds_to_target")).isEqualTo("integer");
        assertThat(columnType("benchmark_runs", "ms_to_target")).isEqualTo("bigint");
    }

    @Test
    void v11_file_declares_both_tables_and_constraints() throws Exception {
        assertThat(Files.exists(V11)).as("V11 migration file present").isTrue();
        String sql = Files.readString(V11).toUpperCase();
        assertThat(sql).contains("CREATE TABLE BENCHMARK_ROUNDS");
        assertThat(sql).contains("CREATE TABLE BENCHMARK_RUNS");
        assertThat(sql).contains("UQ_BENCHMARK_ROUND");
        assertThat(sql).contains("ON DELETE CASCADE");
    }

    private boolean tableExists(String table) {
        Integer n = jdbc.queryForObject(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                Integer.class, table);
        return n != null && n > 0;
    }

    private String columnType(String table, String column) {
        return jdbc.queryForObject(
                "SELECT data_type FROM information_schema.columns " +
                        "WHERE table_name = ? AND column_name = ?",
                String.class, table, column);
    }
}
