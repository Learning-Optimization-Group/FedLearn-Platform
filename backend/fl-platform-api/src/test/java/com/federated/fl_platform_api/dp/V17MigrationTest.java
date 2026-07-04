package com.federated.fl_platform_api.dp;

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
 * Validates the V17 project-DP-policy migration end-to-end, exactly like
 * {@code V12ModelRegistryMigrationTest}: runs every migration through V17 against a real PostgreSQL
 * (Testcontainers) with {@code ddl-auto=validate}, so the Spring context only loads if the
 * {@code Project} entity matches the Flyway-created schema — the dev/prod path the {@code test}
 * profile (create-drop, Flyway off) does not exercise.
 *
 * <p>SE-11: {@code regulated} marks a HIPAA-class project that may not start a run without a
 * complete DP config; {@code dp_enabled} + the three nullable knobs (target epsilon, delta, clip
 * norm S) are handed to the FL server as {@code --dp-*} flags.</p>
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v17",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V17MigrationTest {

    private static final Path V17 = Path.of("src/main/resources/db/migration/V17__project_dp_policy.sql");

    @Autowired
    JdbcTemplate jdbc;

    /** If this loads, Flyway V1→V17 ran and Hibernate validated the Project entity against it. */
    @Test
    void context_loads_with_flyway_and_validate() {
        assertThat(jdbc).isNotNull();
    }

    @Test
    void dp_policy_columns_exist_with_expected_types() {
        assertThat(columnType("projects", "regulated")).isEqualTo("boolean");
        assertThat(columnType("projects", "dp_enabled")).isEqualTo("boolean");
        assertThat(columnType("projects", "dp_target_epsilon")).isEqualTo("double precision");
        assertThat(columnType("projects", "dp_delta")).isEqualTo("double precision");
        assertThat(columnType("projects", "dp_clip_norm")).isEqualTo("double precision");
    }

    @Test
    void flags_are_not_null_defaulting_false_and_knobs_are_nullable() {
        // The two policy flags backfill every legacy row to FALSE and stay NOT NULL so the start
        // gate never reads a three-valued boolean.
        assertThat(isNullable("projects", "regulated")).isEqualTo("NO");
        assertThat(isNullable("projects", "dp_enabled")).isEqualTo("NO");
        assertThat(columnDefault("projects", "regulated")).containsIgnoringCase("false");
        assertThat(columnDefault("projects", "dp_enabled")).containsIgnoringCase("false");
        // The three knobs are nullable by design: a non-DP project carries no config.
        assertThat(isNullable("projects", "dp_target_epsilon")).isEqualTo("YES");
        assertThat(isNullable("projects", "dp_delta")).isEqualTo("YES");
        assertThat(isNullable("projects", "dp_clip_norm")).isEqualTo("YES");
    }

    @Test
    void v17_alters_the_projects_table_only() throws Exception {
        assertThat(Files.exists(V17)).isTrue();
        String sql = Files.readString(V17).toUpperCase();
        assertThat(sql).contains("ALTER TABLE PROJECTS");
        // Completeness (epsilon > 0, delta in (0,1), clip norm > 0) is enforced in Java at creation
        // and at the run-start gate — matching the V14 convention (no CHECK constraints).
        assertThat(sql).doesNotContain("CREATE TABLE");
    }

    private String columnType(String table, String column) {
        return jdbc.queryForObject(
                "SELECT data_type FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
                String.class, table, column);
    }

    private String isNullable(String table, String column) {
        return jdbc.queryForObject(
                "SELECT is_nullable FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
                String.class, table, column);
    }

    private String columnDefault(String table, String column) {
        return jdbc.queryForObject(
                "SELECT column_default FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
                String.class, table, column);
    }
}
