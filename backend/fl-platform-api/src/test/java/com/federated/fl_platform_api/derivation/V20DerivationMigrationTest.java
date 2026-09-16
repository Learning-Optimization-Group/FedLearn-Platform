package com.federated.fl_platform_api.derivation;

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
 * Validates the V20 project-derivation migration end-to-end, exactly like {@code V17MigrationTest}:
 * runs every migration through V20 against a real PostgreSQL (Testcontainers) with
 * {@code ddl-auto=validate}, so the Spring context only loads if the {@code Project} entity matches
 * the Flyway-created schema.
 *
 * <p>DA-14 Ph3.2: the derivation record is opt-in and nullable — a project with a NULL derivation
 * (every legacy row, and any create request that omits it) behaves exactly as a from-scratch recipe
 * project. This asserts the columns exist, the flag is NOT NULL default false, the refs are
 * nullable, and the sha256 content-address CHECK admits NULL but rejects a malformed address.</p>
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v20_derivation",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V20DerivationMigrationTest {

    private static final Path V20 = Path.of("src/main/resources/db/migration/V20__project_derivation.sql");

    @Autowired
    JdbcTemplate jdbc;

    /** If this loads, Flyway V1→V20 ran and Hibernate validated the Project entity against it. */
    @Test
    void context_loads_with_flyway_and_validate() {
        assertThat(jdbc).isNotNull();
    }

    @Test
    void derivation_columns_exist_with_expected_types() {
        assertThat(columnType("projects", "init_from_pretrained")).isEqualTo("boolean");
        assertThat(columnType("projects", "base_ref_sha256")).isEqualTo("character varying");
        assertThat(columnType("projects", "derivation_spec")).isEqualTo("text");
    }

    @Test
    void flag_is_not_null_defaulting_false_and_refs_are_nullable() {
        assertThat(isNullable("projects", "init_from_pretrained")).isEqualTo("NO");
        assertThat(columnDefault("projects", "init_from_pretrained")).containsIgnoringCase("false");
        // A from-scratch project carries no base ref / derivation spec.
        assertThat(isNullable("projects", "base_ref_sha256")).isEqualTo("YES");
        assertThat(isNullable("projects", "derivation_spec")).isEqualTo("YES");
    }

    @Test
    void base_ref_sha256_has_a_hex_check_that_admits_null() {
        // Assert the CHECK exists and has the intended definition (admits NULL, requires 64-hex),
        // via pg_get_constraintdef — no INSERT, so this can't be a false failure over an unrelated
        // NOT NULL column. Postgres enforces the regex/NULL semantics from this clause.
        String clause = jdbc.queryForObject(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conname = ?",
                String.class, "chk_projects_base_ref_sha256_hex");
        assertThat(clause).contains("base_ref_sha256 IS NULL");
        assertThat(clause).contains("[0-9a-f]{64}");
    }

    @Test
    void v20_alters_the_projects_table_only() throws Exception {
        assertThat(Files.exists(V20)).isTrue();
        String sql = Files.readString(V20).toUpperCase();
        assertThat(sql).contains("ALTER TABLE PROJECTS");
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
