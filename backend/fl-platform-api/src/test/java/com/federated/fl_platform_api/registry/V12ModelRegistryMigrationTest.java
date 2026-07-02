package com.federated.fl_platform_api.registry;

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
 * Validates the V12 model-artifact-registry migration end-to-end, exactly like
 * {@code V11BenchmarkMigrationTest}: runs every migration through V12 against a real PostgreSQL
 * (Testcontainers) with {@code ddl-auto=validate}, so the Spring context only loads if
 * {@code ArtifactBlob}/{@code ModelArtifact}/{@code ArtifactLineage} all match the Flyway-created
 * schema — the dev/prod path the {@code test} profile (create-drop, Flyway off) does not exercise.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v12",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V12ModelRegistryMigrationTest {

    private static final Path V12 = Path.of("src/main/resources/db/migration/V12__model_artifact_registry.sql");

    @Autowired
    JdbcTemplate jdbc;

    /** If this loads, Flyway V1→V12 ran and Hibernate validated all three registry entities against it. */
    @Test
    void context_loads_with_flyway_and_validate() {
        assertThat(jdbc).isNotNull();
    }

    @Test
    void registry_tables_and_key_columns_exist() {
        assertThat(tableExists("artifact_blobs")).as("artifact_blobs").isTrue();
        assertThat(tableExists("model_artifacts")).as("model_artifacts").isTrue();
        assertThat(tableExists("artifact_lineage")).as("artifact_lineage").isTrue();

        assertThat(columnType("artifact_blobs", "sha256")).isEqualTo("character varying");
        assertThat(columnType("artifact_blobs", "size_bytes")).isEqualTo("bigint");
        assertThat(columnType("model_artifacts", "org_id")).isEqualTo("uuid");
        assertThat(columnType("model_artifacts", "eval_card_json")).isEqualTo("text");
        assertThat(columnType("model_artifacts", "created_at")).isEqualTo("timestamp with time zone");
        assertThat(columnType("artifact_lineage", "relationship")).isEqualTo("character varying");
    }

    @Test
    void v12_declares_split_tables_and_append_only_fks() throws Exception {
        assertThat(Files.exists(V12)).isTrue();
        String sql = Files.readString(V12).toUpperCase();
        assertThat(sql).contains("CREATE TABLE ARTIFACT_BLOBS");
        assertThat(sql).contains("CREATE TABLE MODEL_ARTIFACTS");
        assertThat(sql).contains("CREATE TABLE ARTIFACT_LINEAGE");
        // Provenance is preserved (run/project SET NULL), lineage never dangles (RESTRICT), and the
        // registry is append-only (no CASCADE anywhere).
        assertThat(sql).contains("ON DELETE SET NULL");
        assertThat(sql).contains("ON DELETE RESTRICT");
        assertThat(sql).doesNotContain("ON DELETE CASCADE");
    }

    private boolean tableExists(String table) {
        Integer n = jdbc.queryForObject(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                Integer.class, table);
        return n != null && n > 0;
    }

    private String columnType(String table, String column) {
        return jdbc.queryForObject(
                "SELECT data_type FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
                String.class, table, column);
    }
}
