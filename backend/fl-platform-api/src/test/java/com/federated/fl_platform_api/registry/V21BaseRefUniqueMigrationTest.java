package com.federated.fl_platform_api.registry;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.sql.Timestamp;
import java.time.Instant;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Validates V21's partial unique index {@code uq_base_ref_org_model} against real PostgreSQL
 * (Testcontainers, Flyway on, {@code ddl-auto=validate}) — the dev/prod path the {@code test} profile
 * (create-drop, Flyway off) never exercises, and where a Postgres PARTIAL index (a {@code WHERE} clause
 * Hibernate/JPA cannot express) can even exist. Pins the DB-level guarantee behind the DA-3
 * find-or-create race fix: exactly ONE BASE_REF per (org_id, base_model_ref), while adapters/checkpoints
 * over the same base are unconstrained.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v21",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V21BaseRefUniqueMigrationTest {

    @Autowired
    JdbcTemplate jdbc;

    private static final String SHA = "a".repeat(64);   // valid per chk_artifact_blobs_sha256_hex

    private UUID org;

    private void seedBlobAndOrg() {
        // model_artifacts.blob_sha256 -> artifact_blobs.sha256 (FK) and org_id -> organizations(id).
        jdbc.update("INSERT INTO artifact_blobs (sha256, size_bytes, backend, created_at) "
                + "VALUES (?, ?, 'LOCAL_FS', ?) ON CONFLICT DO NOTHING", SHA, 4L, Timestamp.from(Instant.now()));
        org = UUID.randomUUID();
        Timestamp now = Timestamp.from(Instant.now());
        jdbc.update("INSERT INTO organizations (id, name, slug, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                org, "org-" + org, "slug-" + org, now, now);
    }

    private void insertArtifact(UUID id, String kind, String baseRef) {
        jdbc.update("INSERT INTO model_artifacts "
                        + "(id, org_id, blob_sha256, kind, base_model_ref, created_at, published) "
                        + "VALUES (?, ?, ?, ?, ?, ?, false)",
                id, org, SHA, kind, baseRef, Timestamp.from(Instant.now()));
    }

    @Test
    void the_partial_unique_index_exists_after_migration() {
        Integer n = jdbc.queryForObject(
                "SELECT count(*) FROM pg_indexes WHERE indexname = 'uq_base_ref_org_model'", Integer.class);
        assertThat(n).isEqualTo(1);
    }

    @Test
    void a_second_base_ref_for_the_same_org_and_base_is_rejected() {
        seedBlobAndOrg();
        insertArtifact(UUID.randomUUID(), "BASE_REF", "Qwen/Qwen2.5-0.5B");
        // Same (org_id, base_model_ref) among BASE_REF -> the one-BASE_REF invariant blocks the duplicate.
        assertThatThrownBy(() -> insertArtifact(UUID.randomUUID(), "BASE_REF", "Qwen/Qwen2.5-0.5B"))
                .isInstanceOf(DataAccessException.class);
    }

    @Test
    void adapters_and_a_different_base_over_the_same_org_are_unconstrained() {
        seedBlobAndOrg();
        insertArtifact(UUID.randomUUID(), "BASE_REF", "Qwen/Qwen2.5-0.5B");
        // A LORA_ADAPTER over the SAME base is fine (the index is partial on kind='BASE_REF')...
        assertThatCode(() -> insertArtifact(UUID.randomUUID(), "LORA_ADAPTER", "Qwen/Qwen2.5-0.5B"))
                .doesNotThrowAnyException();
        // ...and a BASE_REF for a DIFFERENT base in the same org is a distinct, allowed row.
        assertThatCode(() -> insertArtifact(UUID.randomUUID(), "BASE_REF", "meta-llama/Llama-3.2-1B"))
                .doesNotThrowAnyException();
    }
}
