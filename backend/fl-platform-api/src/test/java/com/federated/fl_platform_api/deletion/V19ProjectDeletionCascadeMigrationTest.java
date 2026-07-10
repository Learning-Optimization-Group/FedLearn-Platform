package com.federated.fl_platform_api.deletion;

import com.federated.fl_platform_api.repository.ProjectRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.sql.Timestamp;
import java.time.Instant;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * BA-14 regression: approving a project-deletion request (or the direct admin
 * {@code DELETE /api/projects/{id}}) must succeed even for a project that has
 * ever been <em>started</em> — i.e. one that has a {@code runs} row.
 *
 * <p>Before this fix, {@code runs.project_id -> projects(id)} and
 * {@code run_enrollments.run_id -> runs(id)} had NO {@code ON DELETE} action, so
 * the bare {@code DELETE FROM projects WHERE id=?} that
 * {@code ProjectService.deleteProject} (via {@code projectRepository.deleteById})
 * emits raised Postgres SQLSTATE 23503 (violates {@code runs_project_id_fkey}) —
 * surfaced to the admin as an opaque 409. V19 adds {@code ON DELETE CASCADE} to
 * those two FKs so the whole run sub-tree disappears with the project.
 *
 * <p>Why this test runs Flyway (not the default {@code test} profile): the JPA
 * entities map {@code Run.projectId} / {@code RunEnrollment.runId} as plain
 * {@code @Column} UUIDs (not {@code @ManyToOne}), so Hibernate's create-drop test
 * schema generates NO FK for them and cannot reproduce the bug. It only exists in
 * the real Flyway schema, so — like the other {@code V*MigrationTest} classes —
 * we run every migration against real Postgres (Testcontainers) with
 * {@code flyway.enabled=true} and {@code ddl-auto=validate}.
 *
 * <p>Registry boundary (BA-11 Chunk C): the content-addressed registry is
 * deliberately NOT cascaded. {@code model_artifacts.project_id/run_id} stay
 * {@code ON DELETE SET NULL} (a provenance row outlives its producer) and
 * {@code artifact_blobs} are shared/deduplicated across projects, so a blob
 * referenced by another project MUST survive this project's deletion. Refcount-
 * safe blob garbage-collection is the separate deferred BA-11 Chunk C item.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v19_deletion",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V19ProjectDeletionCascadeMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");
    private static final String SHARED_BLOB_SHA256 =
            "a".repeat(64); // 64 hex chars — satisfies chk_artifact_blobs_sha256_hex

    @Autowired
    JdbcTemplate jdbc;

    @Autowired
    ProjectRepository projectRepository;

    @Test
    void deletingStartedProject_cascadesRunSubtreeAndOwnedChildren_butPreservesSharedRegistryBlob() {
        Timestamp now = Timestamp.from(Instant.now());

        // ── Owner user ────────────────────────────────────────────────────────
        long userId = 4014L;
        jdbc.update("INSERT INTO users (id, username, email, password, created_at, updated_at, platform_role) " +
                        "VALUES (?, ?, ?, ?, ?, ?, 'USER')",
                userId, "ba14-owner", "ba14-owner@example.com", "x", now, now);

        // ── The started project (has a run) + a second project that shares the blob ──
        UUID startedProjectId = UUID.randomUUID();
        UUID otherProjectId = UUID.randomUUID();
        insertProject(startedProjectId, "ba14-started");
        insertProject(otherProjectId, "ba14-other");

        // ── Run sub-tree under the started project ────────────────────────────
        UUID runId = UUID.randomUUID();
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, " +
                        "status, recipe_key, created_at) VALUES (?, ?, 'FED_AVG', 3, 1, 1, 'COMPLETED', 'CNN', ?)",
                runId, startedProjectId, now);
        jdbc.update("INSERT INTO run_enrollments (run_id, user_id, partition_id, client_kind, enrolled_at) " +
                        "VALUES (?, ?, 0, 'SHARD', ?)",
                runId, userId, now);

        // ── An already-cascading owned child (proves the existing CASCADE FKs still work) ──
        jdbc.update("INSERT INTO project_memberships (project_id, user_id, role, joined_via, added_at) " +
                        "VALUES (?, ?, 'CLIENT', 'PUBLIC_JOIN', ?)",
                startedProjectId, userId, now);
        UUID roundResultId = UUID.randomUUID();
        jdbc.update("INSERT INTO round_result (id, project_id, server_round, loss, accuracy) VALUES (?, ?, 1, 0.5, 0.8)",
                roundResultId, startedProjectId);

        // ── Registry: one shared blob, referenced by BOTH projects' provenance rows ──
        jdbc.update("INSERT INTO artifact_blobs (sha256, size_bytes, backend, created_at) VALUES (?, 128, 'LOCAL_FS', ?)",
                SHARED_BLOB_SHA256, now);
        UUID artifactOfStarted = UUID.randomUUID();
        UUID artifactOfOther = UUID.randomUUID();
        jdbc.update("INSERT INTO model_artifacts (id, org_id, blob_sha256, kind, project_id, run_id, created_at) " +
                        "VALUES (?, ?, ?, 'LORA_ADAPTER', ?, ?, ?)",
                artifactOfStarted, DEFAULT_ORG, SHARED_BLOB_SHA256, startedProjectId, runId, now);
        jdbc.update("INSERT INTO model_artifacts (id, org_id, blob_sha256, kind, project_id, run_id, created_at) " +
                        "VALUES (?, ?, ?, 'LORA_ADAPTER', ?, NULL, ?)",
                artifactOfOther, DEFAULT_ORG, SHARED_BLOB_SHA256, otherProjectId, now);

        // ── Precondition: the run sub-tree exists and is what blocks deletion today ──
        assertThat(count("runs", "project_id", startedProjectId)).isEqualTo(1);
        assertThat(count("run_enrollments", "run_id", runId)).isEqualTo(1);

        // ── Act: the exact call ProjectService.deleteProject makes (line 486). Both admin
        //    deletion paths (deletion-request approval AND direct DELETE) funnel through it. ──
        projectRepository.deleteById(startedProjectId);
        projectRepository.flush();

        // ── The project and everything it OWNS is gone (cascade) ──────────────
        assertThat(count("projects", "id", startedProjectId)).as("project deleted").isEqualTo(0);
        assertThat(count("runs", "id", runId)).as("run cascaded").isEqualTo(0);
        assertThat(count("run_enrollments", "run_id", runId)).as("enrollment cascaded").isEqualTo(0);
        assertThat(count("project_memberships", "project_id", startedProjectId)).as("membership cascaded").isEqualTo(0);
        assertThat(count("round_result", "id", roundResultId)).as("round_result cascaded").isEqualTo(0);

        // ── Registry: the started project's provenance row survives with project_id nulled ──
        assertThat(count("model_artifacts", "id", artifactOfStarted)).as("artifact survives").isEqualTo(1);
        Integer nulledProject = jdbc.queryForObject(
                "SELECT COUNT(*) FROM model_artifacts WHERE id = ? AND project_id IS NULL",
                Integer.class, artifactOfStarted);
        assertThat(nulledProject).as("artifact.project_id SET NULL").isEqualTo(1);

        // ── The shared blob is NOT garbage-collected (BA-11 Chunk C boundary) and the other
        //    project + its provenance row still reference it. ──
        assertThat(blobCount(SHARED_BLOB_SHA256)).as("shared blob preserved").isEqualTo(1);
        assertThat(count("projects", "id", otherProjectId)).as("other project intact").isEqualTo(1);
        Integer otherStillRefsBlob = jdbc.queryForObject(
                "SELECT COUNT(*) FROM model_artifacts WHERE id = ? AND blob_sha256 = ?",
                Integer.class, artifactOfOther, SHARED_BLOB_SHA256);
        assertThat(otherStillRefsBlob).as("other project still references the shared blob").isEqualTo(1);
    }

    private void insertProject(UUID id, String name) {
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, status, org_id) " +
                        "VALUES (?, ?, 'CNN', 'ba14', 'COMPLETED', ?)",
                id, name, DEFAULT_ORG);
    }

    private int count(String table, String column, Object value) {
        Integer n = jdbc.queryForObject(
                "SELECT COUNT(*) FROM " + table + " WHERE " + column + " = ?", Integer.class, value);
        return n == null ? 0 : n;
    }

    private int blobCount(String sha256) {
        Integer n = jdbc.queryForObject(
                "SELECT COUNT(*) FROM artifact_blobs WHERE sha256 = ?", Integer.class, sha256);
        return n == null ? 0 : n;
    }
}
