-- V19 (BA-14): make a started project deletable.
--
-- Bug: approving a project-deletion request (ProjectDeletionService.decide) — and the direct admin
-- DELETE /api/projects/{id} — both funnel through ProjectService.deleteProject, which issues a bare
-- `DELETE FROM projects WHERE id=?` (projectRepository.deleteById). Every project-owned child FK
-- already carries ON DELETE CASCADE (round_result V1, server_logs V3, project_memberships /
-- project_access_requests V4, project_deletion_requests V7, benchmark_rounds / benchmark_runs V11)
-- EXCEPT the run sub-tree: V8 created `runs.project_id -> projects` and `run_enrollments.run_id ->
-- runs` with no ON DELETE action. So any project that was ever Started (has a `runs` row) could not
-- be deleted — Postgres raised SQLSTATE 23503 on `runs_project_id_fkey`, surfaced as an opaque 409.
--
-- Fix: add ON DELETE CASCADE to exactly those two FKs so the run + its enrollments disappear with
-- the project. The constraint names are the ones Postgres auto-assigned to V8's inline unnamed FKs
-- ({table}_{column}_fkey); IF EXISTS keeps this safe on any DB baselined slightly differently.
--
-- Deliberately NOT changed — the content-addressed registry stays append-only (V12 design, and the
-- SEPARATE deferred BA-11 Chunk C refcount-safe blob GC):
--   * model_artifacts.project_id / run_id  -> ON DELETE SET NULL  (a provenance row outlives its
--       producer; the row survives with project_id/run_id nulled, it is never deleted here).
--   * artifact_blobs                        -> untouched           (globally deduplicated / shared
--       across orgs and projects; a blob referenced by another project MUST survive, so blobs are
--       never garbage-collected on project deletion — that is BA-11 Chunk C).
--   * artifact_lineage child_id / parent_id -> ON DELETE RESTRICT  (edges never dangle; safe because
--       artifacts are SET NULL, never deleted, so RESTRICT cannot fire on project deletion).
--
-- PostgreSQL (dev/ec2demo/production Flyway path). The `test` profile builds the schema from JPA
-- entities (create-drop, Flyway off); Run.projectId / RunEnrollment.runId are mapped as plain
-- @Column UUIDs (not @ManyToOne), so that schema never had these FKs to begin with — this cascade
-- is exercised by the Flyway-on V19ProjectDeletionCascadeMigrationTest.

ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_project_id_fkey;
ALTER TABLE runs ADD CONSTRAINT runs_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE;

ALTER TABLE run_enrollments DROP CONSTRAINT IF EXISTS run_enrollments_run_id_fkey;
ALTER TABLE run_enrollments ADD CONSTRAINT run_enrollments_run_id_fkey
    FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE;
