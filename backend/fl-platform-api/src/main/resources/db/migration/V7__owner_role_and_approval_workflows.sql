-- =====================================================================
-- V7: Owner role + approval workflows
--   1. Add PROJECT_OWNER to the platform_role CHECK constraint.
--   2. Owner-promotion requests (USER -> PROJECT_OWNER, admin-approved).
--   3. Project-deletion requests (owner-requested, admin-approved).
--
-- The 3rd visibility tier (RESTRICTED) needs NO schema change: projects.visibility
-- is a plain VARCHAR(32) with no CHECK constraint (see V4), so the value is owned
-- entirely by the ProjectVisibility enum.
--
-- PostgreSQL (all profiles). The `test` profile disables Flyway and builds the
-- schema from JPA entities (create-drop on Testcontainers Postgres), so this
-- file runs under the dev/prod Flyway path and the dedicated V*MigrationTest classes.
-- =====================================================================

-- 1) Widen the platform_role domain. V6 created chk_users_platform_role with the
--    two-value set; drop and re-add it including PROJECT_OWNER. IF EXISTS guards
--    against dev DBs baselined past V6.
ALTER TABLE users DROP CONSTRAINT IF EXISTS chk_users_platform_role;
ALTER TABLE users ADD CONSTRAINT chk_users_platform_role
    CHECK (platform_role IN ('USER','PROJECT_OWNER','PLATFORM_ADMIN'));

-- 2) Owner-promotion requests. One row per user; a re-request after a DENY
--    updates the same row (UNIQUE(user_id)), mirroring project_access_requests.
CREATE TABLE owner_promotion_requests (
    id           BIGSERIAL PRIMARY KEY,
    user_id      BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status       VARCHAR(32)  NOT NULL,    -- 'PENDING' | 'APPROVED' | 'DENIED'
    message      TEXT,
    requested_at TIMESTAMP WITH TIME ZONE NOT NULL,
    decided_at   TIMESTAMP WITH TIME ZONE,
    decided_by   BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE (user_id)
);
CREATE INDEX idx_opr_status ON owner_promotion_requests(status);

-- 3) Project-deletion requests. One row per project (UNIQUE(project_id)); a
--    re-request after a DENY updates it. ON DELETE CASCADE means an APPROVED
--    request's row is cleaned up automatically when the project is hard-deleted.
CREATE TABLE project_deletion_requests (
    id           BIGSERIAL PRIMARY KEY,
    project_id   UUID         NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    requested_by BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status       VARCHAR(32)  NOT NULL,    -- 'PENDING' | 'APPROVED' | 'DENIED'
    reason       TEXT,
    requested_at TIMESTAMP WITH TIME ZONE NOT NULL,
    decided_at   TIMESTAMP WITH TIME ZONE,
    decided_by   BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE (project_id)
);
CREATE INDEX idx_pdr_status ON project_deletion_requests(status);
