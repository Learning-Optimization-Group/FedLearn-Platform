-- V4: Per-project membership, visibility, and Model Hub columns.
-- See docs/superpowers/specs/2026-05-12-rbac-and-model-hub-design.md (§4).

-- Visibility. Default PRIVATE so existing rows do not accidentally become
-- world-readable after the migration.
ALTER TABLE projects
    ADD COLUMN visibility VARCHAR(16) NOT NULL DEFAULT 'PRIVATE';
CREATE INDEX idx_projects_visibility ON projects(visibility);

-- Model Hub columns. model_published_at is the first time the owner flipped
-- the publish toggle; null means the model has never been published.
-- One ALTER per column: H2 (dev profile) rejects multi-clause ALTER TABLE.
ALTER TABLE projects ADD COLUMN model_published    BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE projects ADD COLUMN model_description  TEXT;
ALTER TABLE projects ADD COLUMN model_tags         VARCHAR(512);
ALTER TABLE projects ADD COLUMN model_published_at TIMESTAMP WITH TIME ZONE;
CREATE INDEX idx_projects_model_published ON projects(model_published);

-- Per-project membership. One row per (project, user); role discriminates.
-- partition_id is null until the user first connects via gRPC.
-- role='OWNER' rows are inserted lazily for owners on first self-connect
-- and exist solely to hold the owner's partition_id; permission logic
-- never reads them (owner is determined by projects.user_id).
CREATE TABLE project_memberships (
    project_id     UUID         NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id        BIGINT       NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    role           VARCHAR(16)  NOT NULL,    -- 'MEMBER' | 'CLIENT' | 'OWNER'
    partition_id   INTEGER,
    joined_via     VARCHAR(16)  NOT NULL,    -- 'OWNER_ADD' | 'PUBLIC_JOIN' | 'REQUEST_APPROVED' | 'OWNER_SELF'
    added_by       BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    added_at       TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (project_id, user_id)
);
CREATE INDEX idx_project_memberships_user_id ON project_memberships(user_id);
CREATE INDEX idx_project_memberships_role    ON project_memberships(project_id, role);

-- Pending / decided access requests. UNIQUE(project_id, user_id) means a
-- re-request after a DENY updates the same row rather than appending.
CREATE TABLE project_access_requests (
    id              BIGSERIAL PRIMARY KEY,
    project_id      UUID         NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id         BIGINT       NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    requested_role  VARCHAR(16)  NOT NULL,    -- always 'CLIENT' in v1
    status          VARCHAR(16)  NOT NULL,    -- 'PENDING' | 'APPROVED' | 'DENIED'
    message         TEXT,
    requested_at    TIMESTAMP WITH TIME ZONE NOT NULL,
    decided_at      TIMESTAMP WITH TIME ZONE,
    decided_by      BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE (project_id, user_id)
);
CREATE INDEX idx_par_project_status ON project_access_requests(project_id, status);
CREATE INDEX idx_par_user_id        ON project_access_requests(user_id);
