-- =====================================================================
-- V5: Identity Foundations
-- Multi-tenant orgs, user lifecycle/profile columns, audit log
-- =====================================================================

-- 1) organizations
CREATE TABLE organizations (
    id            UUID PRIMARY KEY,
    name          VARCHAR(120) NOT NULL,
    slug          VARCHAR(64)  NOT NULL UNIQUE,
    created_at    TIMESTAMP    NOT NULL,
    updated_at    TIMESTAMP    NOT NULL,
    deleted_at    TIMESTAMP
);

-- 2) organization_memberships (composite PK; no surrogate)
CREATE TABLE organization_memberships (
    org_id     UUID         NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id    BIGINT       NOT NULL REFERENCES users(id)         ON DELETE CASCADE,
    org_role   VARCHAR(16)  NOT NULL CHECK (org_role IN ('OWNER','ADMIN','MEMBER')),
    created_at TIMESTAMP    NOT NULL,
    PRIMARY KEY (org_id, user_id)
);
CREATE INDEX idx_org_mem_user ON organization_memberships(user_id);

-- 3) users: rename role -> platform_role; extend with lifecycle + profile columns.
-- One ALTER per column: H2 (dev profile) rejects multi-clause ALTER TABLE.
ALTER TABLE users ALTER COLUMN role RENAME TO platform_role;
ALTER TABLE users ADD COLUMN status         VARCHAR(16) NOT NULL DEFAULT 'ACTIVE';
ALTER TABLE users ADD COLUMN deleted_at     TIMESTAMP;
ALTER TABLE users ADD COLUMN email_verified BOOLEAN     NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN display_name   VARCHAR(120);
ALTER TABLE users ADD COLUMN avatar_url     VARCHAR(512);
ALTER TABLE users ADD COLUMN last_login_at  TIMESTAMP;
ALTER TABLE users ADD CONSTRAINT chk_users_status
    CHECK (status IN ('PENDING','ACTIVE','SUSPENDED'));

-- 4) projects: pin to an org
ALTER TABLE projects ADD COLUMN org_id UUID REFERENCES organizations(id);
CREATE INDEX idx_projects_org ON projects(org_id);

-- 5) audit_events
CREATE TABLE audit_events (
    id              UUID PRIMARY KEY,
    occurred_at     TIMESTAMP    NOT NULL,
    actor_user_id   BIGINT       REFERENCES users(id),
    org_id          UUID         REFERENCES organizations(id),
    action          VARCHAR(64)  NOT NULL,
    target_type     VARCHAR(32),
    target_id       VARCHAR(64),
    metadata        CLOB,
    request_ip      VARCHAR(45),
    user_agent      VARCHAR(256)
);
CREATE INDEX idx_audit_org_time    ON audit_events(org_id, occurred_at);
CREATE INDEX idx_audit_actor_time  ON audit_events(actor_user_id, occurred_at);
CREATE INDEX idx_audit_action_time ON audit_events(action, occurred_at);

-- =====================================================================
-- Backfill: every existing user joins a "Default" org
-- =====================================================================

INSERT INTO organizations (id, name, slug, created_at, updated_at)
VALUES ('00000000-0000-0000-0000-000000000001', 'Default', 'default',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

INSERT INTO organization_memberships (org_id, user_id, org_role, created_at)
SELECT '00000000-0000-0000-0000-000000000001', id, 'MEMBER', CURRENT_TIMESTAMP
FROM users;

UPDATE organization_memberships
SET org_role = 'OWNER'
WHERE user_id IN (SELECT DISTINCT user_id FROM projects);

UPDATE projects SET org_id = '00000000-0000-0000-0000-000000000001'
WHERE org_id IS NULL;

ALTER TABLE projects ALTER COLUMN org_id SET NOT NULL;
