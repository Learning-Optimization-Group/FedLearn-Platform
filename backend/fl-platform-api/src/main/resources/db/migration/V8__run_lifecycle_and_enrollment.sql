-- Phase 1 of seamless client onboarding. Introduces the Run aggregate (one
-- training execution of a project) as the source of truth for live FL-server
-- state, plus per-run client enrollment with run-scoped partition assignment.
-- Additive: projects.status/server_port stay as a mirror of the active run so
-- existing readers keep working.

CREATE TABLE runs (
    id                  UUID PRIMARY KEY,
    project_id          UUID NOT NULL REFERENCES projects(id),
    strategy            VARCHAR(32) NOT NULL,
    num_rounds          INTEGER NOT NULL,
    min_clients         INTEGER NOT NULL,
    clients_per_round   INTEGER NOT NULL,
    partitioning_mode   VARCHAR(16) NOT NULL DEFAULT 'SHARDED',
    status              VARCHAR(16) NOT NULL,
    server_host         VARCHAR(255),
    server_port         INTEGER,
    grpc_ca_fingerprint VARCHAR(128),
    seed                BIGINT,
    torch_version       VARCHAR(32),
    recipe_key          VARCHAR(64) NOT NULL,
    created_by          BIGINT REFERENCES users(id),
    created_at          TIMESTAMP NOT NULL,
    started_at          TIMESTAMP,
    ended_at            TIMESTAMP
);

CREATE INDEX idx_runs_project_id ON runs(project_id);

CREATE TABLE run_enrollments (
    run_id          UUID NOT NULL REFERENCES runs(id),
    user_id         BIGINT NOT NULL REFERENCES users(id),
    partition_id    INTEGER NOT NULL,
    client_kind     VARCHAR(16) NOT NULL DEFAULT 'SHARD',
    enrolled_at     TIMESTAMP NOT NULL,
    token_issued_at TIMESTAMP,
    PRIMARY KEY (run_id, user_id),
    CONSTRAINT uq_run_partition UNIQUE (run_id, partition_id)
);

ALTER TABLE projects ADD COLUMN active_run_id UUID;
ALTER TABLE projects ADD CONSTRAINT fk_projects_active_run
    FOREIGN KEY (active_run_id) REFERENCES runs(id) ON DELETE SET NULL;
