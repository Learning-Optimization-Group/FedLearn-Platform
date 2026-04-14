-- Baseline schema for fl-platform-api.
-- Mirrors the JPA entities in com.federated.fl_platform_api.model. All
-- subsequent schema changes must be added as V2__*.sql (or later) migrations.

CREATE TABLE users (
    id            BIGSERIAL PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL UNIQUE,
    email         VARCHAR(100) NOT NULL UNIQUE,
    password      VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP WITH TIME ZONE  NOT NULL,
    updated_at    TIMESTAMP WITH TIME ZONE  NOT NULL
);

CREATE TABLE projects (
    id           UUID         PRIMARY KEY,
    name         VARCHAR(255) NOT NULL UNIQUE,
    model_type   VARCHAR(255) NOT NULL,
    model_name   VARCHAR(255) NOT NULL,
    server_port  INTEGER,
    model_path   VARCHAR(1024),
    optimizer    VARCHAR(64),
    status       VARCHAR(32)  NOT NULL,
    user_id      BIGINT REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX idx_projects_user_id ON projects(user_id);

CREATE TABLE round_result (
    id              UUID    PRIMARY KEY,
    project_id      UUID    NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    server_round    INTEGER NOT NULL,
    loss            DOUBLE PRECISION,
    accuracy        DOUBLE PRECISION,
    gpu_utilization DOUBLE PRECISION
);

CREATE INDEX idx_round_result_project_id ON round_result(project_id);

CREATE TABLE server_logs (
    id          BIGSERIAL   PRIMARY KEY,
    project_id  UUID        NOT NULL,
    level       VARCHAR(16) NOT NULL,
    message     TEXT        NOT NULL,
    stack_trace TEXT,
    timestamp   TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE INDEX idx_server_logs_project_id ON server_logs(project_id);
CREATE INDEX idx_server_logs_timestamp  ON server_logs(timestamp);
