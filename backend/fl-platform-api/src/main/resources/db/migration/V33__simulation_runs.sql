CREATE TABLE simulation_runs (
    id UUID PRIMARY KEY,
    version BIGINT NOT NULL DEFAULT 0,
    created_by BIGINT NOT NULL,
    status VARCHAR(20) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    recipe_json TEXT NOT NULL,
    progress_json TEXT,
    record_json TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    requested_rounds INTEGER NOT NULL,
    completed_rounds INTEGER,
    worker_pid BIGINT,
    worker_start_identity VARCHAR(80),
    worker_executable_identity VARCHAR(1024),
    exit_code INTEGER,
    error_code VARCHAR(60),
    error_message VARCHAR(240),
    CONSTRAINT fk_simulation_owner FOREIGN KEY (created_by) REFERENCES users(id),
    CONSTRAINT ck_simulation_status CHECK (status IN ('ACCEPTED','STARTING','RUNNING','STOPPING','COMPLETED','STOPPED','FAILED','TIMED_OUT','INTERRUPTED')),
    CONSTRAINT ck_simulation_rounds CHECK (requested_rounds BETWEEN 1 AND 20 AND (completed_rounds IS NULL OR completed_rounds BETWEEN 0 AND requested_rounds))
);
CREATE INDEX idx_simulation_owner_created ON simulation_runs(created_by, created_at DESC);
CREATE INDEX idx_simulation_status ON simulation_runs(status);
