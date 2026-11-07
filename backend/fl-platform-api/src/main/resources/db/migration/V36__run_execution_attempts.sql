-- V36: Execution attempts for durable, append-only training run continuation.
-- Each attempt records the OS process identity, token hash, port, and outcome
-- for a single execution epoch of a logical run.

ALTER TABLE runs ADD COLUMN current_attempt_id UUID;
ALTER TABLE runs ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 1;

CREATE TABLE run_execution_attempts (
    id                  UUID PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    attempt_number      INTEGER NOT NULL,
    status              VARCHAR(16) NOT NULL,
    server_pid          BIGINT,
    process_started_at  TIMESTAMP WITH TIME ZONE,
    server_port         INTEGER,
    internal_token_hash VARCHAR(64),
    started_at          TIMESTAMP WITH TIME ZONE,
    ended_at            TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL,
    failure_reason      TEXT,
    CONSTRAINT uq_run_attempt_number UNIQUE (run_id, attempt_number)
);

CREATE INDEX idx_run_execution_attempts_run_id ON run_execution_attempts(run_id);
