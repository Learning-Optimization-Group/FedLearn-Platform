-- Retain reported legacy identities and values; their original callback payloads cannot be reconstructed.
ALTER TABLE benchmark_rounds DROP CONSTRAINT uq_benchmark_round;
ALTER TABLE benchmark_rounds ADD COLUMN payload_sha256 VARCHAR(64);
ALTER TABLE benchmark_rounds ADD CONSTRAINT uq_benchmark_run_round UNIQUE (run_id, server_round);
ALTER TABLE benchmark_rounds ADD CONSTRAINT ck_benchmark_payload_sha256
    CHECK (payload_sha256 IS NULL OR payload_sha256 ~ '^[0-9a-f]{64}$');
CREATE INDEX idx_benchmark_rounds_run ON benchmark_rounds(run_id);
ALTER TABLE benchmark_runs ADD COLUMN run_identity_verified BOOLEAN NOT NULL DEFAULT FALSE;
