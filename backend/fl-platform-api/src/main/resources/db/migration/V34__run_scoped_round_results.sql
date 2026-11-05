-- Old project-only records are ambiguous; retain them without inventing run provenance.
ALTER TABLE round_result ADD COLUMN run_id UUID REFERENCES runs(id) ON DELETE CASCADE;
ALTER TABLE round_result ADD CONSTRAINT uq_round_result_run_round UNIQUE (run_id, server_round);
