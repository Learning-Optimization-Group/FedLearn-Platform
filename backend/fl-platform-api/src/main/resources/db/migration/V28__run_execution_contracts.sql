-- V28: each run's execution-contract decision.
--
-- A run with an intent snapshot (V27) is PENDING until exactly one decision is stored here: READY with the
-- contract's canonical protobuf bytes and their lowercase SHA-256 as the contract ID, or UNAVAILABLE with a reason.
-- PENDING is the absence of a row, and a run without an intent snapshot is LEGACY_ONLY, so neither is stored.
--
-- The run_id primary key lets ExecutionContractStore insert with ON CONFLICT DO NOTHING, so racing publication
-- attempts produce one decision. A decision never changes: clients identify a contract by its ID, and a READY
-- contract that later lost an artifact must surface as an integrity failure, not be replaced. The trigger below
-- enforces that for every writer; the row is still deleted with its run, like the rest of the run subtree (V19).

CREATE TABLE run_execution_contracts (
    run_id              UUID          PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    state               VARCHAR(16)   NOT NULL,
    contract_bytes      BYTEA,
    contract_id         VARCHAR(64),
    unavailable_reason  VARCHAR(32),
    unavailable_detail  VARCHAR(2048),
    decided_at          TIMESTAMPTZ   NOT NULL,
    -- NULL-safe throughout: a CHECK that evaluates to NULL passes.
    CONSTRAINT chk_run_execution_contracts_decision CHECK (
        (state = 'READY'
         AND contract_bytes IS NOT NULL
         AND contract_id IS NOT NULL AND contract_id ~ '^[0-9a-f]{64}$'
         AND unavailable_reason IS NULL AND unavailable_detail IS NULL)
        OR (state = 'UNAVAILABLE'
            AND contract_bytes IS NULL AND contract_id IS NULL
            AND unavailable_reason IS NOT NULL
            AND unavailable_reason IN ('STAGING_FAILED', 'STAGING_TIMED_OUT', 'NOT_REPRESENTABLE',
                                       'INVALID_CONTRACT')))
);

CREATE FUNCTION forbid_run_execution_contract_update() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'run_execution_contracts decisions are immutable (run %)', OLD.run_id;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_run_execution_contracts_immutable
    BEFORE UPDATE ON run_execution_contracts
    FOR EACH ROW EXECUTE FUNCTION forbid_run_execution_contract_update();
