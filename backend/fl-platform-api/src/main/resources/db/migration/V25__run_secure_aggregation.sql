-- V25: a run records whether it used secure aggregation, and the reconstruction threshold it ran with.
--
-- The run manifest is how a client learns, before it trains, that the run accepts only masked updates. The
-- phone cannot mask yet, so it has to refuse such a run rather than have every round refused, and that needs the
-- flag stored with the run.
--
-- secure_aggregation is NOT NULL DEFAULT FALSE, so existing rows read as "off", which they were: nothing before
-- this migration could turn it on from the platform. The threshold is set exactly when the flag is on.
--
-- The CHECK constraints are the last line of defence behind DTO and service validation, which any direct writer
-- (a migration, an ops script) bypasses. Masking exists only on DeComFL's gradient-scalar channel. fl_server.py
-- refuses a threshold below 2 or above the clients in a round, and each DeComFL round aggregates exactly
-- min_clients clients.

ALTER TABLE runs ADD COLUMN secure_aggregation   BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE runs ADD COLUMN secure_agg_threshold INTEGER;

-- On any other strategy fl_server.py would accept the flag and mask nothing.
ALTER TABLE runs ADD CONSTRAINT chk_runs_secure_aggregation_only_for_decomfl
    CHECK (secure_aggregation = FALSE OR strategy = 'DeComFL');

-- IS NOT NULL is spelled out because a CHECK that evaluates to NULL passes: "threshold >= 2" alone would admit a
-- secure run with no threshold.
ALTER TABLE runs ADD CONSTRAINT chk_runs_secure_agg_threshold
    CHECK ((secure_aggregation = FALSE AND secure_agg_threshold IS NULL)
           OR (secure_aggregation = TRUE
               AND secure_agg_threshold IS NOT NULL
               AND secure_agg_threshold >= 2
               AND secure_agg_threshold <= min_clients));
