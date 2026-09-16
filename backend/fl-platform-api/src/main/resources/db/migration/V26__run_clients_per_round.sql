-- V26: a run's rounds can wait for more clients than they need.
--
-- clients_per_round has been stored on every run since V8 but never reached the FL server, which built every
-- strategy with min_fit_clients alone. A round therefore needed every client, and one client missing one round
-- stopped the run. It is now passed as --clients-per-round: a round completes as soon as clients_per_round updates
-- arrive, and the deadline still resolves it with as few as min_clients. Two constraints follow.
--
-- 1. clients_per_round >= min_clients, since below the minimum a round would complete short of it. Nothing refused a
--    smaller value before, so older rows may hold one. Because the stored value never reached the server, those runs
--    actually ran with rounds of min_clients, and that is what they are corrected to before the check is added.
--
-- 2. A secure run's reconstruction threshold is bounded by the round size rather than by min_clients (V25).
--    build_servicer refuses a threshold above clients_per_round; one between min_clients and clients_per_round is
--    valid and raises the number of survivors a round needs.

UPDATE runs SET clients_per_round = min_clients WHERE clients_per_round < min_clients;

ALTER TABLE runs ADD CONSTRAINT chk_runs_clients_per_round_at_least_min
    CHECK (clients_per_round >= min_clients);

ALTER TABLE runs DROP CONSTRAINT chk_runs_secure_agg_threshold;

-- IS NOT NULL stays spelled out, for the reason given in V25: a CHECK that evaluates to NULL passes.
ALTER TABLE runs ADD CONSTRAINT chk_runs_secure_agg_threshold
    CHECK ((secure_aggregation = FALSE AND secure_agg_threshold IS NULL)
           OR (secure_aggregation = TRUE
               AND secure_agg_threshold IS NOT NULL
               AND secure_agg_threshold >= 2
               AND secure_agg_threshold <= clients_per_round));
