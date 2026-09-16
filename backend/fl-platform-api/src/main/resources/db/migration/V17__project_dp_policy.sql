-- SE-11: run-level differential-privacy policy on projects.
--
-- regulated:          marks a HIPAA-class/regulated project. The run-start gate refuses to spawn an
--                     FL server for a regulated project unless dp_enabled is TRUE and the DP config
--                     below is complete.
-- dp_enabled:         hands the config to the spawned FL server as --dp-* flags (fl_server.py).
-- dp_target_epsilon:  target privacy budget epsilon (> 0; guidance ~4-8 for medical/regulated data).
-- dp_delta:           DP failure probability delta, in (0,1) exclusive (typically < 1/N clients).
-- dp_clip_norm:       per-user (per-client) L2 contribution bound S (> 0).
--
-- The flags backfill every legacy row to FALSE and stay NOT NULL so policy checks never read a
-- three-valued boolean. The three knobs are nullable by design: a non-DP project carries no config.
-- Completeness/sanity (epsilon > 0, delta in (0,1), clip norm > 0) is enforced in Java at project
-- creation and again at the run-start gate — matching the V14 convention (validation lives in the
-- service layer, columns stay plain; no CHECK constraints).
ALTER TABLE projects
    ADD COLUMN regulated BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN dp_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN dp_target_epsilon DOUBLE PRECISION,
    ADD COLUMN dp_delta DOUBLE PRECISION,
    ADD COLUMN dp_clip_norm DOUBLE PRECISION;
