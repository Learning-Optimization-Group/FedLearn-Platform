-- V24: a run records which Byzantine-robust aggregation rule it used, and the settings sent to the server.
--
-- Before this a Robust run was stored as strategy = 'Robust' and nothing more, so a run manifest could not say
-- whether median or Bulyan produced its result - two different experiments under one label, the bug class
-- V22 closed for the training arm.
--
-- All four columns are nullable and additive, so existing rows need no backfill: a pre-V24 Robust run keeps a
-- NULL rule, which honestly means "not recorded". New Robust runs always record a rule; one started without
-- settings records MEDIAN, fl_server.py's default.
--
-- The CHECK constraints are the last line of defence behind DTO and service validation, which any direct writer
-- (a migration, an ops script) bypasses. Bounds mirror the Python aggregator: trim ratio in [0, 0.5), clipping
-- radius > 0. The Byzantine fraction has no range check in Python, so [0, 0.5) here is its only bound.

ALTER TABLE runs ADD COLUMN robust_method             VARCHAR(16);
ALTER TABLE runs ADD COLUMN robust_byzantine_fraction DOUBLE PRECISION;
ALTER TABLE runs ADD COLUMN robust_trim_ratio         DOUBLE PRECISION;
ALTER TABLE runs ADD COLUMN centered_clip_tau         DOUBLE PRECISION;

-- Widening RobustMethod requires widening this list; V24RobustAggregationMigrationTest asserts every constant is
-- accepted so the enum and the constraint cannot drift.
ALTER TABLE runs ADD CONSTRAINT chk_runs_robust_method
    CHECK (robust_method IS NULL
           OR robust_method IN ('MEDIAN', 'TRIMMED_MEAN', 'KRUM', 'MULTI_KRUM', 'BULYAN', 'CENTERED_CLIP'));

-- Robust settings exist only on Robust runs; on any other strategy the server would ignore them.
ALTER TABLE runs ADD CONSTRAINT chk_runs_robust_settings_only_for_robust
    CHECK ((robust_method IS NULL
            AND robust_byzantine_fraction IS NULL
            AND robust_trim_ratio IS NULL
            AND centered_clip_tau IS NULL)
           OR strategy = 'Robust');

ALTER TABLE runs ADD CONSTRAINT chk_runs_robust_byzantine_fraction
    CHECK (robust_byzantine_fraction IS NULL
           OR (robust_byzantine_fraction >= 0 AND robust_byzantine_fraction < 0.5));

ALTER TABLE runs ADD CONSTRAINT chk_runs_robust_trim_ratio
    CHECK (robust_trim_ratio IS NULL OR (robust_trim_ratio >= 0 AND robust_trim_ratio < 0.5));

ALTER TABLE runs ADD CONSTRAINT chk_runs_centered_clip_tau
    CHECK (centered_clip_tau IS NULL OR centered_clip_tau > 0);
