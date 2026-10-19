-- V27: a run records the intent it was started with.
--
-- The project holds mutable settings, and the TLS and client-auth settings belong to the backend's configuration,
-- which can differ after a restart. A run now snapshots, at start, every project-derived value the FL server is
-- spawned with (training arm, model name, task type, central-DP settings) and the effective deployment settings.
-- The spawn reads this snapshot, and the execution contract is built from it, so neither can drift from what the
-- run actually executes.
--
-- intent_version is NULL for runs that predate the snapshot: they have no execution contract and are served as
-- legacy runs. Version 1 requires the whole snapshot. The project's optimizer is deliberately not recorded: it only
-- parameterises server-side model initialisation, and clients choose their optimizer from the recipe.

ALTER TABLE runs ADD COLUMN intent_version              SMALLINT;
ALTER TABLE runs ADD COLUMN intent_training_arm         VARCHAR(32);
ALTER TABLE runs ADD COLUMN intent_model_name           VARCHAR(255);
ALTER TABLE runs ADD COLUMN intent_task_type            VARCHAR(64);
ALTER TABLE runs ADD COLUMN intent_dp_enabled           BOOLEAN;
ALTER TABLE runs ADD COLUMN intent_dp_target_epsilon    DOUBLE PRECISION;
ALTER TABLE runs ADD COLUMN intent_dp_delta             DOUBLE PRECISION;
ALTER TABLE runs ADD COLUMN intent_dp_clip_norm         DOUBLE PRECISION;
ALTER TABLE runs ADD COLUMN intent_tls_required         BOOLEAN;
ALTER TABLE runs ADD COLUMN intent_client_auth_required BOOLEAN;

-- A legacy run carries no intent value at all; a version-1 run carries every required one. Every predicate is
-- NULL-safe because a CHECK that evaluates to NULL passes: "intent_version = 1" alone would admit intent values
-- with no version.
ALTER TABLE runs ADD CONSTRAINT chk_runs_intent_complete
    CHECK ((intent_version IS NULL
            AND intent_training_arm IS NULL AND intent_model_name IS NULL AND intent_task_type IS NULL
            AND intent_dp_enabled IS NULL AND intent_dp_target_epsilon IS NULL AND intent_dp_delta IS NULL
            AND intent_dp_clip_norm IS NULL AND intent_tls_required IS NULL
            AND intent_client_auth_required IS NULL)
           OR (intent_version IS NOT NULL AND intent_version = 1
               AND intent_training_arm IS NOT NULL AND intent_model_name IS NOT NULL
               AND intent_dp_enabled IS NOT NULL AND intent_tls_required IS NOT NULL
               AND intent_client_auth_required IS NOT NULL));

-- The same vocabulary as chk_projects_training_arm (V22, widened by V23). Widening TrainingArm widens both.
ALTER TABLE runs ADD CONSTRAINT chk_runs_intent_training_arm
    CHECK (intent_training_arm IS NULL OR intent_training_arm IN ('FULL', 'FROZEN_HEAD', 'OVA_LP'));

-- Privacy values are recorded only when central DP is on, and any recorded value is in range. An enabled but
-- incomplete configuration is recorded as requested; the spawn refuses it.
ALTER TABLE runs ADD CONSTRAINT chk_runs_intent_dp
    CHECK ((intent_dp_enabled IS NOT TRUE
            AND intent_dp_target_epsilon IS NULL AND intent_dp_delta IS NULL AND intent_dp_clip_norm IS NULL)
           OR (intent_dp_enabled IS TRUE
               AND (intent_dp_target_epsilon IS NULL OR intent_dp_target_epsilon > 0)
               AND (intent_dp_delta IS NULL OR (intent_dp_delta > 0 AND intent_dp_delta < 1))
               AND (intent_dp_clip_norm IS NULL OR intent_dp_clip_norm > 0)));
