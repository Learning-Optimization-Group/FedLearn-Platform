-- V30: a run intent records where the run's training data comes from.
--
-- A run either trains the recipe's committed fixture data, served to phones by the run's server (a test or demo run),
-- or trains each participant's own dataset imported on its device. The execution contract states this as its data
-- source, and a phone decides from it whether the server supplies any data.
--
-- The snapshot layout becomes version 3, which requires the round timeout (as version 2) and the data source.
-- Version-1 and -2 snapshots were taken without one and record none.

ALTER TABLE runs ADD COLUMN intent_data_source VARCHAR(16);

ALTER TABLE runs DROP CONSTRAINT chk_runs_intent_complete;

-- Every predicate is NULL-safe because a CHECK that evaluates to NULL passes.
ALTER TABLE runs ADD CONSTRAINT chk_runs_intent_complete
    CHECK ((intent_version IS NULL
            AND intent_training_arm IS NULL AND intent_model_name IS NULL AND intent_task_type IS NULL
            AND intent_dp_enabled IS NULL AND intent_dp_target_epsilon IS NULL AND intent_dp_delta IS NULL
            AND intent_dp_clip_norm IS NULL AND intent_tls_required IS NULL
            AND intent_client_auth_required IS NULL AND intent_round_timeout_ms IS NULL
            AND intent_data_source IS NULL)
           OR (intent_version IS NOT NULL AND intent_version IN (1, 2, 3)
               AND intent_training_arm IS NOT NULL AND intent_model_name IS NOT NULL
               AND intent_dp_enabled IS NOT NULL AND intent_tls_required IS NOT NULL
               AND intent_client_auth_required IS NOT NULL
               AND ((intent_version = 1 AND intent_round_timeout_ms IS NULL AND intent_data_source IS NULL)
                    OR (intent_version = 2 AND intent_round_timeout_ms IS NOT NULL
                        AND intent_round_timeout_ms > 0 AND intent_data_source IS NULL)
                    OR (intent_version = 3 AND intent_round_timeout_ms IS NOT NULL
                        AND intent_round_timeout_ms > 0 AND intent_data_source IS NOT NULL
                        AND intent_data_source IN ('FIXTURE', 'LOCAL_SNAPSHOT')))));
