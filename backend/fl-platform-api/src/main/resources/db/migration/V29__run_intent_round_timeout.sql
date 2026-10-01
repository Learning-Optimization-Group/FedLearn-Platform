-- V29: a run intent records the round timeout its FL server was given.
--
-- The FL server's per-round deadline (FEDLEARN_ROUND_TIMEOUT_S) used to be inherited from the backend's environment,
-- so nothing recorded what a run was actually given. The backend now sets it explicitly from its configuration and
-- records it in the run's intent, which the execution contract reports as its round timeout.
--
-- The snapshot layout becomes version 2, which requires the timeout. Version-1 snapshots were taken without one and
-- cannot be backfilled truthfully, so they remain valid without it; their runs cannot publish a v1 contract.

ALTER TABLE runs ADD COLUMN intent_round_timeout_ms BIGINT;

ALTER TABLE runs DROP CONSTRAINT chk_runs_intent_complete;

-- Every predicate is NULL-safe because a CHECK that evaluates to NULL passes.
ALTER TABLE runs ADD CONSTRAINT chk_runs_intent_complete
    CHECK ((intent_version IS NULL
            AND intent_training_arm IS NULL AND intent_model_name IS NULL AND intent_task_type IS NULL
            AND intent_dp_enabled IS NULL AND intent_dp_target_epsilon IS NULL AND intent_dp_delta IS NULL
            AND intent_dp_clip_norm IS NULL AND intent_tls_required IS NULL
            AND intent_client_auth_required IS NULL AND intent_round_timeout_ms IS NULL)
           OR (intent_version IS NOT NULL AND intent_version IN (1, 2)
               AND intent_training_arm IS NOT NULL AND intent_model_name IS NOT NULL
               AND intent_dp_enabled IS NOT NULL AND intent_tls_required IS NOT NULL
               AND intent_client_auth_required IS NOT NULL
               AND ((intent_version = 1 AND intent_round_timeout_ms IS NULL)
                    OR (intent_version = 2 AND intent_round_timeout_ms IS NOT NULL
                        AND intent_round_timeout_ms > 0))));
