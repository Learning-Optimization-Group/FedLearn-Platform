-- P1-2: persist the training arm (frozen-head vs full fine-tune) on the project.
--
-- Before this the arm was not stored anywhere. fl-runtime/client.py inferred it from the recipe
-- key (`USE_DERIVED = (mt == "FROZEN_DEMO")`), which meant the platform could not answer "which
-- arm produced this result?" and could not run one recipe under both arms as the two halves of a
-- comparison -- the comparison research/results/frozen-backbone/ is built on, 177 result files
-- deep. Commit 21699bc ("frozen arm silently mislabelled its backbone, risking cell overwrites")
-- is that bug class: when the arm is implicit, two different experiments write the same cell.
--
-- DEFAULT 'FULL' is load-bearing for backward compatibility: every pre-existing project trained
-- every parameter, so backfilling them to FULL preserves their behaviour exactly, and a client
-- invocation that omits --training-arm resolves to FULL for the same reason.
--
-- The CHECK is deliberately narrow. It is the last line of defence: DTO validation can be bypassed
-- by any direct writer (a migration, an ops script, a future service), and an unrecognised arm
-- would otherwise reach the Python runtime and fail at FL-server spawn instead of at write time.
-- Widening the TrainingArm enum therefore requires a new migration widening this constraint;
-- V22TrainingArmMigrationTest asserts every enum constant is accepted so the two cannot drift.
ALTER TABLE projects
    ADD COLUMN training_arm VARCHAR(32) NOT NULL DEFAULT 'FULL';

ALTER TABLE projects
    ADD CONSTRAINT chk_projects_training_arm
    CHECK (training_arm IN ('FULL', 'FROZEN_HEAD'));
