-- OvA-LP (arXiv:2511.05028) as a third training arm.
--
-- V22's CHECK constraint enumerated the two arms that existed then. The Java enum and the DTO
-- patterns are widened alongside this, and V22TrainingArmMigrationTest asserts that every
-- TrainingArm constant is accepted by the constraint -- so adding a constant without this
-- migration fails that test rather than surfacing as a write error in a user's federation.
ALTER TABLE projects DROP CONSTRAINT IF EXISTS chk_projects_training_arm;
ALTER TABLE projects ADD CONSTRAINT chk_projects_training_arm
    CHECK (training_arm IN ('FULL', 'FROZEN_HEAD', 'OVA_LP'));
