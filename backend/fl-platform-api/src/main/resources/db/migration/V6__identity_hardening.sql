-- =====================================================================
-- V6: Identity Hardening (PostgreSQL-only)
-- - Normalise the legacy 'ADMIN' platform role to 'PLATFORM_ADMIN'
-- - Constrain users.platform_role to the two valid values
-- - Promote audit_events.metadata from text/CLOB to native JSONB
--
-- NOTE: This migration uses PostgreSQL-specific syntax (JSONB, ::jsonb).
-- The `test` profile disables Flyway and builds the schema from JPA entities
-- (create-drop on Testcontainers Postgres), so this file runs under the
-- dev/prod Flyway path and the dedicated V*MigrationTest classes.
-- =====================================================================

UPDATE users SET platform_role = 'PLATFORM_ADMIN' WHERE platform_role = 'ADMIN';

ALTER TABLE users ADD CONSTRAINT chk_users_platform_role
    CHECK (platform_role IN ('USER','PLATFORM_ADMIN'));

ALTER TABLE audit_events ALTER COLUMN metadata TYPE JSONB USING (NULLIF(metadata,'')::jsonb);
