-- Adds a role column to users so we can gate admin-only endpoints
-- (e.g. GET /api/users) without inventing a separate authorization table.
--
-- Existing rows default to 'USER'. To bootstrap an admin after deploying
-- this migration, run:
--   UPDATE users SET role = 'ADMIN' WHERE username = '<your-username>';

ALTER TABLE users
    ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'USER';

-- Speed up the inevitable "list all admins" / dashboard queries.
CREATE INDEX idx_users_role ON users(role);
