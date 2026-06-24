-- Optional per-project device-requirements override (owner may tighten the recipe
-- defaults). Stored as JSON text; merged most-restrictive-wins at read time.
ALTER TABLE projects ADD COLUMN requirements_override TEXT;
