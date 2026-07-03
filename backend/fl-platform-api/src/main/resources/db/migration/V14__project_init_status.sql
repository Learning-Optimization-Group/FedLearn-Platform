-- BA-1: the one-time model-initialisation phase of a project, tracked independently of the
-- run-derived status (BA-4).
--
-- createProject used to run the (unbounded, Python-spawning) model init inside its @Transactional
-- request. It now persists the project shell as INITIALIZING, returns 201, and finishes init on a
-- bounded async worker that transitions the row to DONE (success) or FAILED (timeout/error). Because
-- status is derived from the active run and no run exists at create time, the init phase needs its
-- own column so a project mid-init doesn't read as the idle CREATED.
--
-- Every existing project was created synchronously (its init already completed), so the DEFAULT
-- backfills them all to DONE; the DEFAULT also keeps any future non-JPA insert path valid. Modelled
-- on projects.visibility / users.role (VARCHAR + @Enumerated(STRING) on the entity, no CHECK).
ALTER TABLE projects
    ADD COLUMN init_status VARCHAR(16) NOT NULL DEFAULT 'DONE';
