-- server_logs.project_id was untyped (no FK) so deleting a project left
-- its log rows behind as orphans referencing a missing UUID. Add the FK
-- with ON DELETE CASCADE so logs disappear with their project.
--
-- Cleanup any orphan rows first, otherwise the FK creation fails.
DELETE FROM server_logs
 WHERE project_id NOT IN (SELECT id FROM projects);

ALTER TABLE server_logs
    ADD CONSTRAINT fk_server_logs_project
        FOREIGN KEY (project_id)
        REFERENCES projects(id)
        ON DELETE CASCADE;
