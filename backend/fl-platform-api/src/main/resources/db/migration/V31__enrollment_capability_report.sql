-- Stage 3 D1: a device's capability report, sent with enrollment (platform, OS level, ABIs, memory, storage, app and
-- bridge versions, thermal and battery state). Informational only: static facts never approve training. Bounded so an
-- enrollment cannot carry an arbitrary document, and always stamped with when it was reported.
ALTER TABLE run_enrollments
    ADD COLUMN capability_report JSONB,
    ADD COLUMN capability_reported_at TIMESTAMPTZ;

ALTER TABLE run_enrollments
    ADD CONSTRAINT chk_run_enrollments_capability_report CHECK (
        capability_report IS NULL
        OR (jsonb_typeof(capability_report) = 'object'
            AND octet_length(capability_report::text) <= 4096
            AND capability_reported_at IS NOT NULL));
