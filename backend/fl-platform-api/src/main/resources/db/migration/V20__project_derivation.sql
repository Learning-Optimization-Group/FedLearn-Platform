-- DA-14 Ph3.2: per-project derivation record.
--
-- A "derived" project starts from a base model and fine-tunes/derives (head + freeze + LoRA over a
-- frozen or pretrained backbone) instead of training an architecture from scratch. These columns
-- are OPT-IN and NULLABLE by design: a NULL derivation (the backfill for every legacy row, and the
-- default for any create request that omits them) means the project behaves EXACTLY as before — a
-- normal from-scratch recipe project. Nothing on the training path reads these yet (the
-- derivation-recipe mechanism lands in a later phase); this is the additive schema foundation.
--
-- init_from_pretrained: TRUE iff this project derives from a pretrained/frozen base. Backfills
--                       FALSE and stays NOT NULL so the flag is never three-valued (matches the V17
--                       dp_enabled convention).
-- base_ref_sha256:      content address (sha256) of the frozen BASE_REF backbone blob in the
--                       artifact registry this project derives from; NULL for a from-scratch project.
-- derivation_spec:      JSON describing the derivation (dataset / head / freeze / lora); NULL when
--                       absent. Semantic validation of the JSON (when present) lives in the Java
--                       service layer, matching the V14/V17 convention.
ALTER TABLE projects
    ADD COLUMN init_from_pretrained BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN base_ref_sha256      VARCHAR(64),
    ADD COLUMN derivation_spec      TEXT;

-- A present base_ref_sha256 must be lowercase-hex sha256, matching the artifact_blobs content-address
-- convention (V12 chk_artifact_blobs_sha256_hex). NULL is allowed (from-scratch projects).
ALTER TABLE projects
    ADD CONSTRAINT chk_projects_base_ref_sha256_hex
        CHECK (base_ref_sha256 IS NULL OR base_ref_sha256 ~ '^[0-9a-f]{64}$');
