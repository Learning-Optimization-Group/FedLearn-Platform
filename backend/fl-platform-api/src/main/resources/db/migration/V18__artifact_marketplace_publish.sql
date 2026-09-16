-- V18: FE-12 org-internal adapter marketplace.
--
-- Publishing marks a LORA_ADAPTER artifact discoverable across the PROJECTS within its own org.
-- Discovery stays strictly inside OrgScope — a cross-org / PUBLIC marketplace is a separate,
-- threat-model-sensitive effort and deliberately NOT enabled here. published_at is the sort key
-- for the marketplace feed (newest-published first).
ALTER TABLE model_artifacts ADD COLUMN published    BOOLEAN     NOT NULL DEFAULT FALSE;
ALTER TABLE model_artifacts ADD COLUMN published_at TIMESTAMPTZ;

-- Discovery: published adapters within a set of orgs. org_id + kind + published matches the
-- WHERE of listPublishedAdapters (findByOrgIdInAndKindAndPublishedIsTrue...).
CREATE INDEX idx_model_artifacts_marketplace
    ON model_artifacts (org_id, kind, published);
