-- DA-3 hardening: dedup the shared BASE_REF row per (org_id, base_model_ref).
--
-- ArtifactRegistryService.findOrCreateBaseRef was a non-atomic read-then-insert with NO backing
-- constraint, so two concurrent adapter registrations in one org over the same base model could each
-- see "absent" and each INSERT a BASE_REF -> duplicate "deduped" base rows, whose adapters' ADAPTER_OF
-- edges then fork. This PARTIAL unique index makes one BASE_REF per (org_id, base_model_ref) a DB
-- invariant, while leaving LORA_ADAPTER / FULL_CHECKPOINT rows (which legitimately share those columns)
-- unconstrained. It also backs the ON CONFLICT DO NOTHING insert-if-absent the service now uses, making
-- find-or-create race-safe.
--
-- Assumes no pre-existing duplicate BASE_REFs — a registry created before this migration accrues them
-- only under the rare concurrent-completion race, and a fresh registry has none. If a deployed database
-- somehow already holds duplicates, this migration fails loudly here (rather than silently proceeding);
-- collapse the duplicate BASE_REFs (keep the earliest, repoint ADAPTER_OF edges) before re-applying.
CREATE UNIQUE INDEX uq_base_ref_org_model
    ON model_artifacts (org_id, base_model_ref)
    WHERE kind = 'BASE_REF';
