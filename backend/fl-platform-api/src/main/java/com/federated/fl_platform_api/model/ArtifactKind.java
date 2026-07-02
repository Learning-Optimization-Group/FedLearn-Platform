package com.federated.fl_platform_api.model;

/**
 * What a {@link ModelArtifact} row represents. Stored as VARCHAR via {@code @Enumerated(STRING)}
 * (same convention as {@link RunStatus}); adding a kind never needs a schema change.
 */
public enum ArtifactKind {
    /** A complete model checkpoint (e.g. an imaging CNN) — the air-gap / export unit. */
    FULL_CHECKPOINT,
    /** A federated LoRA / PEFT adapter over a frozen base — the tradable marketplace unit. */
    LORA_ADAPTER,
    /** A reference to a frozen base model an org hosts/uses (many orgs may share one blob). */
    BASE_REF
}
