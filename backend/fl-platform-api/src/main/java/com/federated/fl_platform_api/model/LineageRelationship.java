package com.federated.fl_platform_api.model;

/**
 * The relationship a {@link ArtifactLineage} edge encodes from child to parent. Stored as VARCHAR
 * via {@code @Enumerated(STRING)}.
 */
public enum LineageRelationship {
    /** child is a LoRA adapter fine-tuned over the parent base. */
    ADAPTER_OF,
    /** child is otherwise derived from the parent (e.g. merged / distilled). */
    DERIVED_FROM,
    /** child continued federated training from the parent's weights. */
    CONTINUED_FROM
}
