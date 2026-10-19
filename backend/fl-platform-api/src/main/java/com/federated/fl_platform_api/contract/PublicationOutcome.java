package com.federated.fl_platform_api.contract;

/** What one attempt to decide a run's execution contract did. */
public enum PublicationOutcome {
    /** This attempt stored the READY contract. */
    PUBLISHED,
    /** This attempt stored UNAVAILABLE. */
    MARKED_UNAVAILABLE,
    /** A decision was already stored; nothing changed. */
    ALREADY_DECIDED
}
