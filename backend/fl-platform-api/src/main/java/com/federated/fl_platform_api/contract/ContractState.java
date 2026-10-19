package com.federated.fl_platform_api.contract;

/** Where a run stands with respect to its execution contract. */
public enum ContractState {
    /** The run has an intent snapshot and no decision yet: clients wait, and never train on an inferred contract. */
    PENDING,
    /** One validated contract is stored and never changes. */
    READY,
    /** No contract will be published for this run; the reason says why. */
    UNAVAILABLE,
    /** The run predates the intent snapshot; no contract is fabricated for it. */
    LEGACY_ONLY
}
