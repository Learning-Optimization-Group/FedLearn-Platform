package com.federated.fl_platform_api.contract;

/** Why a run has no execution contract. Stored in run_execution_contracts.unavailable_reason (V28). */
public enum ContractUnavailableReason {
    /** Staging the run's artifacts failed. */
    STAGING_FAILED,
    /** Staging did not finish within its deadline. */
    STAGING_TIMED_OUT,
    /** A fact the contract needs cannot be represented in contract v1. */
    NOT_REPRESENTABLE,
    /** The assembled contract failed validation; the detail lists its issues. */
    INVALID_CONTRACT
}
