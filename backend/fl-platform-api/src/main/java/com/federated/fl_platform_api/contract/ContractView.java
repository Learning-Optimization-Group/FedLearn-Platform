package com.federated.fl_platform_api.contract;

import com.fedlearn.contract.v1.ExecutionContract;

/**
 * A run's execution-contract state as read. {@code contract}, {@code contractBytes} and {@code contractId} are set
 * only when READY; {@code unavailableReason} and {@code unavailableDetail} only when UNAVAILABLE.
 */
public record ContractView(ContractState state, ExecutionContract contract, byte[] contractBytes, String contractId,
                           ContractUnavailableReason unavailableReason, String unavailableDetail) {

    static ContractView of(ContractState state) {
        return new ContractView(state, null, null, null, null, null);
    }
}
