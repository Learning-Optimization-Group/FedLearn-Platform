package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.RunExecutionContract;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.UUID;

/**
 * Stores a run's single execution-contract decision. Each insert is a no-op when a decision already exists, so of
 * any number of racing attempts exactly one returns 1.
 */
public interface RunExecutionContractRepository extends JpaRepository<RunExecutionContract, UUID> {

    @Modifying
    @Query(value = "INSERT INTO run_execution_contracts (run_id, state, contract_bytes, contract_id, decided_at) "
            + "VALUES (:runId, 'READY', :bytes, :contractId, now()) ON CONFLICT (run_id) DO NOTHING",
            nativeQuery = true)
    int insertReady(@Param("runId") UUID runId, @Param("bytes") byte[] bytes,
                    @Param("contractId") String contractId);

    @Modifying
    @Query(value = "INSERT INTO run_execution_contracts (run_id, state, unavailable_reason, unavailable_detail, "
            + "decided_at) VALUES (:runId, 'UNAVAILABLE', :reason, :detail, now()) ON CONFLICT (run_id) DO NOTHING",
            nativeQuery = true)
    int insertUnavailable(@Param("runId") UUID runId, @Param("reason") String reason,
                          @Param("detail") String detail);
}
