package com.federated.fl_platform_api.model;

import com.federated.fl_platform_api.contract.ContractUnavailableReason;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import org.hibernate.annotations.Immutable;

import java.time.Instant;
import java.util.UUID;

/**
 * A run's execution-contract decision (V28): READY with the canonical contract bytes and their SHA-256 ID, or
 * UNAVAILABLE with a reason. Written once by ExecutionContractStore and never updated; a trigger rejects updates.
 */
@Entity
@Immutable
@Table(name = "run_execution_contracts")
public class RunExecutionContract {

    @Id
    @Column(name = "run_id")
    private UUID runId;

    @Column(nullable = false, length = 16)
    private String state;

    @Column(name = "contract_bytes")
    private byte[] contractBytes;

    @Column(name = "contract_id", length = 64)
    private String contractId;

    @Enumerated(EnumType.STRING)
    @Column(name = "unavailable_reason", length = 32)
    private ContractUnavailableReason unavailableReason;

    @Column(name = "unavailable_detail", length = 2048)
    private String unavailableDetail;

    @Column(name = "decided_at", nullable = false)
    private Instant decidedAt;

    protected RunExecutionContract() {
    }

    public UUID getRunId() { return runId; }
    public String getState() { return state; }
    public byte[] getContractBytes() { return contractBytes; }
    public String getContractId() { return contractId; }
    public ContractUnavailableReason getUnavailableReason() { return unavailableReason; }
    public String getUnavailableDetail() { return unavailableDetail; }
    public Instant getDecidedAt() { return decidedAt; }
}
