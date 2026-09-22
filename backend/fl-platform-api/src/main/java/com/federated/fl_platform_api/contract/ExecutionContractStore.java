package com.federated.fl_platform_api.contract;

import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunExecutionContract;
import com.federated.fl_platform_api.repository.RunExecutionContractRepository;
import com.fedlearn.contract.v1.ExecutionContract;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * The publication lifecycle of a run's execution contract (Stage 2C).
 *
 * <p>A run with an intent snapshot is PENDING until one decision is stored: READY with the contract's canonical
 * protobuf bytes and their SHA-256 as the contract ID, or UNAVAILABLE with a reason. The decision is stored by a
 * single insert that does nothing when one already exists, so racing attempts produce exactly one decision, and it
 * is never updated afterwards. A run without an intent snapshot is LEGACY_ONLY and never receives a contract.</p>
 */
@Service
public class ExecutionContractStore {

    /** The fedlearn.v2 RegisterClient protocol version the FL server accepts. */
    static final int SERVER_PROTOCOL_VERSION = 2;

    private static final int MAX_DETAIL_LENGTH = 2048;

    private final RunExecutionContractRepository repository;

    public ExecutionContractStore(RunExecutionContractRepository repository) {
        this.repository = repository;
    }

    /**
     * Validates {@code contract} for {@code run} and stores the decision: READY when it has no issues, otherwise
     * UNAVAILABLE with {@link ContractUnavailableReason#INVALID_CONTRACT} and the issues as detail.
     */
    @Transactional
    public PublicationOutcome publish(Run run, ExecutionContract contract) {
        requireIntent(run);
        List<ContractIssue> issues = ExecutionContractValidator.validate(contract, SERVER_PROTOCOL_VERSION,
                run.getId().toString(), run.getProjectId().toString());
        if (!issues.isEmpty()) {
            return decided(repository.insertUnavailable(run.getId(), ContractUnavailableReason.INVALID_CONTRACT.name(),
                    render(issues)), PublicationOutcome.MARKED_UNAVAILABLE);
        }
        byte[] bytes = contract.toByteArray();
        return decided(repository.insertReady(run.getId(), bytes, sha256(bytes)), PublicationOutcome.PUBLISHED);
    }

    /** Records that no contract will be published for {@code run}. */
    @Transactional
    public PublicationOutcome markUnavailable(Run run, ContractUnavailableReason reason, String detail) {
        requireIntent(run);
        return decided(repository.insertUnavailable(run.getId(), reason.name(), truncate(detail)),
                PublicationOutcome.MARKED_UNAVAILABLE);
    }

    @Transactional(readOnly = true)
    public ContractView read(Run run) {
        Optional<RunExecutionContract> decision = repository.findById(run.getId());
        if (decision.isEmpty()) {
            return ContractView.of(run.getIntent().isPresent() ? ContractState.PENDING : ContractState.LEGACY_ONLY);
        }
        RunExecutionContract d = decision.get();
        if (ContractState.READY.name().equals(d.getState())) {
            byte[] bytes = d.getContractBytes();
            if (!sha256(bytes).equals(d.getContractId())) {
                throw new IllegalStateException("Stored execution contract for run " + run.getId()
                        + " does not match its contract ID");
            }
            try {
                return new ContractView(ContractState.READY, ExecutionContractCodec.parseBinary(bytes), bytes.clone(),
                        d.getContractId(), null, null);
            } catch (MalformedContractException e) {
                throw new IllegalStateException("Stored execution contract for run " + run.getId()
                        + " does not parse", e);
            }
        }
        return new ContractView(ContractState.UNAVAILABLE, null, null, null, d.getUnavailableReason(),
                d.getUnavailableDetail());
    }

    private static void requireIntent(Run run) {
        if (run.getIntent().isEmpty()) {
            throw new IllegalStateException("Run " + run.getId()
                    + " predates the run-intent snapshot and never receives an execution contract");
        }
    }

    private static PublicationOutcome decided(int inserted, PublicationOutcome outcome) {
        return inserted == 1 ? outcome : PublicationOutcome.ALREADY_DECIDED;
    }

    private static String render(List<ContractIssue> issues) {
        return truncate(issues.stream()
                .map(issue -> issue.code().name() + " " + issue.path())
                .sorted()
                .collect(Collectors.joining("; ")));
    }

    private static String truncate(String detail) {
        if (detail == null || detail.length() <= MAX_DETAIL_LENGTH) {
            return detail;
        }
        return detail.substring(0, MAX_DETAIL_LENGTH);
    }

    static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
