package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.RunExecutionAttempt;
import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface RunExecutionAttemptRepository extends JpaRepository<RunExecutionAttempt, UUID> {

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select a from RunExecutionAttempt a where a.id = :id")
    Optional<RunExecutionAttempt> lockById(@Param("id") UUID id);

    List<RunExecutionAttempt> findByRunIdOrderByAttemptNumberAsc(UUID runId);

    Optional<RunExecutionAttempt> findByRunIdAndAttemptNumber(UUID runId, int attemptNumber);

    Optional<RunExecutionAttempt> findFirstByRunIdOrderByAttemptNumberDesc(UUID runId);
}
