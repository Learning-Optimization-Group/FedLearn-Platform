package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.RunEnrollment;
import com.federated.fl_platform_api.model.RunEnrollmentId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.Optional;
import java.util.UUID;

public interface RunEnrollmentRepository extends JpaRepository<RunEnrollment, RunEnrollmentId> {

    Optional<RunEnrollment> findByIdRunIdAndIdUserId(UUID runId, Long userId);

    @Query("select coalesce(max(e.partitionId), -1) from RunEnrollment e where e.id.runId = :runId")
    int maxPartitionIdForRun(@Param("runId") UUID runId);

    long countByIdRunId(UUID runId);
}
