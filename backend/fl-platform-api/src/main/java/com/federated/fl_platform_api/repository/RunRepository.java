package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import jakarta.persistence.LockModeType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface RunRepository extends JpaRepository<Run, UUID> {

    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select r from Run r where r.id = :id")
    Optional<Run> lockById(@Param("id") UUID id);

    Optional<Run> findFirstByProjectIdAndStatusOrderByStartedAtDesc(UUID projectId, RunStatus status);

    List<Run> findByProjectId(UUID projectId);
}
