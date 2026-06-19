package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.ProjectDeletionRequest;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ProjectDeletionRequestRepository extends JpaRepository<ProjectDeletionRequest, Long> {

    Optional<ProjectDeletionRequest> findByProjectId(UUID projectId);

    List<ProjectDeletionRequest> findByStatus(AccessRequestStatus status);

    Optional<ProjectDeletionRequest> findByProjectIdAndStatus(UUID projectId, AccessRequestStatus status);

    long countByStatus(AccessRequestStatus status);
}
