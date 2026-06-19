package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.ProjectAccessRequest;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ProjectAccessRequestRepository extends JpaRepository<ProjectAccessRequest, Long> {

    List<ProjectAccessRequest> findByProjectIdAndStatus(UUID projectId, AccessRequestStatus status);

    List<ProjectAccessRequest> findByProjectId(UUID projectId);

    List<ProjectAccessRequest> findByUserId(Long userId);

    Optional<ProjectAccessRequest> findByProjectIdAndUserId(UUID projectId, Long userId);

    long countByStatus(AccessRequestStatus status);
}
