package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.ProjectAccessRequest;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ProjectAccessRequestRepository extends JpaRepository<ProjectAccessRequest, Long> {

    List<ProjectAccessRequest> findByProjectIdAndStatus(UUID projectId, AccessRequestStatus status);

    List<ProjectAccessRequest> findByProjectId(UUID projectId);

    List<ProjectAccessRequest> findByUserId(Long userId);

    Optional<ProjectAccessRequest> findByProjectIdAndUserId(UUID projectId, Long userId);

    /**
     * All of {@code userId}'s access requests across the supplied project ids, in
     * ONE query — batches the per-candidate lookup on the discover feed (BA-10) so
     * a feed of N candidates no longer issues N access-request SELECTs. Callers
     * must skip this for an empty {@code projectIds} — an empty {@code IN ()}
     * clause is invalid SQL on most databases.
     */
    List<ProjectAccessRequest> findByUserIdAndProjectIdIn(Long userId, Collection<UUID> projectIds);

    long countByStatus(AccessRequestStatus status);
}
