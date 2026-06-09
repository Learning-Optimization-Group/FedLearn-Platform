package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.ProjectMembershipId;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ProjectMembershipRepository
        extends JpaRepository<ProjectMembership, ProjectMembershipId> {

    List<ProjectMembership> findByIdProjectId(UUID projectId);

    List<ProjectMembership> findByIdProjectIdAndRole(UUID projectId, MembershipRole role);

    List<ProjectMembership> findByIdUserId(Long userId);

    Optional<ProjectMembership> findByIdProjectIdAndIdUserId(UUID projectId, Long userId);

    boolean existsByIdProjectIdAndIdUserIdAndRole(UUID projectId, Long userId, MembershipRole role);

    long countByIdProjectId(UUID projectId);

    /**
     * Returns the current MAX(partition_id) for the project; -1 if the project
     * has no membership rows yet. Used by the partition-assignment path in
     * ClientApiController.getConnection. The project-level row lock that
     * serializes concurrent calls lives on ProjectRepository.lockById, which
     * is added in Task 7.
     */
    @Query("select coalesce(max(m.partitionId), -1) from ProjectMembership m where m.id.projectId = :projectId")
    int maxPartitionIdForProject(@Param("projectId") UUID projectId);

    void deleteByIdProjectIdAndIdUserId(UUID projectId, Long userId);
}
