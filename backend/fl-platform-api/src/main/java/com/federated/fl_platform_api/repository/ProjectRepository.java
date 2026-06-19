package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.Project;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Lock;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import jakarta.persistence.LockModeType;

import java.util.Collection;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface ProjectRepository extends JpaRepository<Project, UUID> {

    List<Project> findByUserId(Long userId);

    /**
     * Returns the union of projects owned by the user and projects where the
     * user appears in project_memberships (any role). Used to populate the
     * "My Projects" dashboard.
     */
    @Query("""
        select distinct p from Project p
        left join ProjectMembership m on m.project = p
        where p.user.id = :userId or m.user.id = :userId
        order by p.name
    """)
    List<Project> findOwnedOrMemberOf(@Param("userId") Long userId);

    /**
     * Discover-feed candidates: every non-PRIVATE project (PUBLIC or RESTRICTED)
     * the caller neither owns nor has a membership in. PRIVATE projects are
     * invite-only and never surface in discovery. PUBLIC projects auto-join;
     * RESTRICTED projects require an owner-approved access request — that
     * distinction is enforced in AccessRequestService.submit, not here.
     */
    @Query("""
        select p from Project p
        where p.visibility <> com.federated.fl_platform_api.model.ProjectVisibility.PRIVATE
          and (p.user is null or p.user.id <> :userId)
          and not exists (
            select 1 from ProjectMembership m
            where m.project = p and m.user.id = :userId
          )
        order by p.name
    """)
    List<Project> findDiscoverable(@Param("userId") Long userId);

    /**
     * Org-scoped variant of {@link #findOwnedOrMemberOf(Long)}: same union of
     * owned + joined projects, additionally constrained to the caller's visible
     * orgs for multi-tenant isolation.
     */
    @Query("""
        select distinct p from Project p
        left join ProjectMembership m on m.project = p
        where (p.user.id = :userId or m.user.id = :userId)
          and p.orgId in :orgIds
        order by p.name
    """)
    List<Project> findOwnedOrMemberOfInOrgs(@Param("userId") Long userId,
                                            @Param("orgIds") Collection<UUID> orgIds);

    /**
     * Org-scoped variant of {@link #findDiscoverable(Long)}: same discover-feed
     * candidates, additionally constrained to the caller's visible orgs.
     */
    @Query("""
        select p from Project p
        where p.visibility <> com.federated.fl_platform_api.model.ProjectVisibility.PRIVATE
          and (p.user is null or p.user.id <> :userId)
          and not exists (
            select 1 from ProjectMembership m
            where m.project = p and m.user.id = :userId
          )
          and p.orgId in :orgIds
        order by p.name
    """)
    List<Project> findDiscoverableInOrgs(@Param("userId") Long userId,
                                         @Param("orgIds") Collection<UUID> orgIds);

    /**
     * Acquires a pessimistic write-lock on a single project row. Used by the
     * partition-assignment path to serialize concurrent connections for the
     * same project. Returns Optional.empty() if the project doesn't exist.
     */
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("select p from Project p where p.id = :id")
    Optional<Project> lockById(@Param("id") UUID id);
}
