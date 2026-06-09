package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.OrgRole;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.OrganizationMembershipId;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface OrganizationMembershipRepository
        extends JpaRepository<OrganizationMembership, OrganizationMembershipId> {

    List<OrganizationMembership> findByUserId(Long userId);
    List<OrganizationMembership> findByOrgId(UUID orgId);
    List<OrganizationMembership> findByOrgIdAndOrgRole(UUID orgId, OrgRole orgRole);
}
