package com.federated.fl_platform_api.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.IdClass;
import jakarta.persistence.Table;
import org.hibernate.annotations.Check;

import java.time.Instant;
import java.util.UUID;

@Entity
@Table(name = "organization_memberships")
@IdClass(OrganizationMembershipId.class)
@Check(constraints = "org_role IN ('OWNER','ADMIN','MEMBER')")
public class OrganizationMembership {

    @Id
    @Column(name = "org_id", nullable = false)
    private UUID orgId;

    @Id
    @Column(name = "user_id", nullable = false)
    private Long userId;

    @Enumerated(EnumType.STRING)
    @Column(name = "org_role", nullable = false, length = 16)
    private OrgRole orgRole;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    protected OrganizationMembership() { /* JPA */ }

    public OrganizationMembership(UUID orgId, Long userId, OrgRole orgRole) {
        this.orgId = orgId;
        this.userId = userId;
        this.orgRole = orgRole;
        this.createdAt = Instant.now();
    }

    public UUID getOrgId() { return orgId; }
    public Long getUserId() { return userId; }
    public OrgRole getOrgRole() { return orgRole; }
    public void setOrgRole(OrgRole orgRole) { this.orgRole = orgRole; }
    public Instant getCreatedAt() { return createdAt; }
}
