package com.federated.fl_platform_api.model;

import java.io.Serializable;
import java.util.Objects;
import java.util.UUID;

public class OrganizationMembershipId implements Serializable {

    private UUID orgId;
    private Long userId;

    public OrganizationMembershipId() { /* JPA */ }

    public OrganizationMembershipId(UUID orgId, Long userId) {
        this.orgId = orgId;
        this.userId = userId;
    }

    public UUID getOrgId() { return orgId; }
    public Long getUserId() { return userId; }

    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof OrganizationMembershipId other)) return false;
        return Objects.equals(orgId, other.orgId) && Objects.equals(userId, other.userId);
    }
    @Override public int hashCode() { return Objects.hash(orgId, userId); }
}
