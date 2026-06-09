package com.federated.fl_platform_api.security;

import org.springframework.stereotype.Component;
import org.springframework.web.context.annotation.RequestScope;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.UUID;

/**
 * Request-scoped holder for the set of organization ids the current caller may
 * see, plus an {@code unrestricted} flag for platform admins (who bypass org
 * isolation entirely). Populated once per request by {@link OrgScopeFilter}
 * after authentication, then consulted by the service/authorization layer to
 * enforce multi-tenant isolation.
 *
 * <p>Exposes a no-arg constructor and {@link #set(Set, boolean)} so it can be
 * unit-tested without a Spring context.
 */
@Component
@RequestScope
public class OrgScope {

    private Set<UUID> visibleOrgIds = Collections.emptySet();
    private boolean unrestricted = false;

    /** Replaces the scope. {@code unrestricted} ⇒ caller sees every org. */
    public void set(Set<UUID> visibleOrgIds, boolean unrestricted) {
        this.visibleOrgIds = visibleOrgIds == null
                ? Collections.emptySet()
                : new HashSet<>(visibleOrgIds);
        this.unrestricted = unrestricted;
    }

    public Set<UUID> visibleOrgIds() {
        return Collections.unmodifiableSet(visibleOrgIds);
    }

    public boolean isUnrestricted() {
        return unrestricted;
    }

    /** True if the caller may access the given org (or is unrestricted). */
    public boolean allows(UUID orgId) {
        return unrestricted || visibleOrgIds.contains(orgId);
    }
}
