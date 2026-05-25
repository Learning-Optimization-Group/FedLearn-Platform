package com.federated.fl_platform_api.bootstrap;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.OrgRole;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.model.UserStatus;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.OrganizationRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Verifies the platform-admin bootstrap runner:
 *  - context startup creates the Platform org + PLATFORM_ADMIN user + OWNER membership,
 *  - emits the two bootstrap audit events,
 *  - is idempotent when re-invoked manually.
 *
 * The dev profile is required so that {@code app.jwt.secret} et al. resolve to dev
 * defaults; an isolated in-memory H2 schema avoids polluting the on-disk dev DB.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:h2:mem:bootstrap;DB_CLOSE_DELAY=-1;MODE=PostgreSQL",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=none",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173",
        "app.bootstrap.admin-email=root@fedlearn.io",
        "app.bootstrap.admin-username=root",
        "app.bootstrap.admin-password=devpass1234",
        "app.bootstrap.platform-org-name=Platform"
})
class BootstrapRunnerTest {

    @Autowired BootstrapRunner runner;
    @Autowired UserRepository users;
    @Autowired OrganizationRepository orgs;
    @Autowired OrganizationMembershipRepository memberships;
    @Autowired AuditEventRepository audits;
    @Autowired PasswordEncoder encoder;

    @Test
    void creates_platform_admin_and_org_with_two_audit_events() {
        // Spring Boot already ran the runner once during context startup.
        User admin = users.findByUsername("root").orElseThrow();
        assertThat(admin.getPlatformRole()).isEqualTo("PLATFORM_ADMIN");
        assertThat(admin.getEmailVerified()).isTrue();
        assertThat(admin.getStatus()).isEqualTo(UserStatus.ACTIVE);
        assertThat(encoder.matches("devpass1234", admin.getPassword())).isTrue();

        var org = orgs.findBySlug("platform").orElseThrow();
        assertThat(org.getName()).isEqualTo("Platform");

        List<OrganizationMembership> mems = memberships.findByUserId(admin.getId());
        assertThat(mems).hasSize(1);
        assertThat(mems.get(0).getOrgRole()).isEqualTo(OrgRole.OWNER);
        assertThat(mems.get(0).getOrgId()).isEqualTo(org.getId());

        List<AuditAction> actions = audits.findAll().stream()
                .map(AuditEvent::getAction)
                .toList();
        assertThat(actions).contains(AuditAction.BOOTSTRAP_ORG_CREATED,
                                     AuditAction.BOOTSTRAP_ADMIN_CREATED);
    }

    @Test
    void is_idempotent() {
        long usersBefore = users.count();
        long orgsBefore = orgs.count();
        long auditsBefore = audits.count();

        runner.run(null);   // explicit second invocation

        assertThat(users.count()).isEqualTo(usersBefore);
        assertThat(orgs.count()).isEqualTo(orgsBefore);
        assertThat(audits.count()).isEqualTo(auditsBefore);
    }
}
