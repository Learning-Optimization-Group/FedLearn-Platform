package com.federated.fl_platform_api.organization;

import com.federated.fl_platform_api.model.OrgRole;
import com.federated.fl_platform_api.model.Organization;
import com.federated.fl_platform_api.model.OrganizationMembership;
import com.federated.fl_platform_api.model.OrganizationMembershipId;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OrganizationMembershipRepository;
import com.federated.fl_platform_api.repository.OrganizationRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DataJpaTest
@ActiveProfiles("test")
class OrganizationMembershipTest {

    @Autowired OrganizationRepository orgRepo;
    @Autowired OrganizationMembershipRepository memRepo;
    @Autowired UserRepository userRepo;
    @Autowired JdbcTemplate jdbc;

    @Test
    void persists_with_composite_pk() {
        Organization org = orgRepo.saveAndFlush(new Organization(UUID.randomUUID(), "Acme", "acme-pk"));
        User user = userRepo.saveAndFlush(newUser("u1"));

        OrganizationMembership m = new OrganizationMembership(org.getId(), user.getId(), OrgRole.OWNER);
        memRepo.saveAndFlush(m);

        assertThat(memRepo.findById(new OrganizationMembershipId(org.getId(), user.getId())))
                .isPresent()
                .get()
                .extracting(OrganizationMembership::getOrgRole)
                .isEqualTo(OrgRole.OWNER);
    }

    @Test
    void rejects_invalid_role_at_db_layer() {
        Organization org = orgRepo.saveAndFlush(new Organization(UUID.randomUUID(), "Acme", "acme-role"));
        User user = userRepo.saveAndFlush(newUser("u2"));

        assertThatThrownBy(() -> jdbc.update(
                "INSERT INTO organization_memberships(org_id,user_id,org_role,created_at) " +
                        "VALUES (?,?, 'OVERLORD', CURRENT_TIMESTAMP)",
                org.getId(), user.getId()))
                .isInstanceOf(Exception.class);
    }

    private User newUser(String username) {
        // NOTE: User uses @GeneratedValue(IDENTITY); do not setId. The field is `password`, not passwordHash.
        // platform_role rename happens in Task 3; until that lands, this helper uses setRole(...) instead.
        User u = new User();
        u.setUsername(username);
        u.setEmail(username + "@example.com");
        u.setPassword("x");
        u.setRole("USER");   // Task 3 will rename this to setPlatformRole
        return u;
    }
}
