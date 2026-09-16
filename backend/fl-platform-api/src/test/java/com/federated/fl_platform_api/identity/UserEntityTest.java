package com.federated.fl_platform_api.identity;

import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.model.UserStatus;
import com.federated.fl_platform_api.repository.UserRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.ActiveProfiles;

import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ActiveProfiles("test")
class UserEntityTest {

    @Autowired UserRepository repo;

    @Test
    void persists_user_with_lifecycle_and_profile_columns() {
        // NOTE: User uses @GeneratedValue(IDENTITY); do not setId. The field is `password`, not passwordHash.
        User u = new User();
        u.setUsername("alice");
        u.setEmail("alice@example.com");
        u.setPassword("hash");
        u.setPlatformRole(com.federated.fl_platform_api.model.PlatformRole.USER);
        u.setStatus(UserStatus.ACTIVE);
        u.setEmailVerified(true);
        u.setDisplayName("Alice Liddell");
        u.setAvatarUrl("https://example.com/a.png");
        u.setLastLoginAt(Instant.now());

        repo.saveAndFlush(u);

        User found = repo.findById(u.getId()).orElseThrow();
        assertThat(found.getStatus()).isEqualTo(UserStatus.ACTIVE);
        assertThat(found.getEmailVerified()).isTrue();
        assertThat(found.getDisplayName()).isEqualTo("Alice Liddell");
        assertThat(found.getAvatarUrl()).isEqualTo("https://example.com/a.png");
        assertThat(found.getLastLoginAt()).isNotNull();
        assertThat(found.getPlatformRole()).isEqualTo(com.federated.fl_platform_api.model.PlatformRole.USER);
    }

    @Test
    void status_defaults_to_active_when_unset() {
        User u = new User();
        u.setUsername("bob");
        u.setEmail("bob@example.com");
        u.setPassword("h");
        u.setPlatformRole(com.federated.fl_platform_api.model.PlatformRole.USER);
        repo.saveAndFlush(u);
        assertThat(repo.findById(u.getId()).orElseThrow().getStatus()).isEqualTo(UserStatus.ACTIVE);
    }
}
