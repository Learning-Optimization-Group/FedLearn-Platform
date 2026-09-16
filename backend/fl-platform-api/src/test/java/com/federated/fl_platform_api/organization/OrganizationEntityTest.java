package com.federated.fl_platform_api.organization;

import com.federated.fl_platform_api.model.Organization;
import com.federated.fl_platform_api.repository.OrganizationRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.test.context.ActiveProfiles;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ActiveProfiles("test")
class OrganizationEntityTest {

    @Autowired OrganizationRepository repo;

    @Test
    void persists_and_finds_by_slug() {
        UUID id = UUID.randomUUID();
        repo.saveAndFlush(new Organization(id, "Acme", "acme"));
        assertThat(repo.findBySlug("acme")).isPresent();
    }

    @Test
    void slug_is_unique() {
        repo.saveAndFlush(new Organization(UUID.randomUUID(), "Acme One", "acme"));
        assertThatThrownBy(() ->
                repo.saveAndFlush(new Organization(UUID.randomUUID(), "Acme Two", "acme")))
                .isInstanceOf(DataIntegrityViolationException.class);
    }
}
