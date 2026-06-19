package com.federated.fl_platform_api.model;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class PlatformRoleTest {

    @Test
    void valueOf_resolvesAllRoles() {
        assertThat(PlatformRole.valueOf("USER")).isEqualTo(PlatformRole.USER);
        assertThat(PlatformRole.valueOf("PROJECT_OWNER")).isEqualTo(PlatformRole.PROJECT_OWNER);
        assertThat(PlatformRole.valueOf("PLATFORM_ADMIN")).isEqualTo(PlatformRole.PLATFORM_ADMIN);
    }

    @Test
    void hasExactlyThreeRoles() {
        assertThat(PlatformRole.values()).hasSize(3);
    }

    @Test
    void authority_prefixesRoleName() {
        assertThat(PlatformRole.PLATFORM_ADMIN.authority()).isEqualTo("ROLE_PLATFORM_ADMIN");
        assertThat(PlatformRole.PROJECT_OWNER.authority()).isEqualTo("ROLE_PROJECT_OWNER");
        assertThat(PlatformRole.USER.authority()).isEqualTo("ROLE_USER");
    }
}
