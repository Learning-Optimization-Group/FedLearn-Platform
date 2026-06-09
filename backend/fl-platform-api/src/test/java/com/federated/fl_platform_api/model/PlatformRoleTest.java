package com.federated.fl_platform_api.model;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class PlatformRoleTest {

    @Test
    void valueOf_resolvesBothRoles() {
        assertThat(PlatformRole.valueOf("USER")).isEqualTo(PlatformRole.USER);
        assertThat(PlatformRole.valueOf("PLATFORM_ADMIN")).isEqualTo(PlatformRole.PLATFORM_ADMIN);
    }

    @Test
    void hasExactlyTwoRoles() {
        assertThat(PlatformRole.values()).hasSize(2);
    }

    @Test
    void authority_prefixesRoleName() {
        assertThat(PlatformRole.PLATFORM_ADMIN.authority()).isEqualTo("ROLE_PLATFORM_ADMIN");
        assertThat(PlatformRole.USER.authority()).isEqualTo("ROLE_USER");
    }
}
