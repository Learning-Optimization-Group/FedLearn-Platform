package com.federated.fl_platform_api.identity;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * Validates the V6 hardening migration.
 *
 * <p>Testcontainers is not a dependency of this module, so the Postgres-native
 * {@code JSONB} column type cannot be asserted against a real Postgres here
 * (H2-in-PostgreSQL-mode reports its own type name). The full migration is
 * exercised end-to-end against real Postgres by CI / prod Flyway.
 *
 * <p>What this test <i>does</i> assert:
 * <ul>
 *   <li>The {@code platform_role} CHECK constraint is live — an out-of-range
 *       value is rejected — by running every migration through V6 against the
 *       same H2-in-PostgreSQL-mode Flyway path the V5 test uses.</li>
 *   <li>The V6 migration file exists and contains the platform_role CHECK and
 *       the JSONB conversion of {@code audit_events.metadata}.</li>
 * </ul>
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:h2:mem:v6test;DB_CLOSE_DELAY=-1;MODE=PostgreSQL",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=none",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V6MigrationTest {

    private static final Path V6 = Path.of(
            "src/main/resources/db/migration/V6__identity_hardening.sql");

    @Autowired
    JdbcTemplate jdbc;

    @Test
    void platform_role_check_constraint_rejects_invalid_value() {
        // V6 normalises ADMIN -> PLATFORM_ADMIN then adds a CHECK restricting
        // platform_role to ('USER','PLATFORM_ADMIN'). Prove the CHECK is live.
        assertThatThrownBy(() -> jdbc.update(
                "INSERT INTO users(username, email, password, platform_role, status, " +
                        "created_at, updated_at) " +
                        "VALUES ('v6-bad-role', 'v6@test.com', 'x', 'SUPERUSER', 'ACTIVE', " +
                        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
                .isInstanceOf(Exception.class);
    }

    @Test
    void platform_role_check_constraint_accepts_valid_value() {
        int rows = jdbc.update(
                "INSERT INTO users(username, email, password, platform_role, status, " +
                        "created_at, updated_at) " +
                        "VALUES ('v6-good-role', 'v6ok@test.com', 'x', 'PLATFORM_ADMIN', 'ACTIVE', " +
                        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)");
        assertThat(rows).isEqualTo(1);
    }

    @Test
    void v6_file_contains_check_and_jsonb_conversion() throws Exception {
        assertThat(Files.exists(V6)).as("V6 migration file present").isTrue();
        String sql = Files.readString(V6).toUpperCase();

        assertThat(sql)
                .as("platform_role CHECK constraint")
                .contains("CHECK (PLATFORM_ROLE IN ('USER','PLATFORM_ADMIN'))");
        assertThat(sql)
                .as("audit_events.metadata JSONB promotion")
                .contains("ALTER TABLE AUDIT_EVENTS ALTER COLUMN METADATA TYPE JSONB");
        assertThat(sql)
                .as("legacy ADMIN -> PLATFORM_ADMIN normalisation")
                .contains("UPDATE USERS SET PLATFORM_ROLE = 'PLATFORM_ADMIN' WHERE PLATFORM_ROLE = 'ADMIN'");
    }
}
