package com.federated.fl_platform_api.identity;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:h2:mem:v5test;DB_CLOSE_DELAY=-1;MODE=PostgreSQL",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.jpa.hibernate.ddl-auto=none",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V5MigrationTest {

    @Autowired
    JdbcTemplate jdbc;

    @Test
    void organizations_table_exists_after_v5() {
        Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'ORGANIZATIONS'",
                Integer.class);
        assertThat(count).isEqualTo(1);
    }

    @Test
    void default_org_seeded_with_known_uuid() {
        String name = jdbc.queryForObject(
                "SELECT name FROM organizations WHERE id = '00000000-0000-0000-0000-000000000001'",
                String.class);
        assertThat(name).isEqualTo("Default");
    }

    @Test
    void users_platform_role_column_present() {
        Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS " +
                        "WHERE TABLE_NAME = 'USERS' AND COLUMN_NAME = 'PLATFORM_ROLE'",
                Integer.class);
        assertThat(count).isEqualTo(1);
    }

    @Test
    void users_status_column_check_constraint_rejects_invalid_value() {
        // First confirm the column exists.
        Integer columnCount = jdbc.queryForObject(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS " +
                        "WHERE TABLE_NAME = 'USERS' AND COLUMN_NAME = 'STATUS'",
                Integer.class);
        assertThat(columnCount).isEqualTo(1);

        // Then prove the CHECK constraint is enforced by attempting an invalid status.
        org.assertj.core.api.Assertions.assertThatThrownBy(() -> jdbc.update(
                "INSERT INTO users(username, email, password, platform_role, status, " +
                        "created_at, updated_at) " +
                        "VALUES ('chk-test', 'chk@test.com', 'x', 'USER', 'ZOMBIE', " +
                        "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"))
                .isInstanceOf(Exception.class);
    }

    @Test
    void audit_events_table_exists() {
        Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'AUDIT_EVENTS'",
                Integer.class);
        assertThat(count).isEqualTo(1);
    }
}
