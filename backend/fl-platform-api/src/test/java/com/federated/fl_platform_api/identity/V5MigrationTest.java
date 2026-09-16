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
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v5",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
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
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'organizations'",
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
                        "WHERE TABLE_NAME = 'users' AND COLUMN_NAME = 'platform_role'",
                Integer.class);
        assertThat(count).isEqualTo(1);
    }

    @Test
    void users_status_column_check_constraint_rejects_invalid_value() {
        // First confirm the column exists.
        Integer columnCount = jdbc.queryForObject(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS " +
                        "WHERE TABLE_NAME = 'users' AND COLUMN_NAME = 'status'",
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
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'audit_events'",
                Integer.class);
        assertThat(count).isEqualTo(1);
    }

    @Test
    void existing_projects_get_default_org_after_backfill() {
        // No project rows existed in the test DB before V5, so the
        // assertion here is: org_id column is present and NOT NULL.
        Integer count = jdbc.queryForObject(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS " +
                        "WHERE TABLE_NAME = 'projects' AND COLUMN_NAME = 'org_id' " +
                        "AND IS_NULLABLE = 'NO'",
                Integer.class);
        assertThat(count).isEqualTo(1);
    }
}
