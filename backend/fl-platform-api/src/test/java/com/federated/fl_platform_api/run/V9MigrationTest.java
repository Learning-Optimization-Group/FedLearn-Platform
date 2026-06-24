package com.federated.fl_platform_api.run;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Validates the V9 project requirements_override migration.
 *
 * <p>Runs every migration through V9 against a real PostgreSQL instance
 * (Testcontainers, {@code jdbc:tc:}). Asserts:
 * <ul>
 *   <li>{@code projects.requirements_override} column is added.</li>
 * </ul>
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v9",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=none",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V9MigrationTest {

    @Autowired
    JdbcTemplate jdbc;

    @Test
    void projectsHasRequirementsOverrideColumn() {
        Integer n = jdbc.queryForObject(
                "select count(*) from information_schema.columns " +
                        "where table_name='projects' and column_name='requirements_override'",
                Integer.class);
        assertThat(n).isEqualTo(1);
    }
}
