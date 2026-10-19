package com.federated.fl_platform_api.run;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.sql.Timestamp;
import java.time.Instant;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicLong;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * V31: an enrollment may carry the device's capability report (Stage 3 D1): informational JSON, bounded in size, with
 * the time it was reported. Enrollments without one are unchanged.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v31_capability",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V31EnrollmentCapabilityReportMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");
    private static final AtomicLong USERS = new AtomicLong(31_000);

    @Autowired
    private JdbcTemplate jdbc;

    private void enroll(String report, Instant reportedAt) {
        Timestamp now = Timestamp.from(Instant.now());
        long user = USERS.incrementAndGet();
        jdbc.update("INSERT INTO users (id, username, email, password, created_at, updated_at, platform_role) "
                + "VALUES (?, ?, ?, 'x', ?, ?, 'USER')", user, "v31-" + user, "v31-" + user + "@example.com", now, now);
        UUID project = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'TINYNET_GOLDEN', 'tinynet_golden', ?, 'CREATED')", project, "v31-" + project,
                DEFAULT_ORG);
        UUID run = UUID.randomUUID();
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, status, "
                + "recipe_key, created_at) VALUES (?, ?, 'FedAvg', 3, 1, 1, 'RUNNING', 'TINYNET_GOLDEN', ?)",
                run, project, now);
        jdbc.update("INSERT INTO run_enrollments (run_id, user_id, partition_id, client_kind, enrolled_at, "
                + "capability_report, capability_reported_at) VALUES (?, ?, 0, 'SHARD', ?, ?::jsonb, ?)",
                run, user, now, report, reportedAt == null ? null : Timestamp.from(reportedAt));
    }

    @Test
    void anEnrollmentMayCarryAReportWithItsTime() {
        assertThatCode(() -> enroll("{\"platform\":\"android\",\"apiLevel\":27}", Instant.now()))
                .doesNotThrowAnyException();
        assertThat(jdbc.queryForObject("SELECT capability_report->>'platform' FROM run_enrollments "
                + "WHERE capability_report IS NOT NULL LIMIT 1", String.class)).isEqualTo("android");
    }

    @Test
    void anEnrollmentWithoutAReportIsUnchanged() {
        assertThatCode(() -> enroll(null, null)).doesNotThrowAnyException();
    }

    @Test
    void aReportNeedsItsTimeAndAnObjectAndABoundedSize() {
        assertThatThrownBy(() -> enroll("{\"platform\":\"android\"}", null))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> enroll("[1,2]", Instant.now())).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> enroll("{\"pad\":\"" + "x".repeat(5000) + "\"}", Instant.now()))
                .isInstanceOf(DataIntegrityViolationException.class);
    }
}
