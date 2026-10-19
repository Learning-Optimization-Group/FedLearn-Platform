package com.federated.fl_platform_api.run;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * V29: a version-2 run intent records the round timeout the FL server was given. Version-1 snapshots, taken before
 * the timeout was recorded, remain valid without one.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v29_round_timeout",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V29RunIntentRoundTimeoutMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    private JdbcTemplate jdbc;

    private void run(int version, Long roundTimeoutMs) {
        UUID project = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'TINYNET_GOLDEN', 'tinynet_golden', ?, 'CREATED')", project, "v29-" + project,
                DEFAULT_ORG);
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, status, "
                        + "recipe_key, created_at, intent_version, intent_training_arm, intent_model_name, "
                        + "intent_dp_enabled, intent_tls_required, intent_client_auth_required, "
                        + "intent_round_timeout_ms) VALUES (?, ?, 'FedAvg', 3, 2, 2, 'STARTING', 'TINYNET_GOLDEN', "
                        + "now(), ?, 'FULL', 'tinynet_golden', FALSE, TRUE, FALSE, ?)",
                UUID.randomUUID(), project, version, roundTimeoutMs);
    }

    @Test
    void aVersionTwoIntentRecordsAPositiveRoundTimeout() {
        assertThatCode(() -> run(2, 900_000L)).doesNotThrowAnyException();
        assertThatThrownBy(() -> run(2, null)).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run(2, 0L)).isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void aVersionOneIntentHasNoRoundTimeout() {
        assertThatCode(() -> run(1, null)).doesNotThrowAnyException();
        assertThatThrownBy(() -> run(1, 900_000L)).isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void otherVersionsAreRejected() {
        assertThatThrownBy(() -> run(3, 900_000L)).isInstanceOf(DataIntegrityViolationException.class);
    }
}
