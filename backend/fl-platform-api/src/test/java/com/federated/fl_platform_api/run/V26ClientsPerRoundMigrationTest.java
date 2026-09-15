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
 * V26: clients_per_round reaches the FL server, so a round can wait for more clients than it needs. A secure run's
 * reconstruction threshold is bounded by that round size instead of by min_clients, and a round size below the
 * minimum -- which would let a round finish short of it -- is rejected.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v26_round_size",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V26ClientsPerRoundMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    private JdbcTemplate jdbc;

    private UUID project() {
        UUID id = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'PNEUMONIA_CNN', 'pneumonia_cnn', ?, 'CREATED')", id, "v26-" + id, DEFAULT_ORG);
        return id;
    }

    private void run(int minClients, int clientsPerRound, boolean secure, Integer threshold) {
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, status, "
                        + "recipe_key, created_at, secure_aggregation, secure_agg_threshold) "
                        + "VALUES (?, ?, 'DeComFL', 5, ?, ?, 'STARTING', 'PNEUMONIA_CNN', now(), ?, ?)",
                UUID.randomUUID(), project(), minClients, clientsPerRound, secure, threshold);
    }

    @Test
    void aRoundSizeAboveTheMinimumIsAccepted() {
        assertThatCode(() -> run(3, 5, false, null)).doesNotThrowAnyException();
    }

    @Test
    void aRoundSizeBelowTheMinimumIsRejected() {
        assertThatThrownBy(() -> run(3, 2, false, null)).isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void aSecureThresholdAboveTheMinimumButWithinTheRoundSizeIsAccepted() {
        assertThatCode(() -> run(3, 5, true, 4)).doesNotThrowAnyException();
        assertThatCode(() -> run(3, 5, true, 5)).doesNotThrowAnyException();
    }

    @Test
    void aSecureThresholdAboveTheRoundSizeIsRejected() {
        assertThatThrownBy(() -> run(3, 5, true, 6)).isInstanceOf(DataIntegrityViolationException.class);
    }
}
