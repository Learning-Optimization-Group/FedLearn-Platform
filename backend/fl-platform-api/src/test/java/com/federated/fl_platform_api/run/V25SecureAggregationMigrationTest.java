package com.federated.fl_platform_api.run;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * V25: a run records whether it used secure aggregation, and at what reconstruction threshold.
 *
 * <p>The run manifest is how a phone decides whether it can take part, so the flag has to be stored rather than
 * inferred. The CHECK constraints are the last line of defence behind DTO and service validation: secure
 * aggregation only on DeComFL (masking exists nowhere else), a threshold of at least 2 whenever it is on, and no
 * threshold when it is off.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v25_secagg",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V25SecureAggregationMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    private JdbcTemplate jdbc;

    private UUID project() {
        UUID id = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'PNEUMONIA_CNN', 'pneumonia_cnn', ?, 'CREATED')", id, "v25-" + id, DEFAULT_ORG);
        return id;
    }

    /** Inserts a run; {@code secure == null} leaves the column to its default, as a pre-V25 writer would. */
    private UUID run(String strategy, Boolean secure, Integer threshold) {
        UUID id = UUID.randomUUID();
        if (secure == null) {
            jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, "
                            + "status, recipe_key, created_at, secure_agg_threshold) "
                            + "VALUES (?, ?, ?, 5, 3, 3, 'STARTING', 'PNEUMONIA_CNN', now(), ?)",
                    id, project(), strategy, threshold);
        } else {
            jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, "
                            + "status, recipe_key, created_at, secure_aggregation, secure_agg_threshold) "
                            + "VALUES (?, ?, ?, 5, 3, 3, 'STARTING', 'PNEUMONIA_CNN', now(), ?, ?)",
                    id, project(), strategy, secure, threshold);
        }
        return id;
    }

    @Test
    void aRunWrittenWithoutTheColumnDefaultsToOff() {
        UUID id = run("FedAvg", null, null);
        assertThat(jdbc.queryForObject("SELECT secure_aggregation FROM runs WHERE id = ?", Boolean.class, id)).isFalse();
    }

    @Test
    void aSecureDeComFLRunWithAThresholdIsAccepted() {
        assertThatCode(() -> run("DeComFL", true, 2)).doesNotThrowAnyException();
        assertThatCode(() -> run("DeComFL", true, 3)).doesNotThrowAnyException();   // = min_clients
    }

    @Test
    void aThresholdAboveMinClientsIsRejected() {
        // Each DeComFL round aggregates exactly min_clients (3 here) clients, so no round could gather 4 shares.
        assertThatThrownBy(() -> run("DeComFL", true, 4)).isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void secureAggregationOnAnotherStrategyIsRejected() {
        assertThatThrownBy(() -> run("FedAvg", true, 2)).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run("Robust", true, 2)).isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void aSecureRunNeedsAThresholdOfAtLeastTwo() {
        assertThatThrownBy(() -> run("DeComFL", true, null)).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run("DeComFL", true, 1)).isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void aThresholdWithoutSecureAggregationIsRejected() {
        assertThatThrownBy(() -> run("DeComFL", false, 2)).isInstanceOf(DataIntegrityViolationException.class);
    }
}
