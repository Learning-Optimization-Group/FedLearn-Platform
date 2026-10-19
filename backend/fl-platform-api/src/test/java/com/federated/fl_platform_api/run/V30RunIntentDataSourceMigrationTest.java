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
 * V30: a version-3 run intent records where the run's training data comes from (FIXTURE or LOCAL_SNAPSHOT), which the
 * execution contract states as its data source. Version-1 and -2 snapshots, taken before, record none.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v30_data_source",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V30RunIntentDataSourceMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    private JdbcTemplate jdbc;

    private void run(int version, Long roundTimeoutMs, String dataSource) {
        UUID project = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'TINYNET_GOLDEN', 'tinynet_golden', ?, 'CREATED')", project, "v30-" + project,
                DEFAULT_ORG);
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, status, "
                        + "recipe_key, created_at, intent_version, intent_training_arm, intent_model_name, "
                        + "intent_dp_enabled, intent_tls_required, intent_client_auth_required, "
                        + "intent_round_timeout_ms, intent_data_source) VALUES (?, ?, 'FedAvg', 3, 2, 2, 'STARTING', "
                        + "'TINYNET_GOLDEN', now(), ?, 'FULL', 'tinynet_golden', FALSE, TRUE, FALSE, ?, ?)",
                UUID.randomUUID(), project, version, roundTimeoutMs, dataSource);
    }

    @Test
    void aVersionThreeIntentRecordsAKnownDataSource() {
        assertThatCode(() -> run(3, 900_000L, "FIXTURE")).doesNotThrowAnyException();
        assertThatCode(() -> run(3, 900_000L, "LOCAL_SNAPSHOT")).doesNotThrowAnyException();
        assertThatThrownBy(() -> run(3, 900_000L, null)).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run(3, 900_000L, "SERVER")).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run(3, null, "FIXTURE")).isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void earlierIntentVersionsRecordNoDataSource() {
        assertThatCode(() -> run(2, 900_000L, null)).doesNotThrowAnyException();
        assertThatCode(() -> run(1, null, null)).doesNotThrowAnyException();
        assertThatThrownBy(() -> run(2, 900_000L, "FIXTURE")).isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run(1, null, "LOCAL_SNAPSHOT")).isInstanceOf(DataIntegrityViolationException.class);
    }
}
