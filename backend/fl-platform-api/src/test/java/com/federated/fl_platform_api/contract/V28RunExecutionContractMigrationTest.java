package com.federated.fl_platform_api.contract;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.dao.DataAccessException;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * V28: a run's execution-contract decision is stored once -- READY with its canonical bytes and ID, or UNAVAILABLE
 * with a reason -- and never changes afterwards. It is deleted with its run.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v28_run_contract",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V28RunExecutionContractMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");
    private static final String ID = "a".repeat(64);

    @Autowired
    private JdbcTemplate jdbc;

    private UUID run() {
        UUID project = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'TINYNET_GOLDEN', 'tinynet_golden', ?, 'CREATED')", project, "v28-" + project,
                DEFAULT_ORG);
        UUID run = UUID.randomUUID();
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, status, "
                + "recipe_key, created_at) VALUES (?, ?, 'FedAvg', 3, 2, 2, 'STARTING', 'TINYNET_GOLDEN', now())",
                run, project);
        return run;
    }

    private void decide(UUID run, String state, byte[] bytes, String contractId, String reason) {
        jdbc.update("INSERT INTO run_execution_contracts (run_id, state, contract_bytes, contract_id, "
                + "unavailable_reason, decided_at) VALUES (?, ?, ?, ?, ?, now())", run, state, bytes, contractId, reason);
    }

    @Test
    void aReadyDecisionCarriesItsBytesAndId() {
        assertThatCode(() -> decide(run(), "READY", new byte[] {8, 1}, ID, null)).doesNotThrowAnyException();
        assertThatThrownBy(() -> decide(run(), "READY", null, ID, null))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> decide(run(), "READY", new byte[] {8, 1}, "A".repeat(64), null))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> decide(run(), "READY", new byte[] {8, 1}, ID, "STAGING_FAILED"))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void anUnavailableDecisionCarriesAKnownReasonAndNoContract() {
        assertThatCode(() -> decide(run(), "UNAVAILABLE", null, null, "STAGING_FAILED")).doesNotThrowAnyException();
        assertThatThrownBy(() -> decide(run(), "UNAVAILABLE", null, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> decide(run(), "UNAVAILABLE", null, null, "SOMETHING_ELSE"))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> decide(run(), "UNAVAILABLE", new byte[] {8, 1}, ID, "STAGING_FAILED"))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void pendingIsNotStoredItIsTheAbsenceOfADecision() {
        assertThatThrownBy(() -> decide(run(), "PENDING", null, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void aDecisionNeverChanges() {
        UUID run = run();
        decide(run, "READY", new byte[] {8, 1}, ID, null);
        assertThatThrownBy(() -> jdbc.update("UPDATE run_execution_contracts SET contract_id = ? WHERE run_id = ?",
                "b".repeat(64), run)).isInstanceOf(DataAccessException.class);
        assertThatThrownBy(() -> decide(run, "UNAVAILABLE", null, null, "STAGING_FAILED"))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void aDecisionIsDeletedWithItsRun() {
        UUID run = run();
        decide(run, "READY", new byte[] {8, 1}, ID, null);
        jdbc.update("DELETE FROM runs WHERE id = ?", run);
        assertThat(jdbc.queryForObject("SELECT count(*) FROM run_execution_contracts WHERE run_id = ?",
                Integer.class, run)).isZero();
    }
}
