package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.model.RobustMethod;
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
 * V24: a run records which Byzantine-robust rule actually ran, and with what settings.
 *
 * <p>Before this a Robust run was stored as {@code strategy = 'Robust'} and nothing else, so a run
 * manifest could not say whether median or Bulyan produced a result. That is the same bug class V22
 * closed for the training arm: an implicit setting lets two different experiments share one label.
 *
 * <p>The CHECK constraints are the last line of defence behind DTO and service validation - any direct
 * writer (a migration, an ops script) bypasses those. Like the other migration tests, this runs the real
 * Flyway migrations against Postgres with {@code ddl-auto=validate}, which also proves the entity maps
 * onto the migrated schema.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v24_robust",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V24RobustAggregationMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    private JdbcTemplate jdbc;

    private UUID project() {
        UUID id = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'PNEUMONIA_CNN', 'pneumonia_cnn', ?, 'CREATED')", id, "v24-" + id, DEFAULT_ORG);
        return id;
    }

    private void run(String strategy, String method, Double fraction, Double trim, Double tau) {
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, "
                        + "status, recipe_key, created_at, robust_method, robust_byzantine_fraction, "
                        + "robust_trim_ratio, centered_clip_tau) "
                        + "VALUES (?, ?, ?, 5, 20, 20, 'STARTING', 'PNEUMONIA_CNN', now(), ?, ?, ?, ?)",
                UUID.randomUUID(), project(), strategy, method, fraction, trim, tau);
    }

    @Test
    void aNonRobustRunNeedsNoRobustSettings() {
        assertThatCode(() -> run("FedAvg", null, null, null, null)).doesNotThrowAnyException();
    }

    @Test
    void everyRobustMethodTheBackendDefinesIsAccepted() {
        // Drift guard: widening RobustMethod without widening the CHECK fails here, not at a user's start.
        for (RobustMethod m : RobustMethod.values()) {
            assertThatCode(() -> run("Robust", m.name(), 0.1, null, null)).doesNotThrowAnyException();
        }
    }

    @Test
    void anUnknownMethodIsRejected() {
        assertThatThrownBy(() -> run("Robust", "krum", null, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void robustSettingsOnAnotherStrategyAreRejected() {
        assertThatThrownBy(() -> run("FedAvg", "MEDIAN", null, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run("FedAvg", null, 0.1, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void byzantineFractionMustBeInZeroToHalf() {
        assertThatCode(() -> run("Robust", "KRUM", 0.0, null, null)).doesNotThrowAnyException();
        assertThatCode(() -> run("Robust", "KRUM", 0.49, null, null)).doesNotThrowAnyException();
        assertThatThrownBy(() -> run("Robust", "KRUM", 0.5, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run("Robust", "KRUM", -0.01, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void trimRatioMustBeInZeroToHalf() {
        assertThatCode(() -> run("Robust", "TRIMMED_MEAN", null, 0.0, null)).doesNotThrowAnyException();
        assertThatThrownBy(() -> run("Robust", "TRIMMED_MEAN", null, 0.5, null))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void clippingRadiusMustBePositive() {
        assertThatCode(() -> run("Robust", "CENTERED_CLIP", null, null, 1.5)).doesNotThrowAnyException();
        assertThatThrownBy(() -> run("Robust", "CENTERED_CLIP", null, null, 0.0))
                .isInstanceOf(DataIntegrityViolationException.class);
    }
}
