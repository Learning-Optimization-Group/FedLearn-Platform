package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.model.TrainingArm;
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
 * V27: a run records the intent it was started with -- the project-derived training settings and the effective
 * TLS and client-auth deployment settings -- so the FL server and the execution contract read one immutable
 * snapshot. Runs that predate the snapshot carry none and are recognised as legacy.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v27_run_intent",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V27RunIntentMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    private JdbcTemplate jdbc;

    private UUID project() {
        UUID id = UUID.randomUUID();
        jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                + "VALUES (?, ?, 'TINYNET_GOLDEN', 'tinynet_golden', ?, 'CREATED')", id, "v27-" + id, DEFAULT_ORG);
        return id;
    }

    private void legacyRun() {
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, status, "
                        + "recipe_key, created_at) VALUES (?, ?, 'FedAvg', 3, 2, 2, 'STARTING', 'TINYNET_GOLDEN', now())",
                UUID.randomUUID(), project());
    }

    private void run(Integer version, String arm, String modelName, Boolean dpEnabled, Double epsilon, Double delta,
                     Double clipNorm, Boolean tls, Boolean clientAuth) {
        jdbc.update("INSERT INTO runs (id, project_id, strategy, num_rounds, min_clients, clients_per_round, status, "
                        + "recipe_key, created_at, intent_version, intent_training_arm, intent_model_name, "
                        + "intent_task_type, intent_dp_enabled, intent_dp_target_epsilon, intent_dp_delta, "
                        + "intent_dp_clip_norm, intent_tls_required, intent_client_auth_required) "
                        + "VALUES (?, ?, 'FedAvg', 3, 2, 2, 'STARTING', 'TINYNET_GOLDEN', now(), ?, ?, ?, NULL, ?, ?, ?, "
                        + "?, ?, ?)",
                UUID.randomUUID(), project(), version, arm, modelName, dpEnabled, epsilon, delta, clipNorm, tls,
                clientAuth);
    }

    private void intent(String arm, Boolean dpEnabled, Double epsilon, Double delta, Double clipNorm) {
        run(1, arm, "tinynet_golden", dpEnabled, epsilon, delta, clipNorm, true, false);
    }

    @Test
    void aRunWithoutAnIntentIsAcceptedAsLegacy() {
        assertThatCode(this::legacyRun).doesNotThrowAnyException();
    }

    @Test
    void aCompleteIntentIsAccepted() {
        assertThatCode(() -> intent("FULL", false, null, null, null)).doesNotThrowAnyException();
        assertThatCode(() -> intent("FULL", true, 4.0, 1e-5, 1.0)).doesNotThrowAnyException();
    }

    @Test
    void everyTrainingArmIsAccepted() {
        for (TrainingArm arm : TrainingArm.values()) {
            assertThatCode(() -> intent(arm.name(), false, null, null, null)).doesNotThrowAnyException();
        }
    }

    @Test
    void anUnknownArmIsRejected() {
        assertThatThrownBy(() -> intent("PARTIAL", false, null, null, null))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void onlyVersionOneIsAccepted() {
        assertThatThrownBy(() -> run(2, "FULL", "tinynet_golden", false, null, null, null, true, false))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void intentValuesWithoutAVersionAreRejected() {
        assertThatThrownBy(() -> run(null, "FULL", "tinynet_golden", false, null, null, null, true, false))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void aVersionedIntentMustBeComplete() {
        assertThatThrownBy(() -> run(1, "FULL", "tinynet_golden", false, null, null, null, null, false))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> run(1, "FULL", null, false, null, null, null, true, false))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void privacySettingsAreRecordedOnlyWhenPrivacyIsOn() {
        assertThatThrownBy(() -> intent("FULL", false, 4.0, 1e-5, 1.0))
                .isInstanceOf(DataIntegrityViolationException.class);
    }

    @Test
    void recordedPrivacySettingsMustBeInRange() {
        assertThatThrownBy(() -> intent("FULL", true, 4.0, 1.0, 1.0))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> intent("FULL", true, 0.0, 1e-5, 1.0))
                .isInstanceOf(DataIntegrityViolationException.class);
        assertThatThrownBy(() -> intent("FULL", true, 4.0, 1e-5, 0.0))
                .isInstanceOf(DataIntegrityViolationException.class);
    }
}
