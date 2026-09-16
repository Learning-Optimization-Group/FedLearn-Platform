package com.federated.fl_platform_api.arm;

import com.federated.fl_platform_api.model.TrainingArm;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * P1-2: the training arm (frozen-head vs full fine-tune) becomes a persisted, queryable property
 * of a project.
 *
 * <p>Before this, the arm was not stored anywhere. It was inferred inside the Python client from
 * the recipe key ({@code USE_DERIVED = (mt == "FROZEN_DEMO")}), which meant the platform could
 * neither answer "which arm produced this result?" nor run one recipe under both arms. That is the
 * bug class behind commit {@code 21699bc} — <em>"frozen arm silently mislabelled its backbone,
 * risking cell overwrites"</em>: when the arm is implicit, two different experiments write the
 * same cell.
 *
 * <p>Why this test runs Flyway rather than the default {@code test} profile: the column, its
 * NOT NULL default and its CHECK constraint are schema facts. Hibernate's {@code create-drop} test
 * schema would generate the column from the entity and prove nothing about the migration, and the
 * backfill of existing rows cannot be exercised at all without the real migration running. So —
 * like the other {@code V*MigrationTest} classes — every migration runs against real Postgres
 * (Testcontainers) with {@code flyway.enabled=true} and {@code ddl-auto=validate}.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v23_arm",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V22TrainingArmMigrationTest {

    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    private JdbcTemplate jdbc;

    private UUID insertProject(String name, String arm) {
        UUID id = UUID.randomUUID();
        if (arm == null) {
            // Exercises the column DEFAULT — the path every pre-existing row took at migration.
            jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status) "
                    + "VALUES (?, ?, 'PNEUMONIA_CNN', 'pneumonia_cnn', ?, 'CREATED')",
                    id, name, DEFAULT_ORG);
        } else {
            jdbc.update("INSERT INTO projects (id, name, model_type, model_name, org_id, status, "
                    + "training_arm) VALUES (?, ?, 'PNEUMONIA_CNN', 'pneumonia_cnn', ?, 'CREATED', ?)",
                    id, name, DEFAULT_ORG, arm);
        }
        return id;
    }

    @Test
    void columnExistsAndIsNotNull() {
        List<java.util.Map<String, Object>> cols = jdbc.queryForList(
                "SELECT column_name, is_nullable, column_default FROM information_schema.columns "
                        + "WHERE table_name = 'projects' AND column_name = 'training_arm'");
        assertThat(cols).as("V22 must add projects.training_arm").hasSize(1);
        assertThat(cols.get(0).get("is_nullable")).isEqualTo("NO");
        assertThat(String.valueOf(cols.get(0).get("column_default")))
                .as("existing rows and omitted inserts must land on FULL")
                .contains("FULL");
    }

    @Test
    void omittedArmDefaultsToFull() {
        UUID id = insertProject("arm-default-" + UUID.randomUUID(), null);
        String arm = jdbc.queryForObject(
                "SELECT training_arm FROM projects WHERE id = ?", String.class, id);
        assertThat(arm)
                .as("a project created without an arm must behave exactly as before P1")
                .isEqualTo("FULL");
    }

    @Test
    void frozenHeadRoundTrips() {
        UUID id = insertProject("arm-frozen-" + UUID.randomUUID(), "FROZEN_HEAD");
        assertThat(jdbc.queryForObject("SELECT training_arm FROM projects WHERE id = ?",
                String.class, id)).isEqualTo("FROZEN_HEAD");
    }

    @Test
    void theOvaLpArmRoundTrips() {
        // V23 widened the CHECK for OvA-LP (arXiv:2511.05028). Kept explicit alongside the
        // enum-driven test below so a reader sees which arms exist without running the loop.
        UUID id = insertProject("arm-ova-" + UUID.randomUUID(), "OVA_LP");
        assertThat(jdbc.queryForObject("SELECT training_arm FROM projects WHERE id = ?",
                String.class, id)).isEqualTo("OVA_LP");
    }

    @Test
    void anUnknownArmIsRejectedByTheDatabase() {
        // The DB is the last line of defence: DTO validation can be bypassed by any direct writer
        // (a migration, a script, a future service), and an unrecognised arm would then reach the
        // Python runtime, which resolves it against the recipe and would fail at spawn instead.
        assertThatThrownBy(() -> insertProject("arm-bad-" + UUID.randomUUID(), "SEMI_FROZEN"))
                .hasMessageContaining("training_arm");
    }

    @Test
    void everyEnumConstantIsAcceptedByTheCheckConstraint() {
        // Guards the split-brain failure: adding a Java enum constant without widening the CHECK
        // yields a value the application believes is valid and the database rejects at write time.
        for (TrainingArm arm : TrainingArm.values()) {
            UUID id = insertProject("arm-" + arm + "-" + UUID.randomUUID(), arm.name());
            assertThat(jdbc.queryForObject("SELECT training_arm FROM projects WHERE id = ?",
                    String.class, id)).isEqualTo(arm.name());
        }
    }
}
