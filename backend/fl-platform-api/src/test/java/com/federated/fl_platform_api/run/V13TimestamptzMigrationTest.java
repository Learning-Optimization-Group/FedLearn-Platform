package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.model.PartitioningMode;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.RunRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import java.time.Instant;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Validates the V13 TIMESTAMPTZ-convention migration.
 *
 * <p>Runs every migration through V13 against a real PostgreSQL instance
 * (Testcontainers, {@code jdbc:tc:}). Asserts:
 * <ul>
 *   <li>Every timestamp column carried by the run aggregate ({@code runs.created_at/
 *       started_at/ended_at}, {@code run_enrollments.enrolled_at/token_issued_at}) reports
 *       {@code timestamp with time zone} in {@code information_schema.columns} — the V1
 *       convention that V8 dropped is restored.</li>
 *   <li>An {@link Instant} round-trips through the JPA {@code Run} mapping with no timezone
 *       drift, proving the columns store an absolute instant rather than a naive wall-clock.</li>
 * </ul>
 *
 * <p>{@code ddl-auto=none}: Hibernate never touches DDL, so the schema under test is exactly
 * what Flyway produced and the {@code data_type} assertions are the sole gate.
 */
@SpringBootTest
@ActiveProfiles("dev")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v13",
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=none",
        "spring.flyway.enabled=true",
        "app.jwt.secret=ZGV2LW9ubHktand0LXNlY3JldC1kby1ub3QtdXNlLWluLXByb2QhIQ==",
        "app.internal.api-key=test-internal-key",
        "app.cors.allowed-origins=http://localhost:5173"
})
class V13TimestamptzMigrationTest {

    /** Default org seeded by V5 — always present, satisfies projects.org_id NOT NULL. */
    private static final UUID DEFAULT_ORG = UUID.fromString("00000000-0000-0000-0000-000000000001");

    @Autowired
    JdbcTemplate jdbc;

    @Autowired
    RunRepository runs;

    // -----------------------------------------------------------------------
    // 1) Schema: the converted columns are timestamptz
    // -----------------------------------------------------------------------

    @Test
    void runTimestampColumnsAreTimestamptz() {
        assertThat(columnType("runs", "created_at")).isEqualTo("timestamp with time zone");
        assertThat(columnType("runs", "started_at")).isEqualTo("timestamp with time zone");
        assertThat(columnType("runs", "ended_at")).isEqualTo("timestamp with time zone");
        assertThat(columnType("run_enrollments", "enrolled_at")).isEqualTo("timestamp with time zone");
        assertThat(columnType("run_enrollments", "token_issued_at")).isEqualTo("timestamp with time zone");
    }

    // -----------------------------------------------------------------------
    // 2) Behaviour: an Instant round-trips with no timezone drift
    // -----------------------------------------------------------------------

    @Test
    void instantRoundTripsThroughJpaWithoutDrift() {
        UUID projectId = insertProject();

        // Microsecond precision: Postgres timestamptz truncates to micros, so a
        // micro-aligned Instant round-trips byte-for-byte.
        Instant created = Instant.parse("2026-03-15T14:22:33.123456Z");
        Instant started = Instant.parse("2026-03-15T14:25:00.000001Z");
        Instant ended = Instant.parse("2026-03-15T15:00:59.999999Z");

        Run run = new Run();
        run.setProjectId(projectId);
        run.setStrategy("FEDAVG");
        run.setNumRounds(3);
        run.setMinClients(1);
        run.setClientsPerRound(1);
        run.setPartitioningMode(PartitioningMode.SHARDED);
        run.setStatus(RunStatus.PENDING);
        run.setRecipeKey("CNN");
        run.setCreatedAt(created);
        run.setStartedAt(started);
        run.setEndedAt(ended);

        UUID id = runs.saveAndFlush(run).getId();
        Run reloaded = fetchFresh(id);

        assertThat(reloaded.getCreatedAt()).isEqualTo(created);
        assertThat(reloaded.getStartedAt()).isEqualTo(started);
        assertThat(reloaded.getEndedAt()).isEqualTo(ended);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    private Run fetchFresh(UUID id) {
        // A separate repository call runs in its own transaction, so the row is
        // re-materialised from PostgreSQL rather than served from the persistence
        // context of the save above.
        return runs.findById(id).orElseThrow();
    }

    private UUID insertProject() {
        UUID id = UUID.randomUUID();
        jdbc.update(
                "INSERT INTO projects (id, name, model_type, model_name, status, org_id) " +
                        "VALUES (?, ?, ?, ?, ?, ?)",
                id, "rt-" + id, "CNN", "cnn", "CREATED", DEFAULT_ORG);
        return id;
    }

    private String columnType(String table, String column) {
        return jdbc.queryForObject(
                "SELECT data_type FROM information_schema.columns " +
                        "WHERE table_name = ? AND column_name = ?",
                String.class, table, column);
    }
}
