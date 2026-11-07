package com.federated.fl_platform_api.run;

import java.util.UUID;
import org.flywaydb.core.Flyway;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.junit.jupiter.api.Assertions.*;

@Testcontainers
class V37RoundResultTimingMigrationTest {
    @Container
    static final PostgreSQLContainer<?> POSTGRES = new PostgreSQLContainer<>("postgres:16.6-alpine");

    @Test
    void upgradeAddsTimingColumnsToRoundResult() {
        var source = new DriverManagerDataSource(POSTGRES.getJdbcUrl(), POSTGRES.getUsername(), POSTGRES.getPassword());
        Flyway.configure().dataSource(source).target("36").load().migrate();
        var jdbc = new JdbcTemplate(source);
        UUID project = UUID.randomUUID(), runId = UUID.randomUUID(), resultId = UUID.randomUUID();

        jdbc.update("INSERT INTO projects (id,name,model_type,model_name,status,org_id,visibility) "
                + "VALUES (?,?,'CNN','net','RUNNING','00000000-0000-0000-0000-000000000001','PRIVATE')",
                project, "timing-migration-" + project);
        jdbc.update("INSERT INTO runs (id,project_id,strategy,num_rounds,min_clients,clients_per_round,status,recipe_key,secure_aggregation,created_at) "
                + "VALUES (?,?,'FedAvg',5,1,1,'RUNNING','CNN',false,CURRENT_TIMESTAMP)", runId, project);

        // Run V37 migration
        Flyway.configure().dataSource(source).load().migrate();

        // Insert round_result with timing breakdown
        jdbc.update("INSERT INTO round_result (id,project_id,run_id,server_round,loss,accuracy,gpu_utilization,"
                + "time_data_loading_s,time_forward_backward_s,time_optimizer_step_s,time_wire_encode_s,"
                + "time_aggregation_s,time_wire_decode_s,time_eval_s) "
                + "VALUES (?,?,?,?,0.45,0.85,0.0,0.12,0.34,0.05,0.02,0.08,0.01,0.15)",
                resultId, project, runId, 1);

        Double dataLoading = jdbc.queryForObject("SELECT time_data_loading_s FROM round_result WHERE id = ?", Double.class, resultId);
        Double forwardBackward = jdbc.queryForObject("SELECT time_forward_backward_s FROM round_result WHERE id = ?", Double.class, resultId);
        Double optStep = jdbc.queryForObject("SELECT time_optimizer_step_s FROM round_result WHERE id = ?", Double.class, resultId);
        Double wireEncode = jdbc.queryForObject("SELECT time_wire_encode_s FROM round_result WHERE id = ?", Double.class, resultId);
        Double aggregation = jdbc.queryForObject("SELECT time_aggregation_s FROM round_result WHERE id = ?", Double.class, resultId);
        Double wireDecode = jdbc.queryForObject("SELECT time_wire_decode_s FROM round_result WHERE id = ?", Double.class, resultId);
        Double eval = jdbc.queryForObject("SELECT time_eval_s FROM round_result WHERE id = ?", Double.class, resultId);

        assertEquals(0.12, dataLoading, 1e-4);
        assertEquals(0.34, forwardBackward, 1e-4);
        assertEquals(0.05, optStep, 1e-4);
        assertEquals(0.02, wireEncode, 1e-4);
        assertEquals(0.08, aggregation, 1e-4);
        assertEquals(0.01, wireDecode, 1e-4);
        assertEquals(0.15, eval, 1e-4);
    }
}
