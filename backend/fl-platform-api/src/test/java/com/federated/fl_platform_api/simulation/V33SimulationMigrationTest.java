package com.federated.fl_platform_api.simulation;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.TestPropertySource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@SpringBootTest
@ActiveProfiles("test")
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_simulation_migration",
        "spring.jpa.hibernate.ddl-auto=validate", "spring.flyway.enabled=true"
})
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
class V33SimulationMigrationTest {
    @Autowired JdbcTemplate jdbc;

    @Test void migrationCreatesIndependentJobsAndEnforcesOwnerForeignKey() {
        assertThat(jdbc.queryForObject("SELECT count(*) FROM information_schema.tables WHERE table_name='simulation_runs'", Integer.class))
                .isEqualTo(1);
        assertThatThrownBy(() -> jdbc.update("INSERT INTO simulation_runs(id,created_by,status,config_hash,recipe_json,created_at,requested_rounds) "
                + "VALUES(gen_random_uuid(),-9223372036854775807,'ACCEPTED',repeat('a',64),'{}',now(),2)"))
                .hasMessageContaining("fk_simulation_owner");
        assertThat(jdbc.queryForObject("SELECT count(*) FROM information_schema.columns WHERE table_name='simulation_runs' AND column_name='project_id'", Integer.class))
                .isZero();
    }
}
