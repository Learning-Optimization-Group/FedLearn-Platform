package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.model.PartitioningMode;
import com.federated.fl_platform_api.model.Run;
import com.federated.fl_platform_api.model.RunStatus;
import com.federated.fl_platform_api.repository.RunRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.ActiveProfiles;

import java.time.Instant;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ActiveProfiles("test")
class RunEntityTest {

    @Autowired
    RunRepository repo;

    @Test
    void persists_run_and_round_trips_all_not_null_fields() {
        UUID projectId = UUID.randomUUID();
        Instant now = Instant.now();

        Run run = new Run();
        run.setProjectId(projectId);
        run.setStrategy("FedAvg");
        run.setNumRounds(10);
        run.setMinClients(2);
        run.setClientsPerRound(3);
        run.setPartitioningMode(PartitioningMode.SHARDED);
        run.setStatus(RunStatus.RUNNING);
        run.setRecipeKey("CNN");
        run.setCreatedAt(now);
        run.setSeed(42L);

        repo.saveAndFlush(run);
        UUID id = run.getId();
        assertThat(id).isNotNull();

        // Reload from DB
        Run found = repo.findById(id).orElseThrow();
        assertThat(found.getProjectId()).isEqualTo(projectId);
        assertThat(found.getStrategy()).isEqualTo("FedAvg");
        assertThat(found.getNumRounds()).isEqualTo(10);
        assertThat(found.getMinClients()).isEqualTo(2);
        assertThat(found.getClientsPerRound()).isEqualTo(3);
        assertThat(found.getPartitioningMode()).isEqualTo(PartitioningMode.SHARDED);
        assertThat(found.getStatus()).isEqualTo(RunStatus.RUNNING);
        assertThat(found.getRecipeKey()).isEqualTo("CNN");
        assertThat(found.getSeed()).isEqualTo(42L);
        assertThat(found.getCreatedAt()).isNotNull();
    }

    @Test
    void enum_fields_survive_string_roundtrip() {
        Run run = new Run();
        run.setProjectId(UUID.randomUUID());
        run.setStrategy("DeComFL");
        run.setNumRounds(5);
        run.setMinClients(1);
        run.setClientsPerRound(1);
        run.setPartitioningMode(PartitioningMode.LOCAL);
        run.setStatus(RunStatus.COMPLETED);
        run.setRecipeKey("MLP");
        run.setCreatedAt(Instant.now());

        repo.saveAndFlush(run);

        Run found = repo.findById(run.getId()).orElseThrow();
        assertThat(found.getPartitioningMode()).isEqualTo(PartitioningMode.LOCAL);
        assertThat(found.getStatus()).isEqualTo(RunStatus.COMPLETED);
    }
}
