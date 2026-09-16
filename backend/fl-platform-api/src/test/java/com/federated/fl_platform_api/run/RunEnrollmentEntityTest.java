package com.federated.fl_platform_api.run;

import com.federated.fl_platform_api.model.ClientKind;
import com.federated.fl_platform_api.model.RunEnrollment;
import com.federated.fl_platform_api.model.RunEnrollmentId;
import com.federated.fl_platform_api.repository.RunEnrollmentRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.jdbc.AutoConfigureTestDatabase;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.ActiveProfiles;

import java.time.Instant;
import java.util.Optional;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

@DataJpaTest
@AutoConfigureTestDatabase(replace = AutoConfigureTestDatabase.Replace.NONE)
@ActiveProfiles("test")
class RunEnrollmentEntityTest {

    @Autowired
    RunEnrollmentRepository repo;

    @Test
    void persists_two_enrollments_and_round_trips_repository_queries() {
        UUID runId = UUID.randomUUID();
        Instant now = Instant.now();

        RunEnrollment e1 = new RunEnrollment(
                new RunEnrollmentId(runId, 10L), 0, ClientKind.SHARD, now);
        RunEnrollment e2 = new RunEnrollment(
                new RunEnrollmentId(runId, 20L), 1, ClientKind.SHARD, now);

        repo.saveAndFlush(e1);
        repo.saveAndFlush(e2);

        // findByIdRunIdAndIdUserId — present, correct partitionId
        Optional<RunEnrollment> found = repo.findByIdRunIdAndIdUserId(runId, 10L);
        assertThat(found).isPresent();
        assertThat(found.get().getPartitionId()).isEqualTo(0);

        // maxPartitionIdForRun — returns the max (1)
        assertThat(repo.maxPartitionIdForRun(runId)).isEqualTo(1);

        // maxPartitionIdForRun on unknown runId — COALESCE returns -1
        assertThat(repo.maxPartitionIdForRun(UUID.randomUUID())).isEqualTo(-1);

        // countByIdRunId — two rows
        assertThat(repo.countByIdRunId(runId)).isEqualTo(2L);
    }
}
