package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.BenchmarkRound;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface BenchmarkRoundRepository extends JpaRepository<BenchmarkRound, UUID> {

    Optional<BenchmarkRound> findByRunIdAndServerRound(UUID runId, Integer serverRound);

    List<BenchmarkRound> findByProjectIdAndRunIdOrderByServerRoundAsc(UUID projectId, UUID runId);

    /** Full per-round series for one project, ordered for charting. */
    List<BenchmarkRound> findByProjectIdOrderByServerRoundAsc(UUID projectId);

    long countByProjectId(UUID projectId);
}
