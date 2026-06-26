package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.BenchmarkRound;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface BenchmarkRoundRepository extends JpaRepository<BenchmarkRound, UUID> {

    /** Idempotent ingest: a re-reported round upserts the same row. */
    Optional<BenchmarkRound> findByProjectIdAndServerRound(UUID projectId, Integer serverRound);

    /** Full per-round series for one project, ordered for charting. */
    List<BenchmarkRound> findByProjectIdOrderByServerRoundAsc(UUID projectId);

    long countByProjectId(UUID projectId);
}
