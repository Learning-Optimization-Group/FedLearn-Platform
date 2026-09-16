package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.BenchmarkRun;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface BenchmarkRunRepository extends JpaRepository<BenchmarkRun, UUID> {

    Optional<BenchmarkRun> findByProjectId(UUID projectId);

    /** Most-recently-active benchmarked projects first (dashboard table order). */
    List<BenchmarkRun> findAllByOrderByLastRecordedAtDesc();
}
