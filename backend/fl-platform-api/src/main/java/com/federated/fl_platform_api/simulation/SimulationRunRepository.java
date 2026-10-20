package com.federated.fl_platform_api.simulation;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Collection;
import java.util.List;
import java.util.UUID;

public interface SimulationRunRepository extends JpaRepository<SimulationRun, UUID> {
    Page<SimulationRun> findByCreatedBy(Long createdBy, Pageable pageable);
    List<SimulationRun> findByStatusIn(Collection<SimulationStatus> statuses);
}
