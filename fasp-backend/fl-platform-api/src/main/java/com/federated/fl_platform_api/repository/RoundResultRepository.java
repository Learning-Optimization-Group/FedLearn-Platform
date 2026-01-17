package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.RoundResult;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface RoundResultRepository extends JpaRepository<RoundResult, UUID> {

    List<RoundResult> findByProjectIdOrderByServerRoundAsc(UUID projectId);
}
