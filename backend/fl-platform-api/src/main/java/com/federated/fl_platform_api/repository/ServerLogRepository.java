package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.ServerLog;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.UUID;

public interface ServerLogRepository extends JpaRepository<ServerLog, Long> {
    List<ServerLog> findByProjectIdOrderByTimestampAsc(UUID projectId);
}
