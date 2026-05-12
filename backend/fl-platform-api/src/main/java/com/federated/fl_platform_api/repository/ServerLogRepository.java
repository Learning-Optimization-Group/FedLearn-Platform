package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.ServerLog;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.UUID;

public interface ServerLogRepository extends JpaRepository<ServerLog, Long> {

    /**
     * Returns a single page of logs in chronological order. The {@link Pageable}
     * argument enforces a hard upper bound on the returned row count — long-
     * running projects can produce millions of log rows, so the unbounded
     * variant was OOM-prone in production.
     */
    List<ServerLog> findByProjectIdOrderByTimestampAsc(UUID projectId, Pageable pageable);
}
