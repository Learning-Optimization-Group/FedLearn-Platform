package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.AuditAction;
import com.federated.fl_platform_api.model.AuditEvent;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.Instant;
import java.util.UUID;

public interface AuditEventRepository extends JpaRepository<AuditEvent, UUID> {

    /**
     * {@code from} (inclusive) and {@code to} (exclusive) must be non-null —
     * callers pass {@code Instant.EPOCH} / a far-future sentinel for an open
     * end. A nullable {@code Instant} cannot be bound through the usual
     * {@code (:from IS NULL OR ...)} guard: Postgres cannot infer a type for
     * the bare null parameter and rejects the statement.
     */
    @Query("""
        SELECT e FROM AuditEvent e
        WHERE (:orgId      IS NULL OR e.orgId       = :orgId)
          AND (:actorId    IS NULL OR e.actorUserId = :actorId)
          AND (:action     IS NULL OR e.action      = :action)
          AND (:targetType IS NULL OR e.targetType  = :targetType)
          AND e.occurredAt >= :from
          AND e.occurredAt <  :to
        ORDER BY e.occurredAt DESC
        """)
    Page<AuditEvent> search(
            @Param("orgId")      UUID orgId,
            @Param("actorId")    Long actorId,
            @Param("action")     AuditAction action,
            @Param("targetType") String targetType,
            @Param("from")       Instant from,
            @Param("to")         Instant to,
            Pageable page);
}
