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

    @Query("""
        SELECT e FROM AuditEvent e
        WHERE (:orgId    IS NULL OR e.orgId       = :orgId)
          AND (:actorId  IS NULL OR e.actorUserId = :actorId)
          AND (:action   IS NULL OR e.action      = :action)
          AND (:from     IS NULL OR e.occurredAt  >= :from)
          AND (:to       IS NULL OR e.occurredAt  <  :to)
        ORDER BY e.occurredAt DESC
        """)
    Page<AuditEvent> search(
            @Param("orgId")   UUID orgId,
            @Param("actorId") Long actorId,
            @Param("action")  AuditAction action,
            @Param("from")    Instant from,
            @Param("to")      Instant to,
            Pageable page);
}
