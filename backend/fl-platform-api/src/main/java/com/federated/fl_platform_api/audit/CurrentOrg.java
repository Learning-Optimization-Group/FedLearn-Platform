package com.federated.fl_platform_api.audit;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks the parameter on an {@link Auditable @Auditable} method whose value is the
 * organisation UUID this audit row should be filed under. {@link AuditAspect} reads
 * this parameter and writes it to {@code audit_events.org_id}.
 */
@Target(ElementType.PARAMETER)
@Retention(RetentionPolicy.RUNTIME)
public @interface CurrentOrg { }
