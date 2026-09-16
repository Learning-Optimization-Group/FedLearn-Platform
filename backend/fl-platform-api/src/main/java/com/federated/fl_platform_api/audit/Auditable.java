package com.federated.fl_platform_api.audit;

import com.federated.fl_platform_api.model.AuditAction;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method as audit-emitting. Successful invocations (no thrown exception)
 * cause {@link AuditAspect} to persist an {@link com.federated.fl_platform_api.model.AuditEvent}
 * after the method returns.
 *
 * <p>The aspect joins the caller's transaction, so a rolled-back caller also rolls back
 * the audit row. See {@link AuditAspect} for the full lifecycle.
 */
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Auditable {

    /** The action recorded on the {@link com.federated.fl_platform_api.model.AuditEvent}. */
    AuditAction action();

    /**
     * Name of the method parameter whose value should be serialised into the
     * {@code target_id} column. Empty means no target id is recorded.
     */
    String targetIdParam() default "";

    /** Value written to {@code target_type}. Empty means no target type recorded. */
    String targetType() default "";
}
