package com.federated.fl_platform_api.audit;

import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletRequest;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.stereotype.Component;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import java.lang.annotation.Annotation;
import java.lang.reflect.Method;
import java.util.Map;
import java.util.UUID;

/**
 * Persists an {@link AuditEvent} after every successful invocation of a method
 * annotated with {@link Auditable @Auditable}.
 *
 * <p><b>Transactional semantics:</b> {@link #record(ProceedingJoinPoint)} proceeds
 * <i>first</i> then writes the audit row. If the caller is {@code @Transactional},
 * both the caller's mutation and the audit row commit (or roll back) together. If
 * there is no transaction, the two writes auto-commit independently.
 *
 * <p><b>Actor resolution:</b> Spring's principal in this codebase is a
 * {@link UserDetails}; the application's {@code User.id} is a {@code Long}. This
 * aspect looks up the {@link User} via {@code UserRepository.findByUsername} and
 * stores the numeric id on the audit row.
 */
@Aspect
@Component
public class AuditAspect {

    private static final Logger log = LoggerFactory.getLogger(AuditAspect.class);

    private final AuditEventRepository repo;
    private final UserRepository users;
    private final ObjectMapper objectMapper;

    public AuditAspect(AuditEventRepository repo, UserRepository users, ObjectMapper objectMapper) {
        this.repo = repo;
        this.users = users;
        this.objectMapper = objectMapper;
    }

    @Around("@annotation(com.federated.fl_platform_api.audit.Auditable)")
    public Object record(ProceedingJoinPoint pjp) throws Throwable {
        Object result;
        try {
            result = pjp.proceed();         // run first; audit only after success
        } catch (Throwable t) {
            // No audit row on failure. Drain to clear any metadata the method
            // staged before throwing, so it can't leak onto the next request
            // that reuses this pooled thread.
            AuditContext.drain();
            throw t;
        }

        MethodSignature sig = (MethodSignature) pjp.getSignature();
        Method method = sig.getMethod();
        Auditable a = method.getAnnotation(Auditable.class);

        Object[] args = pjp.getArgs();
        String[] paramNames = sig.getParameterNames();
        Annotation[][] paramAnnotations = method.getParameterAnnotations();

        String targetId = resolveTargetId(a.targetIdParam(), paramNames, args);
        UUID orgId      = resolveOrgId(paramAnnotations, args);
        Long actor      = resolveActor();
        String ip       = resolveIp();
        String ua       = resolveUserAgent();
        String meta     = serialise(AuditContext.drain());

        AuditEvent.Builder b = AuditEvent.builder()
                .action(a.action())
                .actorUserId(actor)
                .orgId(orgId)
                .targetType(a.targetType().isBlank() ? null : a.targetType())
                .targetId(targetId)
                .metadata(meta)
                .requestIp(ip)
                .userAgent(ua);

        repo.save(b.build());
        return result;
    }

    private static String resolveTargetId(String paramName, String[] names, Object[] args) {
        if (paramName == null || paramName.isBlank() || names == null) return null;
        for (int i = 0; i < names.length; i++) {
            if (paramName.equals(names[i]) && args[i] != null) return args[i].toString();
        }
        return null;
    }

    private static UUID resolveOrgId(Annotation[][] paramAnnotations, Object[] args) {
        for (int i = 0; i < paramAnnotations.length; i++) {
            for (Annotation an : paramAnnotations[i]) {
                if (an.annotationType() == CurrentOrg.class && args[i] instanceof UUID u) return u;
            }
        }
        return null;
    }

    private Long resolveActor() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated()) return null;
        Object p = auth.getPrincipal();
        if (p instanceof UserDetails ud) {
            return users.findByUsername(ud.getUsername())
                    .map(User::getId)
                    .orElse(null);
        }
        return null;
    }

    private static String resolveIp() {
        HttpServletRequest req = currentRequest();
        return req == null ? null : req.getRemoteAddr();
    }

    private static String resolveUserAgent() {
        HttpServletRequest req = currentRequest();
        return req == null ? null : req.getHeader("User-Agent");
    }

    private static HttpServletRequest currentRequest() {
        try {
            ServletRequestAttributes attrs =
                    (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            return attrs == null ? null : attrs.getRequest();
        } catch (IllegalStateException e) { return null; }
    }

    /**
     * Serialises drained audit metadata to a JSON object string via Jackson, so
     * keys and values are correctly escaped (backslashes, newlines, control
     * chars, embedded quotes). Returns {@code null} for an empty map.
     *
     * <p>Correctness matters because {@code audit_events.metadata} is JSONB (V6):
     * an invalid string would fail the insert and — since the aspect writes in
     * the same transaction as the audited mutation — roll the mutation back. On
     * the (now unlikely) serialisation failure we fall back to {@code null} and
     * log a warning, so the audit row still persists and the mutation is never
     * sacrificed for an unserialisable metadata blob.
     */
    private String serialise(Map<String, String> ctx) {
        if (ctx.isEmpty()) return null;
        try {
            return objectMapper.writeValueAsString(ctx);
        } catch (JsonProcessingException e) {
            log.warn("Failed to serialise audit metadata to JSON; persisting null metadata", e);
            return null;
        }
    }
}
