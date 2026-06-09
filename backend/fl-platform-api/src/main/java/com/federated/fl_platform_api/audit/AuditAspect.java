package com.federated.fl_platform_api.audit;

import com.federated.fl_platform_api.model.AuditEvent;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.AuditEventRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import jakarta.servlet.http.HttpServletRequest;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.reflect.MethodSignature;
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
import java.util.stream.Collectors;

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

    private final AuditEventRepository repo;
    private final UserRepository users;

    public AuditAspect(AuditEventRepository repo, UserRepository users) {
        this.repo = repo;
        this.users = users;
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

    private static String serialise(Map<String, String> ctx) {
        if (ctx.isEmpty()) return null;
        return ctx.entrySet().stream()
                .map(e -> "\"" + e.getKey() + "\":\"" + e.getValue().replace("\"", "\\\"") + "\"")
                .collect(Collectors.joining(",", "{", "}"));
    }
}
