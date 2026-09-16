package com.federated.fl_platform_api.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.lang.NonNull;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Optional;
import java.util.UUID;

/**
 * Gates /api/internal/** endpoints (FL-server → backend callbacks) behind a
 * shared-secret header. Fails closed: if APP_INTERNAL_API_KEY is unset or the
 * request header is missing/mismatched, the request is rejected with 401.
 *
 * This is intentionally *before* the JWT filter in the chain so internal
 * service-to-service traffic never needs a user token.
 */
@Component
public class InternalApiKeyFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(InternalApiKeyFilter.class);
    private static final String INTERNAL_PATH_PREFIX = "/api/internal/";
    private static final String HEADER_NAME = "X-Internal-Key";
    // SE-7: scoped per-run token — binds a callback to exactly one project.
    private static final String RUN_TOKEN_HEADER = "X-Internal-Run-Token";

    private final String expectedKey;
    private final RunTokenRegistry runTokenRegistry;

    public InternalApiKeyFilter(@Value("${app.internal.api-key:}") String expectedKey,
                                RunTokenRegistry runTokenRegistry) {
        this.expectedKey = expectedKey == null ? "" : expectedKey.trim();
        this.runTokenRegistry = runTokenRegistry;
    }

    @Override
    protected void doFilterInternal(@NonNull HttpServletRequest request,
                                    @NonNull HttpServletResponse response,
                                    @NonNull FilterChain chain) throws ServletException, IOException {

        String path = request.getRequestURI();
        if (path == null || !path.startsWith(INTERNAL_PATH_PREFIX)) {
            chain.doFilter(request, response);
            return;
        }

        if (expectedKey.isEmpty()) {
            log.error("Rejecting {} {}: app.internal.api-key is not configured.",
                    request.getMethod(), path);
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED,
                    "Internal API key not configured on server");
            return;
        }

        String provided = request.getHeader(HEADER_NAME);
        if (provided == null || !constantTimeEquals(provided, expectedKey)) {
            log.warn("Rejected internal call to {} from {}: {} header missing or invalid.",
                    path, request.getRemoteAddr(), HEADER_NAME);
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Invalid internal API key");
            return;
        }

        // SE-7: the static key above is the outer gate; the scoped run token binds this callback to
        // ONE project so a leaked/compromised run token cannot mutate any other project. Every
        // /api/internal/** endpoint carries its target project id as the 5th path segment.
        UUID targetProject = extractProjectId(path);
        Optional<RunTokenRegistry.Scope> scope = runTokenRegistry.resolve(request.getHeader(RUN_TOKEN_HEADER));
        if (targetProject == null || scope.isEmpty()) {
            log.warn("Rejected internal call to {} from {}: missing/unknown run token or non-project path.",
                    path, request.getRemoteAddr());
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Missing or invalid run token");
            return;
        }
        if (!scope.get().projectId().equals(targetProject)) {
            log.warn("Rejected internal call to {} from {}: run token scoped to project {} cannot act on {}.",
                    path, request.getRemoteAddr(), scope.get().projectId(), targetProject);
            response.sendError(HttpServletResponse.SC_FORBIDDEN, "Run token not authorized for this project");
            return;
        }

        chain.doFilter(request, response);
    }

    /**
     * The target project id of an {@code /api/internal/**} call — always the 5th path segment:
     * {@code /api/internal/{results|benchmarks}/{projectId}[/...]} and
     * {@code /api/internal/projects/{projectId}/artifacts}. Returns null if absent or not a UUID.
     */
    private static UUID extractProjectId(String path) {
        String[] seg = path.split("/");   // ["", "api", "internal", "<resource>", "<projectId>", ...]
        if (seg.length < 5) {
            return null;
        }
        try {
            return UUID.fromString(seg[4]);
        } catch (IllegalArgumentException e) {
            return null;
        }
    }

    private static boolean constantTimeEquals(String a, String b) {
        byte[] ab = a.getBytes(StandardCharsets.UTF_8);
        byte[] bb = b.getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(ab, bb);
    }
}
