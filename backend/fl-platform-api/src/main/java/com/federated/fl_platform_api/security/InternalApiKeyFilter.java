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

    private final String expectedKey;

    public InternalApiKeyFilter(@Value("${app.internal.api-key:}") String expectedKey) {
        this.expectedKey = expectedKey == null ? "" : expectedKey.trim();
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

        chain.doFilter(request, response);
    }

    private static boolean constantTimeEquals(String a, String b) {
        byte[] ab = a.getBytes(StandardCharsets.UTF_8);
        byte[] bb = b.getBytes(StandardCharsets.UTF_8);
        return MessageDigest.isEqual(ab, bb);
    }
}
