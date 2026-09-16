package com.federated.fl_platform_api.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.lang.NonNull;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

/**
 * Rejects oversized inference request bodies <em>before</em> the dispatcher reads
 * them into memory, capping the heap pressure a single caller can create.
 *
 * <p>The {@code @Size} bean-validation on {@code InferenceRequest.imageBase64}
 * only fires <em>after</em> Jackson has already materialised the whole body as a
 * String, so it cannot prevent a parse-time OOM on the direct-to-Spring path
 * (desktop/local, where there is no nginx {@code client_max_body_size} in front).
 * This Content-Length pre-check closes that gap.
 *
 * <p>Limitation: chunked requests with no {@code Content-Length} cannot be
 * pre-checked here; those are still bounded by {@code @Size} validation and the
 * inference concurrency semaphore. Scoped strictly to {@code POST /api/inference/*}
 * so no other endpoint's body limits change.
 */
@Component
public class InferenceRequestSizeFilter extends OncePerRequestFilter {

    private final long maxRequestBytes;

    public InferenceRequestSizeFilter(
            @Value("${inference.max-request-bytes:16777216}") long maxRequestBytes) {
        this.maxRequestBytes = maxRequestBytes;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        // Only inspect inference POSTs; everything else passes straight through.
        return !("POST".equalsIgnoreCase(request.getMethod())
                && request.getRequestURI().startsWith("/api/inference/"));
    }

    @Override
    protected void doFilterInternal(@NonNull HttpServletRequest request,
                                    @NonNull HttpServletResponse response,
                                    @NonNull FilterChain filterChain)
            throws ServletException, IOException {

        long declared = request.getContentLengthLong();
        if (declared > maxRequestBytes) {
            response.setStatus(HttpStatus.PAYLOAD_TOO_LARGE.value());
            response.setContentType(MediaType.APPLICATION_JSON_VALUE);
            response.getWriter().write(
                    "{\"status\":413,\"error\":\"Payload Too Large\","
                            + "\"message\":\"Inference request body exceeds the maximum allowed size.\"}");
            return;
        }
        filterChain.doFilter(request, response);
    }
}
