package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.ConnectionTokenVerificationDto;
import com.federated.fl_platform_api.dto.VerifyConnectionTokenRequest;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.UUID;

/**
 * BA-6 — the FL-boundary enforcement endpoint. When a client opens a gRPC session, the FL server
 * relays the client's connection token here to have the backend (the token's issuer) authenticate
 * it and resolve the enrolled identity, rather than trusting anything asserted on the wire.
 *
 * <p>Lives under {@code /api/internal/**}, so it is already gated by
 * {@link com.federated.fl_platform_api.security.InternalApiKeyFilter}: the static {@code X-Internal-Key}
 * plus a per-run {@code X-Internal-Run-Token} whose project scope must equal the {@code projectId}
 * path segment (the 5th segment the filter reads). The endpoint therefore carries {@code projectId}
 * ahead of {@code runId} so that gate applies unchanged — no SecurityConfig change is needed.
 *
 * <p>Status contract:
 * <ul>
 *   <li><b>200</b> — the token verifies AND its {@code runId}/{@code projectId} match the path.
 *       Body is the resolved enrolled identity.</li>
 *   <li><b>401</b> — the token is not authentic: bad/forged signature, expired, malformed, wrong
 *       audience, or missing/incoherent identity claims. The bearer proved nothing.</li>
 *   <li><b>403</b> — the token is authentic but was minted for a DIFFERENT run or project than the
 *       path asserts. A genuine credential presented against the wrong resource.</li>
 * </ul>
 */
@RestController
@RequestMapping("/api/internal/runs")
public class InternalRunController {

    private static final Logger log = LoggerFactory.getLogger(InternalRunController.class);

    private final ConnectionTokenService tokenService;

    public InternalRunController(ConnectionTokenService tokenService) {
        this.tokenService = tokenService;
    }

    @PostMapping("/{projectId}/{runId}/verify-connection-token")
    public ResponseEntity<ConnectionTokenVerificationDto> verifyConnectionToken(
            @PathVariable UUID projectId,
            @PathVariable UUID runId,
            @RequestBody(required = false) VerifyConnectionTokenRequest request) {

        String token = request == null ? null : request.getConnectionToken();

        // 1. Authenticity: signature + expiry + audience. verify() throws on any of these; a
        //    null/blank token throws IllegalArgumentException. Either way the token is not
        //    authentic → 401. Never let the exception reach the 500 catch-all handler.
        Claims claims;
        try {
            claims = tokenService.verify(token);
        } catch (JwtException | IllegalArgumentException e) {
            log.warn("Rejected connection token for run {} (project {}): not authentic — {}",
                    runId, projectId, e.getClass().getSimpleName());
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        // 2. Decode the enrolled identity. A signed-but-incoherent token (missing/garbled
        //    identity claims) is not a well-formed connection token → 401.
        ConnectionTokenService.Claims decoded;
        try {
            decoded = decode(claims);
        } catch (RuntimeException e) {
            log.warn("Rejected connection token for run {} (project {}): unreadable claims — {}",
                    runId, projectId, e.getClass().getSimpleName());
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).build();
        }

        // 3. Scope: an authentic token minted for another run/project must not authenticate here.
        if (!runId.equals(decoded.runId()) || !projectId.equals(decoded.projectId())) {
            log.warn("Rejected connection token: authentic but scoped to run {}/project {}, "
                            + "presented against run {}/project {}",
                    decoded.runId(), decoded.projectId(), runId, projectId);
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }

        ConnectionTokenVerificationDto dto = new ConnectionTokenVerificationDto();
        dto.setUserId(decoded.userId());
        dto.setRunId(decoded.runId());
        dto.setProjectId(decoded.projectId());
        dto.setPartitionId(decoded.partitionId());
        dto.setClientKind(decoded.clientKind());
        dto.setGrpcEndpoint(decoded.grpcEndpoint());
        return ResponseEntity.ok(dto);
    }

    /**
     * Rebuild the typed claims from the verified JWT payload, mirroring how
     * {@link ConnectionTokenService#mint} wrote them. Throws (→ 401) if a required identity claim
     * is absent or malformed, or if the identity fields are not internally coherent.
     */
    private static ConnectionTokenService.Claims decode(Claims claims) {
        Long userId = Long.valueOf(claims.getSubject());
        UUID runId = UUID.fromString(claims.get("runId", String.class));
        UUID projectId = UUID.fromString(claims.get("projectId", String.class));
        Integer partitionId = claims.get("partitionId", Integer.class);
        String clientKind = claims.get("clientKind", String.class);
        String grpcEndpoint = claims.get("grpcEndpoint", String.class);
        String caFingerprint = claims.get("caFingerprint", String.class);

        if (partitionId == null || partitionId < 0 || clientKind == null || clientKind.isBlank()) {
            throw new IllegalArgumentException("incoherent enrolled identity in connection token");
        }
        return new ConnectionTokenService.Claims(
                userId, runId, projectId, partitionId, grpcEndpoint, caFingerprint, clientKind);
    }
}
