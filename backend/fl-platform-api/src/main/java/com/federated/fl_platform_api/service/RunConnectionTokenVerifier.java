package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.ConnectionTokenVerificationDto;
import com.federated.fl_platform_api.security.ConnectionTokenService;
import com.federated.fl_platform_api.security.InternalCallbackAuthorization;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/** Verify client identity while the presenting server's execution authorization remains locked. */
@Service
public class RunConnectionTokenVerifier {
    private static final Logger log = LoggerFactory.getLogger(RunConnectionTokenVerifier.class);
    private final ConnectionTokenService tokenService;
    private final InternalCallbackAuthorization authorization;

    public RunConnectionTokenVerifier(ConnectionTokenService tokenService, InternalCallbackAuthorization authorization) {
        this.tokenService = tokenService;
        this.authorization = authorization;
    }

    @Transactional
    public ConnectionTokenVerificationDto verify(UUID projectId, UUID runId,
            InternalCallbackContext context, String token) {
        if (context == null || !runId.equals(context.runId())) {
            throw new AccessDeniedException("Server authorization does not match the requested run");
        }
        authorization.requireActiveRun(projectId, context);

        // 1. Authenticity: signature + expiry + audience. verify() throws on any of these; a
        //    null/blank token throws IllegalArgumentException. Either way the token is not
        //    authentic → 401. Never let the exception reach the 500 catch-all handler.
        Claims claims;
        try {
            claims = tokenService.verify(token);
        } catch (JwtException | IllegalArgumentException e) {
            log.warn("Rejected connection token for run {} (project {}): not authentic — {}",
                    runId, projectId, e.getClass().getSimpleName());
            throw new BadCredentialsException("Invalid connection token");
        }

        // 2. Decode the enrolled identity. A signed-but-incoherent token (missing/garbled
        //    identity claims) is not a well-formed connection token → 401.
        ConnectionTokenService.Claims decoded;
        try {
            decoded = decode(claims);
        } catch (RuntimeException e) {
            log.warn("Rejected connection token for run {} (project {}): unreadable claims — {}",
                    runId, projectId, e.getClass().getSimpleName());
            throw new BadCredentialsException("Invalid connection token");
        }

        // 3. Scope: an authentic token minted for another run/project must not authenticate here.
        if (!runId.equals(decoded.runId()) || !projectId.equals(decoded.projectId())) {
            log.warn("Rejected connection token: authentic but scoped to run {}/project {}, "
                            + "presented against run {}/project {}",
                    decoded.runId(), decoded.projectId(), runId, projectId);
            throw new AccessDeniedException("Client connection token belongs to a different run or project");
        }

        ConnectionTokenVerificationDto dto = new ConnectionTokenVerificationDto();
        dto.setUserId(decoded.userId());
        dto.setRunId(decoded.runId());
        dto.setProjectId(decoded.projectId());
        dto.setPartitionId(decoded.partitionId());
        dto.setClientKind(decoded.clientKind());
        dto.setGrpcEndpoint(decoded.grpcEndpoint());
        return dto;
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
