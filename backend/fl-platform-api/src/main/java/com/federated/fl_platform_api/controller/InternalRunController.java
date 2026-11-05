package com.federated.fl_platform_api.controller;

import com.federated.fl_platform_api.dto.ConnectionTokenVerificationDto;
import com.federated.fl_platform_api.dto.VerifyConnectionTokenRequest;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import com.federated.fl_platform_api.service.RunConnectionTokenVerifier;
import java.util.UUID;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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
 * <p>The service also binds server authorization to the path run and holds current
 * Project/Run authorization locks throughout verification. Inactive executions are refused.
 *
 * <p>Status contract:
 * <ul>
 *   <li><b>200</b> — the token verifies AND its {@code runId}/{@code projectId} match the path.
 *       Body is the resolved enrolled identity.</li>
 *   <li><b>401</b> — the token is not authentic: bad/forged signature, expired, malformed, wrong
 *       audience, or missing/incoherent identity claims. The bearer proved nothing.</li>
 *   <li><b>403</b> — the token is authentic but was minted for a DIFFERENT run or project than the
 *       path asserts, or the server authorization has been retired.</li>
 *   <li><b>409</b> — the server execution is no longer active.</li>
 * </ul>
 */
@RestController
@RequestMapping("/api/internal/runs")
public class InternalRunController {
    private final RunConnectionTokenVerifier verifier;

    public InternalRunController(RunConnectionTokenVerifier verifier) {
        this.verifier = verifier;
    }

    @PostMapping("/{projectId}/{runId}/verify-connection-token")
    public ResponseEntity<ConnectionTokenVerificationDto> verifyConnectionToken(
            @PathVariable UUID projectId,
            @PathVariable UUID runId,
            @RequestAttribute(InternalCallbackContext.REQUEST_ATTRIBUTE) InternalCallbackContext context,
            @RequestBody(required = false) VerifyConnectionTokenRequest request) {
        return ResponseEntity.ok(verifier.verify(projectId, runId, context,
                request == null ? null : request.getConnectionToken()));
    }
}
