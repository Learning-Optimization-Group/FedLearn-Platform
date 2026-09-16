package com.federated.fl_platform_api.dto;

/**
 * Body of {@code POST /api/internal/runs/{projectId}/{runId}/verify-connection-token}: the
 * connection token a joining client presented to the FL server, which the FL server hands back
 * to the backend to authenticate. Intentionally minimal — the projectId/runId being asserted
 * travel in the path (and are scoped by the internal filter), not in the body.
 */
public class VerifyConnectionTokenRequest {

    private String connectionToken;

    public String getConnectionToken() {
        return connectionToken;
    }

    public void setConnectionToken(String connectionToken) {
        this.connectionToken = connectionToken;
    }
}
