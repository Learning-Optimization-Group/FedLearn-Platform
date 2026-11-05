package com.federated.fl_platform_api.security;

import java.util.UUID;

/** Verified callback identity supplied only by the internal authentication filter. */
public record InternalCallbackContext(UUID projectId, UUID runId, String tokenHash) {
    public static final String REQUEST_ATTRIBUTE = "com.federated.fl_platform_api.security.InternalCallbackContext";
}
