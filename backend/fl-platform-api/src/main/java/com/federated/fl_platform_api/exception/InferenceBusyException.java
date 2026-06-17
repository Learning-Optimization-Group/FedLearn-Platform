package com.federated.fl_platform_api.exception;

/**
 * Thrown when the inference subsystem is at capacity (all concurrent-process
 * permits are in use). Mapped to HTTP 503 Service Unavailable with a
 * {@code Retry-After} hint by GlobalExceptionHandler — it is a transient
 * capacity signal, not a client error, so the caller should simply retry.
 */
public class InferenceBusyException extends RuntimeException {

    public InferenceBusyException(String message) {
        super(message);
    }
}
