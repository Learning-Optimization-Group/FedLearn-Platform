package com.federated.fl_platform_api.exception;

/**
 * Thrown when an external process (FL server spawn, model initializer script,
 * ECS RunTask) fails. Mapped to HTTP 502 Bad Gateway by GlobalExceptionHandler
 * since it represents an upstream/dependency failure rather than a client error.
 */
public class ServerProcessException extends RuntimeException {

    public ServerProcessException(String message) {
        super(message);
    }

    public ServerProcessException(String message, Throwable cause) {
        super(message, cause);
    }
}
