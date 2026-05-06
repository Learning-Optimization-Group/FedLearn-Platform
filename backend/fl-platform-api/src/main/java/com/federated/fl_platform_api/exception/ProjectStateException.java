package com.federated.fl_platform_api.exception;

/**
 * Thrown when an action is attempted against a project in an incompatible state
 * (e.g. starting a server that is already running, stopping one that isn't).
 * Mapped to HTTP 409 Conflict by GlobalExceptionHandler.
 */
public class ProjectStateException extends RuntimeException {

    public ProjectStateException(String message) {
        super(message);
    }
}
