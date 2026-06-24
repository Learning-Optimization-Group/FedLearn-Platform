package com.federated.fl_platform_api.exception;

import java.util.UUID;

/**
 * Thrown when a requested entity (project, user, result, log, etc.) does not exist.
 * Mapped to HTTP 404 by GlobalExceptionHandler.
 */
public class ResourceNotFoundException extends RuntimeException {

    public ResourceNotFoundException(String message) {
        super(message);
    }

    public static ResourceNotFoundException forEntity(String entity, Object id) {
        return new ResourceNotFoundException(entity + " not found with id: " + id);
    }

    public static ResourceNotFoundException project(UUID id) {
        return forEntity("Project", id);
    }

    public static ResourceNotFoundException run(UUID id) {
        return forEntity("Run", id);
    }
}
