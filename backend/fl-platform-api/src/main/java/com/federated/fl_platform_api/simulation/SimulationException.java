package com.federated.fl_platform_api.simulation;

import org.springframework.http.HttpStatus;

final class SimulationException extends RuntimeException {
    final HttpStatus status;
    final String code;

    SimulationException(HttpStatus status, String code, String message) {
        super(message);
        this.status = status;
        this.code = code;
    }

    static SimulationException invalid() {
        return new SimulationException(HttpStatus.BAD_REQUEST, "INVALID_CHOICES", "Submit a supported choices object.");
    }

    static SimulationException unavailable() {
        return new SimulationException(HttpStatus.SERVICE_UNAVAILABLE, "WORKLOAD_UNAVAILABLE", "The qualified workload is unavailable.");
    }
}
