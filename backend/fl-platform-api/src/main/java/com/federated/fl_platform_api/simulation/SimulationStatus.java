package com.federated.fl_platform_api.simulation;

public enum SimulationStatus {
    ACCEPTED, STARTING, RUNNING, STOPPING, COMPLETED, STOPPED, FAILED, TIMED_OUT, INTERRUPTED;

    public boolean terminal() {
        return switch (this) {
            case COMPLETED, STOPPED, FAILED, TIMED_OUT, INTERRUPTED -> true;
            default -> false;
        };
    }
}
