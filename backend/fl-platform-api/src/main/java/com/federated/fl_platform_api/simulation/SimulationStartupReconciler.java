package com.federated.fl_platform_api.simulation;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

@Component
public class SimulationStartupReconciler implements ApplicationRunner {
    private final SimulationService service;
    private final SimulationProperties properties;

    public SimulationStartupReconciler(SimulationService service, SimulationProperties properties) {
        this.service = service; this.properties = properties;
    }

    @Override public void run(ApplicationArguments arguments) {
        if (properties.isEnabled() && properties.isReconcileOnStartup()) service.reconcile();
    }
}
