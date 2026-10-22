package com.federated.fl_platform_api.simulation;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.file.Path;

@ConfigurationProperties("app.simulation")
@Getter
@Setter
public class SimulationProperties {
    private boolean enabled;
    private Path interpreter;
    private Path sourceRoot;
    private Path cache;
    private String reportSha256;
    /** Optional capstone Effusion R0 bundle (train + development selection only); unset means the workload is unavailable. */
    private Path effusionBundle;
    private String effusionReportSha256;
    private Path storage;
    private int deadlineSeconds = 180;
    private boolean reconcileOnStartup = true;
}
