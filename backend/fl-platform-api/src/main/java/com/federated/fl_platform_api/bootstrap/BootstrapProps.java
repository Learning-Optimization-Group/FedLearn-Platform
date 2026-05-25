package com.federated.fl_platform_api.bootstrap;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration for the first-run platform-admin bootstrap.
 *
 * All four values are env-driven (see {@code application.properties}) and default
 * to empty strings (except {@code platformOrgName}, which defaults to "Platform").
 * If {@code adminEmail} is blank the {@link BootstrapRunner} no-ops at startup.
 */
@ConfigurationProperties("app.bootstrap")
public record BootstrapProps(
        String adminEmail,
        String adminUsername,
        String adminPassword,
        String platformOrgName
) { }
