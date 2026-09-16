package com.federated.fl_platform_api.config;

import jakarta.annotation.PostConstruct;
import java.util.Arrays;
import java.util.Set;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

/**
 * SE-14: fail the boot closed on an insecure FL-boundary auth configuration.
 *
 * <p>Turning on mandatory client auth ({@code app.fl.require-client-auth=true}) makes each client
 * present a connection token on every gRPC call. If the wire is NOT encrypted
 * ({@code app.fl.require-tls=false}) on a deployed profile, that token rides plaintext gRPC over the
 * WAN (audit #37) where an eavesdropper can capture and replay it — so "auth on, TLS off" is a false
 * sense of security. On {@code ec2demo}/{@code production} we therefore refuse to start when auth is
 * on but TLS is off, UNLESS the operator explicitly acknowledges a trusted-network deployment
 * (Tailscale / loopback / private VPC) via {@code app.fl.allow-plaintext-client-auth=true}.
 *
 * <p>Companion to {@link FlSecretDistinctnessValidator} (SE-20, which fails closed when the FL token
 * secret equals the web JWT secret). Off the deployed profiles this is never gated, so local dev is
 * free to run plaintext with or without auth.
 */
@Component
public class FlBoundaryAuthPolicyValidator {

    /** Profiles where the insecure auth-on/TLS-off combination is refused. */
    static final Set<String> DEPLOYED_PROFILES = Set.of("ec2demo", "production");

    private final boolean requireClientAuth;
    private final boolean requireTls;
    private final boolean allowPlaintextClientAuth;
    private final Environment environment;

    public FlBoundaryAuthPolicyValidator(
            @Value("${app.fl.require-client-auth}") boolean requireClientAuth,
            @Value("${app.fl.require-tls}") boolean requireTls,
            @Value("${app.fl.allow-plaintext-client-auth:false}") boolean allowPlaintextClientAuth,
            Environment environment) {
        this.requireClientAuth = requireClientAuth;
        this.requireTls = requireTls;
        this.allowPlaintextClientAuth = allowPlaintextClientAuth;
        this.environment = environment;
    }

    @PostConstruct
    void validateOnBoot() {
        check(requireClientAuth, requireTls, allowPlaintextClientAuth, environment.getActiveProfiles());
    }

    /**
     * Throw when a deployed profile enables client auth over an unencrypted wire without an explicit
     * trusted-network acknowledgement. Package-private + static so the policy is unit-testable without
     * a Spring context.
     */
    static void check(boolean requireClientAuth, boolean requireTls, boolean allowPlaintextClientAuth,
                      String[] activeProfiles) {
        boolean deployed = Arrays.stream(activeProfiles).anyMatch(DEPLOYED_PROFILES::contains);
        if (deployed && requireClientAuth && !requireTls && !allowPlaintextClientAuth) {
            throw new IllegalStateException(
                    "SE-14: app.fl.require-client-auth=true with app.fl.require-tls=false on a deployed "
                            + "profile is insecure — the connection token would ride plaintext gRPC over "
                            + "the WAN and be replayable. Enable TLS (app.fl.require-tls=true), or set "
                            + "app.fl.allow-plaintext-client-auth=true to acknowledge a trusted network "
                            + "(Tailscale / loopback / private VPC).");
        }
    }
}
