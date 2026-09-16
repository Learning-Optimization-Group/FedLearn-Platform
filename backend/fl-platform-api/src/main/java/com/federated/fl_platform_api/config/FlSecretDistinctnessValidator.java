package com.federated.fl_platform_api.config;

import jakarta.annotation.PostConstruct;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

/**
 * SE-20: fail the boot closed when the FL connection-token signing secret is the SAME as the web-auth
 * JWT secret on a DEPLOYED profile.
 *
 * <p>{@code app.fl.token-secret} defaults to {@code app.jwt.secret} ({@code
 * ${APP_FL_TOKEN_SECRET:${app.jwt.secret}}}) for local backward-compat. But the FL server is
 * network-facing and holds the FL secret (SE-1/SE-7); if that secret equals the web-auth key, a
 * compromise of the FL server can mint valid web/admin sessions — defeating the SE-7/SE-17
 * trust-domain isolation (which deliberately keeps {@code APP_JWT_SECRET} out of the FL child). On
 * {@code ec2demo}/{@code production} we therefore require a DISTINCT {@code APP_FL_TOKEN_SECRET} and
 * refuse to start otherwise. Off those profiles (dev/test/base) the fallback stays allowed so a local
 * run needs no extra secret.
 *
 * <p>Note: this is the "boot check they differ" half of finding #7. The complementary web-JWT
 * audience/type binding (so the two token classes can't be cross-presented even under a shared dev
 * secret) is a deliberate, separate change to the auth hot path and is tracked as remaining.
 */
@Component
public class FlSecretDistinctnessValidator {

    /** Profiles that must not share the web + FL signing secret. */
    static final Set<String> DEPLOYED_PROFILES = Set.of("ec2demo", "production");

    private final String jwtSecret;
    private final String flTokenSecret;
    private final Environment environment;

    public FlSecretDistinctnessValidator(
            @Value("${app.jwt.secret}") String jwtSecret,
            @Value("${app.fl.token-secret}") String flTokenSecret,
            Environment environment) {
        this.jwtSecret = jwtSecret;
        this.flTokenSecret = flTokenSecret;
        this.environment = environment;
    }

    @PostConstruct
    void validateOnBoot() {
        check(jwtSecret, flTokenSecret, environment.getActiveProfiles());
    }

    /**
     * Throw when a deployed profile is active and the two secrets resolve equal. Package-private +
     * static so the policy is unit-testable without a Spring context.
     */
    static void check(String jwtSecret, String flTokenSecret, String[] activeProfiles) {
        boolean deployed = Arrays.stream(activeProfiles).anyMatch(DEPLOYED_PROFILES::contains);
        if (deployed && Objects.equals(jwtSecret, flTokenSecret)) {
            throw new IllegalStateException(
                    "SE-20: app.fl.token-secret must be DISTINCT from app.jwt.secret on a deployed "
                            + "profile (ec2demo/production). Set a dedicated APP_FL_TOKEN_SECRET so a "
                            + "compromise of the network-facing FL server cannot forge web/admin "
                            + "sessions. It currently falls back to the web-auth secret.");
        }
    }
}
