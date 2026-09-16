package com.federated.fl_platform_api.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * SE-20: {@code app.fl.token-secret} defaults to {@code app.jwt.secret} for backward-compat. That is
 * fine locally, but in a DEPLOYED profile the FL-token signing key and the web-auth signing key MUST
 * differ — otherwise a compromise of the network-facing FL server (which holds the FL secret) can
 * forge web/admin sessions, defeating the SE-7/SE-17 trust-domain isolation. This validator fails the
 * boot closed when the two secrets resolve equal on ec2demo/production.
 */
class FlSecretDistinctnessValidatorTest {

    @Test
    void deployedProfile_equalSecrets_failsClosed() {
        assertThrows(IllegalStateException.class, () ->
                FlSecretDistinctnessValidator.check("same-secret", "same-secret", new String[]{"production"}));
        assertThrows(IllegalStateException.class, () ->
                FlSecretDistinctnessValidator.check("same-secret", "same-secret", new String[]{"ec2demo"}));
    }

    @Test
    void deployedProfile_distinctSecrets_ok() {
        assertDoesNotThrow(() ->
                FlSecretDistinctnessValidator.check("web-secret", "fl-secret", new String[]{"production", "extra"}));
    }

    @Test
    void devOrTestOrBaseProfile_equalSecrets_allowed() {
        // Off the deployed profiles the fallback is intentional (no separate secret needed locally).
        assertDoesNotThrow(() ->
                FlSecretDistinctnessValidator.check("same", "same", new String[]{"dev"}));
        assertDoesNotThrow(() ->
                FlSecretDistinctnessValidator.check("same", "same", new String[]{"test"}));
        assertDoesNotThrow(() ->
                FlSecretDistinctnessValidator.check("same", "same", new String[]{}));  // no active profile
    }
}
