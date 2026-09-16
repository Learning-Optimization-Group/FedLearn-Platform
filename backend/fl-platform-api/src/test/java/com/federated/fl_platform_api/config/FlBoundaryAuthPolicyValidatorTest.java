package com.federated.fl_platform_api.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * SE-14: enabling mandatory FL-boundary client auth (`app.fl.require-client-auth=true`) on a DEPLOYED
 * profile is only safe if the wire is also encrypted — otherwise the connection token the client
 * presents rides plaintext gRPC over the WAN (audit #37) and an eavesdropper can capture and replay
 * it, defeating the auth. This validator fails the boot closed on that insecure combination
 * (auth-on + TLS-off) unless the operator explicitly acknowledges a trusted-network deployment
 * (e.g. Tailscale/loopback) via `app.fl.allow-plaintext-client-auth=true`.
 */
class FlBoundaryAuthPolicyValidatorTest {

    // deployed + auth-on + TLS-off + no ack -> insecure -> fail closed
    @Test
    void deployedAuthOnWithoutTls_failsClosed() {
        assertThrows(IllegalStateException.class, () ->
                FlBoundaryAuthPolicyValidator.check(true, false, false, new String[]{"production"}));
        assertThrows(IllegalStateException.class, () ->
                FlBoundaryAuthPolicyValidator.check(true, false, false, new String[]{"ec2demo"}));
    }

    // deployed + auth-on + TLS-on -> the intended secure config
    @Test
    void deployedAuthOnWithTls_ok() {
        assertDoesNotThrow(() ->
                FlBoundaryAuthPolicyValidator.check(true, true, false, new String[]{"production"}));
    }

    // deployed + auth-off -> nothing to protect (fail-open boundary, pre-SE-14 state)
    @Test
    void deployedAuthOff_ok() {
        assertDoesNotThrow(() ->
                FlBoundaryAuthPolicyValidator.check(false, false, false, new String[]{"production"}));
    }

    // deployed + auth-on + TLS-off but EXPLICITLY acknowledged (trusted network, e.g. Tailscale) -> ok
    @Test
    void deployedAuthOnPlaintextButAcknowledged_ok() {
        assertDoesNotThrow(() ->
                FlBoundaryAuthPolicyValidator.check(true, false, true, new String[]{"ec2demo"}));
    }

    // non-deployed (dev/test) is never gated — local runs stay plaintext + auth-off/on freely
    @Test
    void nonDeployed_neverGated() {
        assertDoesNotThrow(() ->
                FlBoundaryAuthPolicyValidator.check(true, false, false, new String[]{"dev"}));
        assertDoesNotThrow(() ->
                FlBoundaryAuthPolicyValidator.check(true, false, false, new String[]{}));
    }
}
