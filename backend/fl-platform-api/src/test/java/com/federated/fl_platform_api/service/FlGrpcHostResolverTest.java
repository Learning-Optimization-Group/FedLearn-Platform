package com.federated.fl_platform_api.service;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Optional;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

/**
 * OP-15: in the dev profile the advertised FL gRPC host is upgraded from a default {@code localhost}
 * to the detected LAN IP so a same-LAN client (a phone) can reach the FL server out of the box. An
 * explicit override, and every non-dev profile, is returned unchanged. This pins that decision table.
 */
class FlGrpcHostResolverTest {

    private static Supplier<Optional<String>> lan(String ip) {
        return () -> Optional.ofNullable(ip);
    }

    @Test
    void devWithDefaultLocalhost_isUpgradedToTheLanIp() {
        assertEquals("10.0.0.130", FlGrpcHostResolver.resolve("localhost", true, lan("10.0.0.130")));
    }

    @Test
    void devWithDefaultLocalhost_butNoLanIp_fallsBackToLocalhost() {
        assertEquals("localhost", FlGrpcHostResolver.resolve("localhost", true, lan(null)));
        assertEquals("localhost", FlGrpcHostResolver.resolve("localhost", true, lan("   ")));
    }

    @Test
    void devWithExplicitOverride_isRespected() {
        assertEquals("192.168.1.9", FlGrpcHostResolver.resolve("192.168.1.9", true, lan("10.0.0.130")));
    }

    @Test
    void nonDev_isAlwaysUnchanged_evenAtLocalhost() {
        // OP-3 (BootstrapRunner) governs deployed profiles; the resolver must not touch them.
        assertEquals("localhost", FlGrpcHostResolver.resolve("localhost", false, lan("10.0.0.130")));
        assertEquals("fl.internal", FlGrpcHostResolver.resolve("fl.internal", false, lan("10.0.0.130")));
    }

    @Test
    void blankOrNullConfiguredHost_isTreatedAsLocalhostDefault() {
        assertEquals("10.0.0.130", FlGrpcHostResolver.resolve(null, true, lan("10.0.0.130")));
        assertEquals("10.0.0.130", FlGrpcHostResolver.resolve("  ", true, lan("10.0.0.130")));
        // non-dev + null still returns the (normalized) localhost, unchanged behaviour
        assertEquals("localhost", FlGrpcHostResolver.resolve(null, false, lan("10.0.0.130")));
    }

    @Test
    void theLanSupplierIsNotConsultedWhenNotNeeded() {
        // An explicit host (dev) and any non-dev call must not even call the (potentially slow) detector.
        Supplier<Optional<String>> boom = () -> {
            throw new AssertionError("LAN detector must not be consulted here");
        };
        assertEquals("myhost", FlGrpcHostResolver.resolve("myhost", true, boom));
        assertEquals("localhost", FlGrpcHostResolver.resolve("localhost", false, boom));
    }
}
