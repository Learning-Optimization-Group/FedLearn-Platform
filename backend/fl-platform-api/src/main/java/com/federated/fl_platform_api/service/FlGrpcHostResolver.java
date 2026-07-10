package com.federated.fl_platform_api.service;

import java.util.Optional;
import java.util.function.Supplier;

/**
 * OP-15: resolve the FL gRPC host the backend ADVERTISES to clients (in the enroll response).
 *
 * <p>In the {@code dev} profile, a run FL server binds all interfaces but the advertised host defaults to
 * {@code localhost} — unreachable by a same-LAN client (e.g. a phone), which then dials its own loopback.
 * This upgrades a default {@code localhost} to the detected LAN IP so local mobile/multi-host testing
 * works out of the box, while respecting an explicit {@code FL_SERVER_GRPC_HOST} override and leaving
 * every non-dev profile untouched (deployed profiles are governed by OP-3's fail-loud check in
 * {@code BootstrapRunner}). Pure + side-effect-free: the LAN lookup is supplied by the caller.
 */
public final class FlGrpcHostResolver {

    static final String LOOPBACK = "localhost";

    private FlGrpcHostResolver() {}

    /**
     * @param configuredHost the {@code app.fl-server.grpc-host} value (null/blank normalized to localhost)
     * @param isDev          whether the {@code dev} profile is active
     * @param lanIp          supplier of the detected primary LAN IPv4 (only consulted when needed)
     * @return the host to advertise
     */
    public static String resolve(String configuredHost, boolean isDev, Supplier<Optional<String>> lanIp) {
        String host = (configuredHost == null || configuredHost.isBlank()) ? LOOPBACK : configuredHost.trim();
        if (!isDev) {
            return host;                       // non-dev: unchanged (OP-3 guards deployed localhost)
        }
        if (!host.equals(LOOPBACK)) {
            return host;                       // explicit dev override respected
        }
        return lanIp.get()
                .filter(ip -> ip != null && !ip.isBlank())
                .map(String::trim)
                .orElse(LOOPBACK);             // upgrade localhost -> LAN IP, or fall back to localhost
    }
}
