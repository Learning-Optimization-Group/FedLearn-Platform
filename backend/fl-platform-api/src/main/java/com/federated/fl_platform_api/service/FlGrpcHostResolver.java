package com.federated.fl_platform_api.service;

import java.util.Optional;
import java.util.function.Supplier;

/**
 * OP-15 / BA-16: resolve the FL gRPC host the backend ADVERTISES to clients (in the enroll response).
 *
 * <p>A run FL server binds all interfaces, but the advertised host defaults to {@code localhost} —
 * unreachable by a remote client (a phone, another laptop), which then dials its own loopback. This
 * applies a fixed, documented resolution precedence:</p>
 * <ol>
 *   <li><b>Explicit config always wins.</b> A non-{@code localhost} {@code app.fl-server.grpc-host} /
 *       {@code FL_SERVER_GRPC_HOST} is returned verbatim in every profile — the operator's escape hatch.</li>
 *   <li><b>Auto-detect (dev only, default {@code localhost}).</b> Consult {@code reachableIp} for the best
 *       client-reachable address: a Tailscale/CGNAT {@code 100.64.0.0/10} address is preferred over a
 *       site-local LAN IP by default (see {@link LanAddressDetector}); reachable across the whole tailnet,
 *       it is what our cross-network demo devices actually dial. Neither present → fall back to
 *       {@code localhost}.</li>
 *   <li><b>Non-dev profiles are never auto-upgraded here</b> — deployed profiles are governed by OP-3's
 *       fail-loud check in {@code BootstrapRunner} (which accepts a real host, incl. a {@code 100.x}).</li>
 * </ol>
 *
 * <p><b>Heuristic caveat:</b> if a host has both a Tailscale and a LAN address but clients are actually on
 * the LAN, preferring the Tailscale address could be sub-optimal. The choice is predictable and
 * overridable: flip {@code app.fl-server.prefer-cgnat=false} to prefer the LAN IP, or set
 * {@code FL_SERVER_GRPC_HOST} explicitly (rule 1) to pin an exact host. Pure + side-effect-free: the
 * detection lookup is supplied by the caller.</p>
 */
public final class FlGrpcHostResolver {

    static final String LOOPBACK = "localhost";

    private FlGrpcHostResolver() {}

    /**
     * @param configuredHost the {@code app.fl-server.grpc-host} value (null/blank normalized to localhost)
     * @param isDev          whether the {@code dev} profile is active
     * @param reachableIp    supplier of the detected best client-reachable IPv4 — CGNAT/Tailscale-preferred
     *                       (only consulted when auto-detecting, i.e. dev + default localhost)
     * @return the host to advertise
     */
    public static String resolve(String configuredHost, boolean isDev, Supplier<Optional<String>> reachableIp) {
        String host = (configuredHost == null || configuredHost.isBlank()) ? LOOPBACK : configuredHost.trim();
        if (!isDev) {
            return host;                       // non-dev: unchanged (OP-3 guards deployed localhost)
        }
        if (!host.equals(LOOPBACK)) {
            return host;                       // explicit dev override respected
        }
        return reachableIp.get()
                .filter(ip -> ip != null && !ip.isBlank())
                .map(String::trim)
                .orElse(LOOPBACK);             // upgrade localhost -> reachable IP, or fall back to localhost
    }
}
