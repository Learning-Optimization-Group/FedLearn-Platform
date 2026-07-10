package com.federated.fl_platform_api.service;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.Optional;

/**
 * OP-15: best-effort detection of this host's primary site-local IPv4 (10.x / 172.16-31.x / 192.168.x),
 * used only in the dev profile to advertise a LAN-reachable FL gRPC host. Returns empty on any error or
 * when there is no site-local address (e.g. offline) — the caller then falls back to localhost.
 */
final class LanAddressDetector {

    private LanAddressDetector() {}

    static Optional<String> primarySiteLocalIPv4() {
        try {
            return NetworkInterface.networkInterfaces()
                    .filter(LanAddressDetector::isUsable)
                    .flatMap(NetworkInterface::inetAddresses)
                    .filter(a -> a instanceof Inet4Address && a.isSiteLocalAddress())
                    .map(InetAddress::getHostAddress)
                    .findFirst();
        } catch (SocketException e) {
            return Optional.empty();
        }
    }

    private static boolean isUsable(NetworkInterface ni) {
        try {
            return ni.isUp() && !ni.isLoopback() && !ni.isVirtual();
        } catch (SocketException e) {
            return false;
        }
    }
}
