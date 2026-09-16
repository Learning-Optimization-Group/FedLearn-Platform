package com.federated.fl_platform_api.service;

import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.List;
import java.util.Optional;

/**
 * BA-16 (was OP-15): best-effort detection of this host's primary client-reachable IPv4, used only in
 * the dev profile to auto-advertise a reachable FL gRPC host so cross-device demos connect without
 * hand-setting {@code FL_SERVER_GRPC_HOST}.
 *
 * <p>Two address classes are considered "advertisable":</p>
 * <ul>
 *   <li><b>Tailscale / CGNAT shared space</b> — {@code 100.64.0.0/10} (100.64.0.0 – 100.127.255.255).
 *       Reachable from any device on the same tailnet, <em>across</em> physical LANs / NAT. This is the
 *       address our Mac/Windows/phone demo devices actually dial.</li>
 *   <li><b>Site-local LAN</b> — RFC-1918 {@code 10/8}, {@code 172.16/12}, {@code 192.168/16}
 *       (JDK {@link InetAddress#isSiteLocalAddress()}). Reachable only from the same subnet.</li>
 * </ul>
 *
 * <p>When both classes are present the {@code preferCgnat} flag decides the order (see
 * {@link #selectPreferredIPv4}); a single present class is always advertised regardless of preference.
 * Anything else (public IPs, the surrounding public {@code 100.0.0.0/8} outside the /10, loopback, IPv6,
 * link-local) is never advertised — the caller then falls back to localhost.</p>
 *
 * <p>The classification/selection logic ({@link #isCgnatShared}, {@link #selectPreferredIPv4}) is pure and
 * unit-testable with a hand-built candidate list; {@link #primaryReachableIPv4} is the thin real-NIC
 * adapter around it.</p>
 */
final class LanAddressDetector {

    private LanAddressDetector() {}

    /**
     * Real-NIC adapter: enumerate this host's usable interfaces, then pick the preferred advertisable IPv4.
     *
     * @param preferCgnat when both a CGNAT and a site-local address exist, prefer the CGNAT (Tailscale) one
     * @return the host to advertise, or empty when no advertisable address is found (e.g. offline)
     */
    static Optional<String> primaryReachableIPv4(boolean preferCgnat) {
        return selectPreferredIPv4(usableIPv4Addresses(), preferCgnat);
    }

    /** Enumerate the IPv4 addresses of all up, non-loopback, non-virtual interfaces (empty on error). */
    private static List<InetAddress> usableIPv4Addresses() {
        try {
            return NetworkInterface.networkInterfaces()
                    .filter(LanAddressDetector::isUsable)
                    .flatMap(NetworkInterface::inetAddresses)
                    .filter(a -> a instanceof Inet4Address)
                    .toList();
        } catch (SocketException e) {
            return List.of();
        }
    }

    /**
     * Pure selection: from a candidate list, return the preferred advertisable IPv4 as a string.
     *
     * <p>Precedence is an <em>ordering</em>, not an exclusion — a lone address of either class is still
     * returned so a Tailscale-only or a LAN-only host both work:</p>
     * <ul>
     *   <li>{@code preferCgnat == true}  → first CGNAT ({@code 100.64/10}), else first site-local.</li>
     *   <li>{@code preferCgnat == false} → first site-local, else first CGNAT.</li>
     * </ul>
     */
    static Optional<String> selectPreferredIPv4(List<InetAddress> candidates, boolean preferCgnat) {
        Optional<String> cgnat = candidates.stream()
                .filter(LanAddressDetector::isCgnatShared)
                .map(InetAddress::getHostAddress)
                .findFirst();
        Optional<String> siteLocal = candidates.stream()
                .filter(a -> a instanceof Inet4Address && a.isSiteLocalAddress())
                .map(InetAddress::getHostAddress)
                .findFirst();
        return preferCgnat ? cgnat.or(() -> siteLocal) : siteLocal.or(() -> cgnat);
    }

    /**
     * True iff {@code a} is an IPv4 address in the CGNAT / shared-address block {@code 100.64.0.0/10}
     * (RFC 6598), i.e. 100.64.0.0 – 100.127.255.255 inclusive — the range Tailscale hands out.
     *
     * <p>A /10 fixes the first 10 bits: octet0 == 100 plus the top two bits of octet1, so octet1 spans
     * 64 (0b01000000) .. 127 (0b01111111). This precisely EXCLUDES the surrounding public
     * {@code 100.0.0.0/8} (e.g. 100.0.0.5, 100.63.255.255, 100.128.0.0).</p>
     */
    static boolean isCgnatShared(InetAddress a) {
        if (!(a instanceof Inet4Address)) {
            return false;
        }
        byte[] b = a.getAddress();               // network byte order, 4 bytes for IPv4
        int octet0 = b[0] & 0xFF;
        int octet1 = b[1] & 0xFF;
        return octet0 == 100 && octet1 >= 64 && octet1 <= 127;
    }

    private static boolean isUsable(NetworkInterface ni) {
        try {
            return ni.isUp() && !ni.isLoopback() && !ni.isVirtual();
        } catch (SocketException e) {
            return false;
        }
    }
}
