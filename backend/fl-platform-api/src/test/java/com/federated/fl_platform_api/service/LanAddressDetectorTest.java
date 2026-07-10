package com.federated.fl_platform_api.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.InetAddress;
import java.net.UnknownHostException;
import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;

/**
 * BA-16: the FL gRPC host auto-detector must be able to advertise a Tailscale/CGNAT
 * ({@code 100.64.0.0/10}) address, not only an RFC-1918 site-local LAN IP, so cross-network
 * demo devices (Mac/Windows/phone on a tailnet) connect without hand-setting
 * FL_SERVER_GRPC_HOST. These tests pin the pure classification + ordered-selection logic
 * without touching real network interfaces.
 */
class LanAddressDetectorTest {

    /** Build an InetAddress from a numeric IPv4/IPv6 literal — offline-safe (no DNS for literals). */
    private static InetAddress addr(String ip) {
        try {
            return InetAddress.getByName(ip);
        } catch (UnknownHostException e) {
            throw new RuntimeException(e);
        }
    }

    // --- CGNAT 100.64.0.0/10 boundary: 100.64.0.0 – 100.127.255.255 inclusive ---

    @Test
    void cgnatRange_lowAndHighBoundaries_areCgnat() {
        assertTrue(LanAddressDetector.isCgnatShared(addr("100.64.0.0")));        // first address of the /10
        assertTrue(LanAddressDetector.isCgnatShared(addr("100.127.255.255")));   // last address of the /10
        assertTrue(LanAddressDetector.isCgnatShared(addr("100.100.50.7")));      // typical Tailscale IP
    }

    @Test
    void publicHundredSlashEight_outsideThe_slash10_isNotCgnat() {
        // 100.0.0.0/8 is public space; only the 100.64/10 sub-block is CGNAT/shared.
        assertFalse(LanAddressDetector.isCgnatShared(addr("100.0.0.5")));
        assertFalse(LanAddressDetector.isCgnatShared(addr("100.63.255.255")));   // one below the /10
        assertFalse(LanAddressDetector.isCgnatShared(addr("100.128.0.0")));      // one above the /10
        assertFalse(LanAddressDetector.isCgnatShared(addr("100.255.255.255")));
    }

    @Test
    void siteLocalAndOtherAddresses_areNotCgnat() {
        assertFalse(LanAddressDetector.isCgnatShared(addr("192.168.1.10")));
        assertFalse(LanAddressDetector.isCgnatShared(addr("10.0.0.1")));
        assertFalse(LanAddressDetector.isCgnatShared(addr("172.16.5.5")));
        assertFalse(LanAddressDetector.isCgnatShared(addr("8.8.8.8")));
        assertFalse(LanAddressDetector.isCgnatShared(addr("::1")));              // IPv6 is never CGNAT-v4
    }

    @Test
    void cgnatAndSiteLocal_areDisjointCategories() {
        // The JDK does NOT classify 100.64/10 as site-local, so the two selectors never double-count.
        assertFalse(addr("100.64.0.7").isSiteLocalAddress());
        assertTrue(addr("192.168.1.10").isSiteLocalAddress());
    }

    // --- ordered selection: CGNAT-first vs site-local-first ---

    @Test
    void whenBothPresent_preferCgnat_choosesTheTailscaleAddress() {
        List<InetAddress> candidates = List.of(addr("192.168.1.5"), addr("100.64.0.7"));
        assertEquals(Optional.of("100.64.0.7"),
                LanAddressDetector.selectPreferredIPv4(candidates, true));
    }

    @Test
    void whenBothPresent_preferSiteLocal_choosesTheLanAddress() {
        List<InetAddress> candidates = List.of(addr("192.168.1.5"), addr("100.64.0.7"));
        assertEquals(Optional.of("192.168.1.5"),
                LanAddressDetector.selectPreferredIPv4(candidates, false));
    }

    @Test
    void selectionIsOrderIndependent_notJustFirstMatch() {
        // CGNAT appears first in the list, but prefer-site-local must still pick the LAN IP.
        List<InetAddress> candidates = List.of(addr("100.64.0.7"), addr("192.168.1.5"));
        assertEquals(Optional.of("192.168.1.5"),
                LanAddressDetector.selectPreferredIPv4(candidates, false));
    }

    @Test
    void whenOnlyCgnatPresent_itIsAdvertised_regardlessOfPreference() {
        List<InetAddress> candidates = List.of(addr("100.100.0.9"));
        assertEquals(Optional.of("100.100.0.9"),
                LanAddressDetector.selectPreferredIPv4(candidates, true));
        assertEquals(Optional.of("100.100.0.9"),
                LanAddressDetector.selectPreferredIPv4(candidates, false));
    }

    @Test
    void whenOnlySiteLocalPresent_itIsAdvertised_regardlessOfPreference() {
        List<InetAddress> candidates = List.of(addr("10.0.0.5"));
        assertEquals(Optional.of("10.0.0.5"),
                LanAddressDetector.selectPreferredIPv4(candidates, true));
        assertEquals(Optional.of("10.0.0.5"),
                LanAddressDetector.selectPreferredIPv4(candidates, false));
    }

    @Test
    void publicOnlyCandidates_yieldNoAdvertisableAddress() {
        // A public 100/8 (outside the /10) and a routable public IP are neither CGNAT nor site-local.
        List<InetAddress> candidates = List.of(addr("100.0.0.5"), addr("8.8.8.8"));
        assertEquals(Optional.empty(), LanAddressDetector.selectPreferredIPv4(candidates, true));
        assertEquals(Optional.empty(), LanAddressDetector.selectPreferredIPv4(candidates, false));
    }

    @Test
    void emptyCandidateList_yieldsEmpty() {
        assertEquals(Optional.empty(), LanAddressDetector.selectPreferredIPv4(List.of(), true));
    }
}
