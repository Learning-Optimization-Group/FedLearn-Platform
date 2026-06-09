package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Proves the rate-limit bucket map is bounded: stale (expired-window) buckets
 * are swept by {@link UserSearchService#evictStale(long)}, while buckets whose
 * window is the current minute survive. Pure unit test — no Spring context, no
 * dependencies — exercising the eviction seam directly.
 */
class UserSearchServiceEvictionTest {

    private final int originalCap = UserSearchService.MAX_BUCKETS;

    @AfterEach
    void restoreCap() {
        UserSearchService.MAX_BUCKETS = originalCap;
    }

    @Test
    void evictStale_removesExpiredWindows_keepsCurrentWindow() {
        UserSearchService svc = new UserSearchService();

        long nowMin = 1_000_000L;
        // 5 stale buckets (windows in the past) ...
        for (long i = 0; i < 5; i++) {
            svc.seedBucket(i, nowMin - 1);
        }
        // ... and 1 current-window bucket that must survive.
        svc.seedBucket(99L, nowMin);
        assertThat(svc.bucketCount()).isEqualTo(6);

        svc.evictStale(nowMin);

        assertThat(svc.bucketCount())
                .as("all stale windows swept; only the current-minute bucket remains")
                .isEqualTo(1);
    }

    @Test
    void overCap_sweepDropsStaleBuckets() {
        UserSearchService svc = new UserSearchService();
        // Lower the cap so we can exercise the over-cap sweep cheaply.
        UserSearchService.MAX_BUCKETS = 3;

        long nowMin = 2_000_000L;
        // Seed 4 stale buckets (> cap of 3) with expired windows.
        for (long i = 0; i < 4; i++) {
            svc.seedBucket(i, nowMin - 2);
        }
        assertThat(svc.bucketCount()).isGreaterThan(UserSearchService.MAX_BUCKETS);

        // The same sweep the over-cap branch runs in consumeToken.
        svc.evictStale(nowMin);

        assertThat(svc.bucketCount())
                .as("once over cap, all expired-window buckets are evicted")
                .isZero();
    }
}
