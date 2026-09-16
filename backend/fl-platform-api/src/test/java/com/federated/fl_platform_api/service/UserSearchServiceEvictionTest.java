package com.federated.fl_platform_api.service;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.time.Instant;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Proves the rate-limit bucket map is bounded: buckets whose rolling window has
 * fully drained are swept by {@link UserSearchService#evictStale(Instant)},
 * while a bucket with a live (recent) request survives. Pure unit test — no
 * Spring context, no dependencies — exercising the eviction seam directly.
 */
class UserSearchServiceEvictionTest {

    private final int originalCap = UserSearchService.MAX_BUCKETS;

    @AfterEach
    void restoreCap() {
        UserSearchService.MAX_BUCKETS = originalCap;
    }

    @Test
    void evictStale_removesDrainedWindows_keepsLiveWindow() {
        UserSearchService svc = new UserSearchService();

        Instant now = Instant.parse("2026-07-03T12:00:00Z");
        // 5 drained buckets: their only request has aged out of the window ...
        for (long i = 0; i < 5; i++) {
            svc.seedBucket(i, now.minus(Duration.ofMinutes(2)));
        }
        // ... and 1 with a request inside the window that must survive.
        svc.seedBucket(99L, now);
        assertThat(svc.bucketCount()).isEqualTo(6);

        svc.evictStale(now);

        assertThat(svc.bucketCount())
                .as("all drained windows swept; only the live bucket remains")
                .isEqualTo(1);
    }

    @Test
    void overCap_sweepDropsDrainedBuckets() {
        UserSearchService svc = new UserSearchService();
        // Lower the cap so we can exercise the over-cap sweep cheaply.
        UserSearchService.MAX_BUCKETS = 3;

        Instant now = Instant.parse("2026-07-03T12:00:00Z");
        // Seed 4 drained buckets (> cap of 3) whose windows have all aged out.
        for (long i = 0; i < 4; i++) {
            svc.seedBucket(i, now.minus(Duration.ofMinutes(5)));
        }
        assertThat(svc.bucketCount()).isGreaterThan(UserSearchService.MAX_BUCKETS);

        // The same sweep the over-cap branch runs in consumeToken.
        svc.evictStale(now);

        assertThat(svc.bucketCount())
                .as("once over cap, all drained buckets are evicted")
                .isZero();
    }
}
