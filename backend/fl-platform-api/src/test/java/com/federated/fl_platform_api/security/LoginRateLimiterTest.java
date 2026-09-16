package com.federated.fl_platform_api.security;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * SE-4 login throttling: a key (username or IP) locks after {@code MAX_FAILURES} failures within the
 * window, a success resets it, and the lock expires once failures age out. No Spring context — the
 * limiter takes an injectable {@link Clock} so window expiry is testable without sleeping.
 */
class LoginRateLimiterTest {

    /** A clock the test can advance to exercise the sliding window deterministically. */
    static final class MutableClock extends Clock {
        private Instant now;
        MutableClock(Instant start) { this.now = start; }
        void advance(Duration d) { now = now.plus(d); }
        @Override public Instant instant() { return now; }
        @Override public ZoneId getZone() { return ZoneOffset.UTC; }
        @Override public Clock withZone(ZoneId zone) { return this; }
    }

    private MutableClock clock;
    private LoginRateLimiter limiter;

    @BeforeEach
    void setUp() {
        clock = new MutableClock(Instant.parse("2026-01-01T00:00:00Z"));
        limiter = new LoginRateLimiter(clock);
    }

    @Test
    void retryAfter_isEmpty_belowTheThreshold() {
        for (int i = 0; i < LoginRateLimiter.MAX_FAILURES - 1; i++) {
            limiter.recordFailure("u:alice");
        }
        assertThat(limiter.retryAfter("u:alice")).isEmpty();
    }

    @Test
    void retryAfter_reportsFullLockout_whenJustLocked() {
        for (int i = 0; i < LoginRateLimiter.MAX_FAILURES; i++) {
            limiter.recordFailure("u:alice");
        }
        // Just locked, no time has passed: the whole lockout window remains.
        assertThat(limiter.retryAfter("u:alice")).hasValue(LoginRateLimiter.LOCKOUT);
    }

    @Test
    void retryAfter_shrinksAsTheWindowAges() {
        for (int i = 0; i < LoginRateLimiter.MAX_FAILURES; i++) {
            limiter.recordFailure("u:alice");
        }
        clock.advance(Duration.ofMinutes(5));
        // The oldest (releasing) failure is now 5 min old, so 10 min of the 15-min lockout remain.
        assertThat(limiter.retryAfter("u:alice")).hasValue(Duration.ofMinutes(10));
    }

    @Test
    void not_locked_below_the_threshold() {
        for (int i = 0; i < LoginRateLimiter.MAX_FAILURES - 1; i++) {
            limiter.recordFailure("u:alice");
        }
        assertThat(limiter.isLocked("u:alice")).isFalse();
    }

    @Test
    void locks_at_the_failure_threshold_and_keys_are_isolated() {
        for (int i = 0; i < LoginRateLimiter.MAX_FAILURES; i++) {
            limiter.recordFailure("u:alice");
        }
        assertThat(limiter.isLocked("u:alice")).isTrue();
        assertThat(limiter.isLocked("u:bob")).isFalse(); // one account's brute-force does not lock another
    }

    @Test
    void a_successful_login_resets_the_counter() {
        for (int i = 0; i < LoginRateLimiter.MAX_FAILURES; i++) {
            limiter.recordFailure("u:alice");
        }
        limiter.reset("u:alice");
        assertThat(limiter.isLocked("u:alice")).isFalse();
    }

    @Test
    void the_lock_expires_after_the_window() {
        for (int i = 0; i < LoginRateLimiter.MAX_FAILURES; i++) {
            limiter.recordFailure("u:alice");
        }
        assertThat(limiter.isLocked("u:alice")).isTrue();

        clock.advance(LoginRateLimiter.LOCKOUT.plusSeconds(1)); // all failures age out of the window
        assertThat(limiter.isLocked("u:alice")).isFalse();
    }
}
