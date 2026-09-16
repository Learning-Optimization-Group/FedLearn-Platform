package com.federated.fl_platform_api.security;

import org.springframework.stereotype.Component;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * In-memory sliding-window throttle for failed logins (SE-4). A key locks once it accumulates
 * {@link #MAX_FAILURES} failures within {@link #LOCKOUT}; a successful login {@link #reset(String)}s
 * it, and old failures age out of the window so the lock is temporary. Keys are opaque — the caller
 * uses both a per-username and a per-IP key so neither a single-account brute-force nor a
 * username-spraying source can hammer {@code /api/auth/login} indefinitely.
 *
 * <p>Same style as {@code UserSearchService}'s limiter (no external dependency); the {@link Clock}
 * is an injectable seam so window expiry is testable without sleeping. In-memory means per-instance;
 * a shared/distributed store is a later hardening for multi-replica deployments.
 */
@Component
public class LoginRateLimiter {

    static final int MAX_FAILURES = 5;
    static final Duration LOCKOUT = Duration.ofMinutes(15);

    private final Clock clock;
    private final ConcurrentHashMap<String, Deque<Instant>> failures = new ConcurrentHashMap<>();

    public LoginRateLimiter() {
        this(Clock.systemUTC());
    }

    LoginRateLimiter(Clock clock) { // package-private test seam
        this.clock = clock;
    }

    /** True if {@code key} has reached the failure threshold within the current window. */
    public boolean isLocked(String key) {
        Deque<Instant> window = failures.get(key);
        if (window == null) {
            return false;
        }
        synchronized (window) {
            prune(window);
            return window.size() >= MAX_FAILURES;
        }
    }

    /** Record a failed attempt for {@code key}. */
    public void recordFailure(String key) {
        Deque<Instant> window = failures.computeIfAbsent(key, k -> new ArrayDeque<>());
        synchronized (window) {
            prune(window);
            window.addLast(clock.instant());
        }
    }

    /** Clear {@code key} — called on a successful login so a good password ends the throttle. */
    public void reset(String key) {
        failures.remove(key);
    }

    /**
     * SE-4: how long until {@code key} unlocks — the earliest a pruned window drops back below
     * {@link #MAX_FAILURES} — or empty if the key is not currently locked. The controller emits this
     * as the {@code Retry-After} header on the 429 so a throttled client backs off for the right
     * delay instead of hammering {@code /api/auth/login}.
     */
    public Optional<Duration> retryAfter(String key) {
        Deque<Instant> window = failures.get(key);
        if (window == null) {
            return Optional.empty();
        }
        synchronized (window) {
            prune(window);
            if (window.size() < MAX_FAILURES) {
                return Optional.empty();
            }
            // The lock releases when enough of the OLDEST failures age out to drop the count below
            // the threshold: that is when the failure at index (size - MAX_FAILURES) from the oldest
            // expires (peekFirst for exactly MAX_FAILURES). The deque iterates oldest -> newest.
            int releaseIdx = window.size() - MAX_FAILURES;
            Instant releasing = window.peekFirst();
            int i = 0;
            for (Instant t : window) {
                if (i++ == releaseIdx) {
                    releasing = t;
                    break;
                }
            }
            Duration remaining = Duration.between(clock.instant(), releasing.plus(LOCKOUT));
            return Optional.of(remaining.isNegative() ? Duration.ZERO : remaining);
        }
    }

    private void prune(Deque<Instant> window) {
        Instant cutoff = clock.instant().minus(LOCKOUT);
        while (!window.isEmpty() && window.peekFirst().isBefore(cutoff)) {
            window.pollFirst();
        }
    }
}
