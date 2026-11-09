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

    public static final int MAX_ENTRIES = 50_000;

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
        Deque<Instant> window = failures.computeIfPresent(key, (k, w) -> {
            prune(w);
            return w.isEmpty() ? null : w;
        });
        return window != null && window.size() >= MAX_FAILURES;
    }

    /** Record a failed attempt for {@code key}. */
    public void recordFailure(String key) {
        failures.compute(key, (k, w) -> {
            if (w == null && failures.size() >= MAX_ENTRIES) {
                return null;
            }
            Deque<Instant> window = (w == null) ? new ArrayDeque<>() : w;
            prune(window);
            window.addLast(clock.instant());
            return window;
        });
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
        java.util.concurrent.atomic.AtomicReference<Duration> result = new java.util.concurrent.atomic.AtomicReference<>(null);
        failures.computeIfPresent(key, (k, window) -> {
            prune(window);
            if (window.isEmpty()) {
                return null;
            }
            if (window.size() < MAX_FAILURES) {
                return window;
            }
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
            result.set(remaining.isNegative() ? Duration.ZERO : remaining);
            return window;
        });
        return Optional.ofNullable(result.get());
    }

    private void prune(Deque<Instant> window) {
        Instant cutoff = clock.instant().minus(LOCKOUT);
        while (!window.isEmpty() && window.peekFirst().isBefore(cutoff)) {
            window.pollFirst();
        }
    }
}
