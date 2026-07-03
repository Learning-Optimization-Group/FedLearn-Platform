package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.UserSearchResultDto;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Service
public class UserSearchService {

    @Autowired private UserRepository userRepository;
    @Autowired private AuthorizationService authz;

    private static final int MIN_QUERY_LENGTH = 2;
    private static final int MAX_RESULTS = 20;
    private static final int MAX_REQUESTS_PER_MINUTE = 30;

    /**
     * Upper bound on retained rate-limit buckets. One entry per distinct caller
     * would otherwise grow without bound. When exceeded we sweep stale (expired-
     * window) entries opportunistically. Package-private (non-final) so a test
     * can lower the cap to exercise eviction cheaply.
     */
    static int MAX_BUCKETS = 50_000;

    /** Rolling window of recent request timestamps per caller (oldest first). */
    private final ConcurrentHashMap<Long, Deque<Instant>> buckets = new ConcurrentHashMap<>();

    /** Length of the rolling rate-limit window. */
    private static final Duration WINDOW = Duration.ofMinutes(1);

    /** Injectable clock so a test can drive window expiry deterministically. */
    private final Clock clock;

    public UserSearchService() {
        this.clock = Clock.systemUTC();
    }

    /** Test seam: construct with a controllable clock. */
    UserSearchService(Clock clock) {
        this.clock = clock;
    }

    public List<UserSearchResultDto> search(String query) {
        Long callerId = authz.currentUser().getId();
        if (!consumeToken(callerId)) {
            throw new ResponseStatusException(HttpStatus.TOO_MANY_REQUESTS,
                "Search rate limit exceeded; retry in a minute");
        }
        if (query == null || query.length() < MIN_QUERY_LENGTH) return Collections.emptyList();

        return userRepository
            .findByUsernameStartingWithIgnoreCaseOrderByUsernameAsc(query, PageRequest.of(0, MAX_RESULTS))
            .stream().map(u -> {
                UserSearchResultDto d = new UserSearchResultDto();
                d.setId(u.getId());
                d.setUsername(u.getUsername());
                return d;
            }).collect(Collectors.toList());
    }

    private boolean consumeToken(Long callerId) {
        Instant now = clock.instant();
        // Opportunistic eviction: only when the map has grown past the cap do we
        // pay for a sweep, dropping buckets whose window has fully drained. Keeps
        // the common path a single map lookup.
        if (buckets.size() > MAX_BUCKETS) {
            evictStale(now);
        }
        Deque<Instant> window = buckets.computeIfAbsent(callerId, k -> new ArrayDeque<>());
        synchronized (window) {
            pruneExpired(window, now);
            if (window.size() >= MAX_REQUESTS_PER_MINUTE) return false;
            window.addLast(now);
            return true;
        }
    }

    /**
     * Drops timestamps at or beyond the window edge relative to {@code now}.
     * Requests are appended in time order, so expired entries are always at the
     * head — a single forward scan suffices.
     */
    private static void pruneExpired(Deque<Instant> window, Instant now) {
        Instant cutoff = now.minus(WINDOW);
        while (!window.isEmpty() && !window.peekFirst().isAfter(cutoff)) {
            window.pollFirst();
        }
    }

    /**
     * Removes buckets whose rolling window has fully drained as of {@code now} —
     * every retained timestamp has aged out. A returning caller simply gets a
     * fresh window, so dropping an empty one is safe. Package-private for direct
     * testing.
     */
    void evictStale(Instant now) {
        buckets.entrySet().removeIf(e -> {
            Deque<Instant> window = e.getValue();
            synchronized (window) {
                pruneExpired(window, now);
                return window.isEmpty();
            }
        });
    }

    /** Test seam: current number of retained rate-limit buckets. */
    int bucketCount() {
        return buckets.size();
    }

    /** Test seam: seed a bucket carrying a single request made at {@code timestamp}. */
    void seedBucket(Long callerId, Instant timestamp) {
        Deque<Instant> window = new ArrayDeque<>();
        window.addLast(timestamp);
        buckets.put(callerId, window);
    }
}
