package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.UserSearchResultDto;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;
import java.util.Collections;
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

    private final ConcurrentHashMap<Long, RateState> buckets = new ConcurrentHashMap<>();

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
        long nowMin = Instant.now().getEpochSecond() / 60;
        // Opportunistic eviction: only when the map has grown past the cap do we
        // pay for a sweep, dropping buckets whose rate-limit window has already
        // expired (minute < now). Keeps the common path a single map lookup.
        if (buckets.size() > MAX_BUCKETS) {
            evictStale(nowMin);
        }
        RateState s = buckets.computeIfAbsent(callerId, k -> new RateState(nowMin, 0));
        synchronized (s) {
            if (s.minute != nowMin) {
                s.minute = nowMin;
                s.count = 0;
            }
            if (s.count >= MAX_REQUESTS_PER_MINUTE) return false;
            s.count++;
            return true;
        }
    }

    /**
     * Removes buckets whose rate-limit window is older than {@code nowMin}. A
     * stale bucket carries no live count for the current minute, so dropping it
     * is safe — a returning caller simply gets a fresh bucket. Package-private
     * for direct testing.
     */
    void evictStale(long nowMin) {
        buckets.entrySet().removeIf(e -> e.getValue().minute < nowMin);
    }

    /** Test seam: current number of retained rate-limit buckets. */
    int bucketCount() {
        return buckets.size();
    }

    /** Test seam: seed a bucket whose window is {@code minute}. */
    void seedBucket(Long callerId, long minute) {
        buckets.put(callerId, new RateState(minute, 0));
    }

    private static class RateState {
        long minute;
        int count;
        RateState(long m, int c) { this.minute = m; this.count = c; }
    }
}
