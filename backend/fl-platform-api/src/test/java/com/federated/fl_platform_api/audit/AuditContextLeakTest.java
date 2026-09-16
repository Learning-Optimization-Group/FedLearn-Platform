package com.federated.fl_platform_api.audit;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Guards the contract that {@link AuditContext#drain()} both returns the staged
 * metadata <i>and</i> clears the thread-local, so a subsequent request on the same
 * pooled thread starts with an empty context (no cross-request leak).
 */
class AuditContextLeakTest {

    @Test
    void drain_returns_value_then_clears() {
        AuditContext.put("k", "v");

        Map<String, String> first = AuditContext.drain();
        assertThat(first).containsEntry("k", "v");

        // Second drain must be empty: the thread-local was cleared by the first drain.
        Map<String, String> second = AuditContext.drain();
        assertThat(second).isEmpty();
    }
}
