package com.federated.fl_platform_api.audit;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Thread-local sidecar for {@link Auditable @Auditable} methods to attach arbitrary
 * key/value metadata to the audit row that {@link AuditAspect} writes on return.
 *
 * <p>Entries added via {@link #put(String, String)} are drained by the aspect after
 * {@code pjp.proceed()} returns and serialised into {@code audit_events.metadata}.
 * The thread-local is cleared on drain to avoid leaking state across requests in a
 * pooled servlet container.
 */
public final class AuditContext {

    private static final ThreadLocal<Map<String, String>> CTX =
            ThreadLocal.withInitial(LinkedHashMap::new);

    private AuditContext() { }

    /** Adds a single key/value pair to the current thread's audit metadata. */
    public static void put(String key, String value) { CTX.get().put(key, value); }

    /**
     * Returns a snapshot of the current thread's metadata and clears the underlying
     * thread-local. Called by {@link AuditAspect} immediately before persisting.
     */
    public static Map<String, String> drain() {
        Map<String, String> snapshot = new LinkedHashMap<>(CTX.get());
        CTX.get().clear();
        return snapshot;
    }
}
