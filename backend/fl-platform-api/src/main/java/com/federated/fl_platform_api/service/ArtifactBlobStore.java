package com.federated.fl_platform_api.service;

/**
 * Content-addressed, write-once blob storage — the storage half of "write-new-not-overwrite".
 *
 * <p>The key IS the sha256 of the content, computed by the store (never trusted from a caller).
 * {@link #put(byte[])} is idempotent: writing identical bytes to an existing key is a no-op success,
 * which is exactly how content dedup works. An existing key is never overwritten with different
 * bytes (it cannot be — a different content produces a different key).
 */
public interface ArtifactBlobStore {

    /** Store {@code content} keyed by its sha256; returns the lowercase-hex key. Idempotent. */
    String put(byte[] content);

    /** Read the bytes stored under {@code sha256}. */
    byte[] get(String sha256);

    /** True if a blob with this key exists. */
    boolean exists(String sha256);

    /** Backend id for the artifact_blobs.backend column, e.g. "LOCAL_FS". */
    String backendId();
}
