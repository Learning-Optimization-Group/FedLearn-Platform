package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;

/**
 * Immutable, globally content-addressed blob. The sha256 of the stored bytes IS the primary key,
 * so identical bytes from any run/org deduplicate to one row. Carries no tenant or provenance
 * semantics — those live on {@link ModelArtifact}, which points here (many artifacts : one blob).
 */
@Entity
@Table(name = "artifact_blobs")
public class ArtifactBlob {

    /** Lowercase-hex SHA-256 of the blob bytes. */
    @Id
    @Column(length = 64)
    private String sha256;

    @Column(name = "size_bytes", nullable = false)
    private long sizeBytes;

    /** Storage backend that holds the bytes: 'LOCAL_FS' | 'S3'. */
    @Column(nullable = false, length = 16)
    private String backend;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected ArtifactBlob() { }

    public ArtifactBlob(String sha256, long sizeBytes, String backend, Instant createdAt) {
        this.sha256 = sha256;
        this.sizeBytes = sizeBytes;
        this.backend = backend;
        this.createdAt = createdAt;
    }

    public String getSha256() { return sha256; }
    public void setSha256(String sha256) { this.sha256 = sha256; }
    public long getSizeBytes() { return sizeBytes; }
    public void setSizeBytes(long sizeBytes) { this.sizeBytes = sizeBytes; }
    public String getBackend() { return backend; }
    public void setBackend(String backend) { this.backend = backend; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
