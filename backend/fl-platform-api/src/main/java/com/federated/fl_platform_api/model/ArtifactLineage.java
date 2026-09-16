package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;
import java.util.UUID;

/**
 * A directed provenance edge in the artifact DAG: {@code child} was produced from {@code parent}
 * under {@code relationship}. Surrogate UUID PK with a UNIQUE(child, parent, relationship) triple;
 * a DB CHECK forbids self-loops. FKs to model_artifacts are ON DELETE RESTRICT so lineage never
 * dangles (artifacts are append-only).
 */
@Entity
@Table(name = "artifact_lineage")
public class ArtifactLineage {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @Column(name = "child_id", nullable = false)
    private UUID childId;

    @Column(name = "parent_id", nullable = false)
    private UUID parentId;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private LineageRelationship relationship;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected ArtifactLineage() { }

    public ArtifactLineage(UUID childId, UUID parentId, LineageRelationship relationship, Instant createdAt) {
        this.childId = childId;
        this.parentId = parentId;
        this.relationship = relationship;
        this.createdAt = createdAt;
    }

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public UUID getChildId() { return childId; }
    public void setChildId(UUID childId) { this.childId = childId; }
    public UUID getParentId() { return parentId; }
    public void setParentId(UUID parentId) { this.parentId = parentId; }
    public LineageRelationship getRelationship() { return relationship; }
    public void setRelationship(LineageRelationship relationship) { this.relationship = relationship; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}
