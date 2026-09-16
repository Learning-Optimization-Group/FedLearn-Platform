package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "project_memberships")
public class ProjectMembership {

    @EmbeddedId
    private ProjectMembershipId id;

    @MapsId("projectId")
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "project_id", nullable = false)
    private Project project;

    @MapsId("userId")
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 32)
    private MembershipRole role;

    @Column(name = "partition_id")
    private Integer partitionId;

    @Enumerated(EnumType.STRING)
    @Column(name = "joined_via", nullable = false, length = 32)
    private JoinedVia joinedVia;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "added_by")
    private User addedBy;

    @Column(name = "added_at", nullable = false)
    private Instant addedAt;

    public ProjectMembership() {}

    public ProjectMembership(Project project, User user, MembershipRole role,
                             JoinedVia joinedVia, User addedBy) {
        this.id = new ProjectMembershipId(project.getId(), user.getId());
        this.project = project;
        this.user = user;
        this.role = role;
        this.joinedVia = joinedVia;
        this.addedBy = addedBy;
        this.addedAt = Instant.now();
    }

    public ProjectMembershipId getId() { return id; }
    public void setId(ProjectMembershipId id) { this.id = id; }
    public Project getProject() { return project; }
    public void setProject(Project project) { this.project = project; }
    public User getUser() { return user; }
    public void setUser(User user) { this.user = user; }
    public MembershipRole getRole() { return role; }
    public void setRole(MembershipRole role) { this.role = role; }
    public Integer getPartitionId() { return partitionId; }
    public void setPartitionId(Integer partitionId) { this.partitionId = partitionId; }
    public JoinedVia getJoinedVia() { return joinedVia; }
    public void setJoinedVia(JoinedVia joinedVia) { this.joinedVia = joinedVia; }
    public User getAddedBy() { return addedBy; }
    public void setAddedBy(User addedBy) { this.addedBy = addedBy; }
    public Instant getAddedAt() { return addedAt; }
    public void setAddedAt(Instant addedAt) { this.addedAt = addedAt; }
}
