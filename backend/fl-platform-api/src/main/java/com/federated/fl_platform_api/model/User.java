package com.federated.fl_platform_api.model;


import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.*;
import java.time.Instant;
import java.util.Objects;

@Entity
@Table(name = "users")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, unique = true, length = 50)
    private String username;

    @Column(nullable = false, unique = true, length = 100)
    private String email;

    @JsonIgnore
    @Column(nullable = false)
    private String password; // This will store the HASHED password

    /**
     * Coarse platform-level role used for endpoint authorization (USER | ADMIN).
     * Mapped to a Spring Security {@code GrantedAuthority} of
     * {@code ROLE_<value>} by {@link com.federated.fl_platform_api.service.CustomUserDetailsService}.
     * Column renamed from {@code role} to {@code platform_role} in V5 migration.
     */
    @Column(name = "platform_role", nullable = false, length = 32)
    private String platformRole = "USER";

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 16)
    private UserStatus status = UserStatus.ACTIVE;

    @Column(name = "deleted_at")
    private Instant deletedAt;

    @Column(name = "email_verified", nullable = false)
    private Boolean emailVerified = false;

    @Column(name = "display_name", length = 120)
    private String displayName;

    @Column(name = "avatar_url", length = 512)
    private String avatarUrl;

    @Column(name = "last_login_at")
    private Instant lastLoginAt;

    @Column(name = "created_at", nullable = false, updatable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    // No-argument constructor (REQUIRED by JPA)
    public User() {
        // JPA requires a no-arg constructor
        // Initialize timestamps here if you want defaults upon new User()
        this.createdAt = Instant.now();
        this.updatedAt = Instant.now();
    }

    // Constructor for creating a new user before persistence (ID will be null)
    public User(String username, String email, String password) {
        this.username = username;
        this.email = email;
        this.password = password; // Plain password, will be hashed by service
        this.createdAt = Instant.now();
        this.updatedAt = Instant.now();
    }

    // Getters
    public Long getId() {
        return id;
    }

    public String getUsername() {
        return username;
    }

    public String getEmail() {
        return email;
    }

    public String getPassword() {
        return password;
    }

    public String getPlatformRole() {
        return platformRole;
    }

    public UserStatus getStatus() {
        return status;
    }

    public Instant getDeletedAt() {
        return deletedAt;
    }

    public Boolean getEmailVerified() {
        return emailVerified;
    }

    public String getDisplayName() {
        return displayName;
    }

    public String getAvatarUrl() {
        return avatarUrl;
    }

    public Instant getLastLoginAt() {
        return lastLoginAt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    // Setters
    public void setId(Long id) {
        this.id = id;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public void setPlatformRole(String platformRole) {
        this.platformRole = platformRole;
    }

    public void setStatus(UserStatus status) {
        this.status = status;
    }

    public void setDeletedAt(Instant deletedAt) {
        this.deletedAt = deletedAt;
    }

    public void setEmailVerified(Boolean emailVerified) {
        this.emailVerified = emailVerified;
    }

    public void setDisplayName(String displayName) {
        this.displayName = displayName;
    }

    public void setAvatarUrl(String avatarUrl) {
        this.avatarUrl = avatarUrl;
    }

    public void setLastLoginAt(Instant lastLoginAt) {
        this.lastLoginAt = lastLoginAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }

    // equals() and hashCode() for JPA Entities
    // Often based on the primary key (id) for persisted entities.
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        User user = (User) o;
        // If id is null, objects are equal only if they are the same instance
        // After persistence, id should be the basis of equality
        if (id == null || user.id == null) {
            return Objects.equals(username, user.username) && Objects.equals(email, user.email); // Or just false, or super.equals()
        }
        return Objects.equals(id, user.id);
    }

    @Override
    public int hashCode() {
        // Use a constant for unpersisted entities, or base on a business key if available
        // Once persisted, use the id.
        return id != null ? Objects.hash(id) : Objects.hash(username, email); // Simplified example
        // A common pattern: return getClass().hashCode(); if id is null
    }

    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", username='" + username + '\'' +
                ", email='" + email + '\'' +
                // Do NOT include password in toString() for security
                ", createdAt=" + createdAt +
                ", updatedAt=" + updatedAt +
                '}';
    }

}
