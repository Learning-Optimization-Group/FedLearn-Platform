package com.federated.fl_platform_api.model;


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

    @Column(nullable = false)
    private String password; // This will store the HASHED password

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