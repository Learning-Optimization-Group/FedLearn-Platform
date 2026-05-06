package com.federated.fl_platform_api.model;
import jakarta.persistence.*;
import lombok.Data;
import java.time.Instant;
import java.util.UUID;

@Entity
@Data
@Table(name = "server_logs")
public class ServerLog {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "project_id", nullable = false)
    private UUID projectId;
    
    @Column(nullable = false)
    private String level;
    
    @Column(columnDefinition = "TEXT", nullable = false)
    private String message;
    
    @Column(columnDefinition = "TEXT")
    private String stackTrace;
    
    @Column(nullable = false)
    private Instant timestamp;
}
