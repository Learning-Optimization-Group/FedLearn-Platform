package com.federated.fl_platform_api.simulation;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.Instant;
import java.util.UUID;

/** Private persistence model. Only explicit projections cross the HTTP boundary. */
@Entity
@Table(name = "simulation_runs")
@Getter
@Setter
public class SimulationRun {
    @Id private UUID id;
    @Version private Long version;
    @Column(name = "created_by", nullable = false) private Long createdBy;
    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20) private SimulationStatus status;
    @Column(name = "config_hash", nullable = false, length = 64) private String configHash;
    @Column(name = "recipe_json", nullable = false, columnDefinition = "text") private String recipeJson;
    @Column(name = "progress_json", columnDefinition = "text") private String progressJson;
    @Column(name = "record_json", columnDefinition = "text") private String recordJson;
    @Column(name = "created_at", nullable = false) private Instant createdAt;
    @Column(name = "started_at") private Instant startedAt;
    @Column(name = "ended_at") private Instant endedAt;
    @Column(name = "requested_rounds", nullable = false) private Integer requestedRounds;
    @Column(name = "completed_rounds") private Integer completedRounds;
    @Column(name = "worker_pid") private Long workerPid;
    @Column(name = "worker_start_identity", length = 80) private String workerStartIdentity;
    @Column(name = "worker_executable_identity", length = 1024) private String workerExecutableIdentity;
    @Column(name = "exit_code") private Integer exitCode;
    @Column(name = "error_code", length = 60) private String errorCode;
    @Column(name = "error_message", length = 240) private String errorMessage;
}
