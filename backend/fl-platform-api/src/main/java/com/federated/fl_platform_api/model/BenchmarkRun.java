package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import lombok.Data;

import java.time.Instant;
import java.util.UUID;

/**
 * Denormalized one-row-per-project benchmark rollup, recomputed from
 * {@link BenchmarkRound} rows on every ingest. Powers the dashboard landing
 * view as a single cheap read (no per-request aggregation over round rows).
 */
@Entity
@Data
@Table(name = "benchmark_runs")
public class BenchmarkRun {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @Column(name = "project_id", nullable = false, unique = true)
    private UUID projectId;

    @Column(name = "run_id")
    private UUID runId;

    @Column(name = "project_name")
    private String projectName;

    @Column(name = "model_type")
    private String modelType;

    @Column(name = "task_type")
    private String taskType;

    @Column(name = "rounds_completed")  private Integer roundsCompleted;
    @Column(name = "final_loss")        private Double finalLoss;
    @Column(name = "final_accuracy")    private Double finalAccuracy;
    @Column(name = "best_accuracy")     private Double bestAccuracy;
    @Column(name = "best_round")        private Integer bestRound;
    @Column(name = "final_f1_macro")    private Double finalF1Macro;
    @Column(name = "final_perplexity")  private Double finalPerplexity;
    @Column(name = "best_perplexity")   private Double bestPerplexity;
    @Column(name = "final_ece")         private Double finalEce;
    @Column(name = "target_accuracy")   private Double targetAccuracy;
    @Column(name = "rounds_to_target")  private Integer roundsToTarget;
    @Column(name = "ms_to_target")      private Long msToTarget;
    @Column(name = "total_round_ms")    private Long totalRoundMs;
    @Column(name = "avg_round_ms")      private Long avgRoundMs;
    @Column(name = "model_size_mb")     private Double modelSizeMb;
    @Column(name = "param_count")       private Long paramCount;
    @Column(name = "client_count")      private Integer clientCount;

    @Column(name = "first_recorded_at") private Instant firstRecordedAt;
    @Column(name = "last_recorded_at")  private Instant lastRecordedAt;
}
