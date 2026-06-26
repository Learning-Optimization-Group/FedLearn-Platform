package com.federated.fl_platform_api.model;

import jakarta.persistence.*;
import lombok.Data;

import java.time.Instant;
import java.util.UUID;

/**
 * One round's rich benchmark vector, written by scripts/benchmarks.py via the
 * internal ingest. Decoupled from {@link RoundResult} (the lightweight live
 * telemetry path); this row carries the full model-quality + system-efficiency
 * metric set the admin benchmarking dashboard renders.
 */
@Entity
@Data
@Table(name = "benchmark_rounds")
public class BenchmarkRound {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @Column(name = "project_id", nullable = false)
    private UUID projectId;

    @Column(name = "run_id")
    private UUID runId;

    @Column(name = "server_round", nullable = false)
    private Integer serverRound;

    @Column(name = "model_type")
    private String modelType;

    @Column(name = "task_type")
    private String taskType;

    // ── Model quality (macro / run-level) ────────────────────────────────────
    private Double loss;
    private Double accuracy;
    @Column(name = "balanced_accuracy")   private Double balancedAccuracy;
    @Column(name = "precision_macro")     private Double precisionMacro;
    @Column(name = "recall_macro")        private Double recallMacro;
    @Column(name = "f1_macro")            private Double f1Macro;
    @Column(name = "precision_micro")     private Double precisionMicro;
    @Column(name = "recall_micro")        private Double recallMicro;
    @Column(name = "f1_micro")            private Double f1Micro;
    @Column(name = "precision_weighted")  private Double precisionWeighted;
    @Column(name = "recall_weighted")     private Double recallWeighted;
    @Column(name = "f1_weighted")         private Double f1Weighted;
    private Double mcc;
    @Column(name = "cohen_kappa")         private Double cohenKappa;
    @Column(name = "roc_auc")             private Double rocAuc;
    @Column(name = "log_loss")            private Double logLoss;
    private Double perplexity;
    @Column(name = "token_accuracy")      private Double tokenAccuracy;
    private Double ece;
    private Double brier;
    @Column(name = "target_accuracy")     private Double targetAccuracy;

    // ── System / efficiency ──────────────────────────────────────────────────
    @Column(name = "round_duration_ms")   private Long roundDurationMs;
    @Column(name = "eval_duration_ms")     private Long evalDurationMs;
    @Column(name = "model_size_mb")        private Double modelSizeMb;
    @Column(name = "param_count")          private Long paramCount;
    @Column(name = "client_count")         private Integer clientCount;
    @Column(name = "samples_evaluated")    private Integer samplesEvaluated;
    @Column(name = "num_classes")          private Integer numClasses;

    // ── Micro / structured (JSON text) ───────────────────────────────────────
    @Column(name = "per_class_json", columnDefinition = "TEXT")
    private String perClassJson;

    @Column(name = "confusion_matrix_json", columnDefinition = "TEXT")
    private String confusionMatrixJson;

    @Column(name = "class_labels_json", columnDefinition = "TEXT")
    private String classLabelsJson;

    @Column(name = "extra_metrics_json", columnDefinition = "TEXT")
    private String extraMetricsJson;

    @Column(name = "recorded_at", nullable = false)
    private Instant recordedAt;
}
