-- Benchmarking & observability suite.
--
-- Two tables, decoupled from round_result (which stays the lightweight live-
-- telemetry path). benchmark_rounds holds the rich per-round metric vector
-- (model quality + system efficiency) computed by scripts/benchmarks.py;
-- benchmark_runs is a denormalized one-row-per-project rollup so the admin
-- dashboard's landing view is a single cheap read.
--
-- Portability: only types already used in this schema (UUID, DOUBLE PRECISION,
-- BIGINT, INTEGER, TEXT, TIMESTAMP WITH TIME ZONE). No DB-side UUID default —
-- ids are assigned by JPA (GenerationType.AUTO), mirroring runs/round_result.

CREATE TABLE benchmark_rounds (
    id                    UUID    PRIMARY KEY,
    project_id            UUID    NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id                UUID,
    server_round          INTEGER NOT NULL,
    model_type            VARCHAR(64),
    task_type             VARCHAR(32),

    -- Model quality (macro / run-level scalars). NULL where not applicable to
    -- the task (e.g. accuracy is NULL for a generative run, perplexity for a
    -- classification run).
    loss                  DOUBLE PRECISION,
    accuracy              DOUBLE PRECISION,
    balanced_accuracy     DOUBLE PRECISION,
    precision_macro       DOUBLE PRECISION,
    recall_macro          DOUBLE PRECISION,
    f1_macro              DOUBLE PRECISION,
    precision_micro       DOUBLE PRECISION,
    recall_micro          DOUBLE PRECISION,
    f1_micro              DOUBLE PRECISION,
    precision_weighted    DOUBLE PRECISION,
    recall_weighted       DOUBLE PRECISION,
    f1_weighted           DOUBLE PRECISION,
    mcc                   DOUBLE PRECISION,
    cohen_kappa           DOUBLE PRECISION,
    roc_auc               DOUBLE PRECISION,
    log_loss              DOUBLE PRECISION,
    perplexity            DOUBLE PRECISION,
    token_accuracy        DOUBLE PRECISION,
    ece                   DOUBLE PRECISION,   -- expected calibration error
    brier                 DOUBLE PRECISION,   -- multiclass Brier score
    target_accuracy       DOUBLE PRECISION,   -- per-recipe TTA target (if any)

    -- System / efficiency metrics.
    round_duration_ms     BIGINT,
    eval_duration_ms      BIGINT,
    model_size_mb         DOUBLE PRECISION,
    param_count           BIGINT,
    client_count          INTEGER,
    samples_evaluated     INTEGER,
    num_classes           INTEGER,

    -- Micro / structured metrics serialized as JSON text (per-class table,
    -- confusion matrix, class labels, and an open-ended extension bag so new
    -- metrics never require a schema change).
    per_class_json        TEXT,
    confusion_matrix_json TEXT,
    class_labels_json     TEXT,
    extra_metrics_json    TEXT,

    recorded_at           TIMESTAMP WITH TIME ZONE NOT NULL,

    CONSTRAINT uq_benchmark_round UNIQUE (project_id, server_round)
);

CREATE INDEX idx_benchmark_rounds_project ON benchmark_rounds(project_id);

CREATE TABLE benchmark_runs (
    id                    UUID    PRIMARY KEY,
    project_id            UUID    NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id                UUID,
    project_name          VARCHAR(255),
    model_type            VARCHAR(64),
    task_type             VARCHAR(32),

    rounds_completed      INTEGER,
    final_loss            DOUBLE PRECISION,
    final_accuracy        DOUBLE PRECISION,
    best_accuracy         DOUBLE PRECISION,
    best_round            INTEGER,
    final_f1_macro        DOUBLE PRECISION,
    final_perplexity      DOUBLE PRECISION,
    best_perplexity       DOUBLE PRECISION,
    final_ece             DOUBLE PRECISION,
    -- Time-to-target-accuracy (the canonical FL system headline).
    target_accuracy       DOUBLE PRECISION,
    rounds_to_target      INTEGER,
    ms_to_target          BIGINT,
    total_round_ms        BIGINT,
    avg_round_ms          BIGINT,
    model_size_mb         DOUBLE PRECISION,
    param_count           BIGINT,
    client_count          INTEGER,

    first_recorded_at     TIMESTAMP WITH TIME ZONE,
    last_recorded_at      TIMESTAMP WITH TIME ZONE,

    CONSTRAINT uq_benchmark_run_project UNIQUE (project_id)
);

CREATE INDEX idx_benchmark_runs_project ON benchmark_runs(project_id);
