package com.federated.fl_platform_api.model;

import java.util.Objects;

/**
 * The intent a run was started with: the project-derived training settings and the effective TLS and client-auth
 * deployment settings, captured once at start (V27). The FL server is spawned from it and the execution contract is
 * built from it, so a later project edit or configuration change cannot alter what an active run executes or
 * claims.
 *
 * <p>Privacy values are recorded only when central DP is on. Completeness of an enabled DP configuration is checked
 * where it is used (the spawn refuses an incomplete one), not here, so a refused start still records what was
 * requested.</p>
 */
public record RunIntent(TrainingArm trainingArm, String modelName, String taskType, boolean dpEnabled,
                        Double dpTargetEpsilon, Double dpDelta, Double dpClipNorm, boolean tlsRequired,
                        boolean clientAuthRequired, Long roundTimeoutMs, TrainingDataSource dataSource) {

    /**
     * The snapshot layouts stored in runs.intent_version: version 2 records the FL server's round timeout
     * (V29); version 1, taken before that, has none.
     */
    public static final int VERSION_WITHOUT_ROUND_TIMEOUT = 1;
    public static final int VERSION_WITHOUT_DATA_SOURCE = 2;
    /** Version 3 (V30) also records the data source. */
    public static final int VERSION = 3;

    public RunIntent {
        Objects.requireNonNull(trainingArm, "trainingArm");
        Objects.requireNonNull(modelName, "modelName");
        if (roundTimeoutMs != null && roundTimeoutMs <= 0) {
            throw new IllegalArgumentException("roundTimeoutMs must be positive, not " + roundTimeoutMs);
        }
        if (dataSource != null && roundTimeoutMs == null) {
            throw new IllegalArgumentException("a data source is recorded only in a version-3 snapshot, with a round timeout");
        }
    }

    /** An intent without a recorded data source, as a version-2 (or, without a timeout, version-1) snapshot reads. */
    public RunIntent(TrainingArm trainingArm, String modelName, String taskType, boolean dpEnabled,
                     Double dpTargetEpsilon, Double dpDelta, Double dpClipNorm, boolean tlsRequired,
                     boolean clientAuthRequired, Long roundTimeoutMs) {
        this(trainingArm, modelName, taskType, dpEnabled, dpTargetEpsilon, dpDelta, dpClipNorm, tlsRequired,
                clientAuthRequired, roundTimeoutMs, null);
    }

    /** An intent without a recorded round timeout, as a version-1 snapshot reads. */
    public RunIntent(TrainingArm trainingArm, String modelName, String taskType, boolean dpEnabled,
                     Double dpTargetEpsilon, Double dpDelta, Double dpClipNorm, boolean tlsRequired,
                     boolean clientAuthRequired) {
        this(trainingArm, modelName, taskType, dpEnabled, dpTargetEpsilon, dpDelta, dpClipNorm, tlsRequired,
                clientAuthRequired, null);
    }

    public static RunIntent capture(Project project, boolean tlsRequired, boolean clientAuthRequired,
                                    Long roundTimeoutMs) {
        return capture(project, tlsRequired, clientAuthRequired, roundTimeoutMs, null);
    }

    public static RunIntent capture(Project project, boolean tlsRequired, boolean clientAuthRequired,
                                    Long roundTimeoutMs, TrainingDataSource dataSource) {
        boolean dp = project.isDpEnabled();
        return new RunIntent(
                project.getTrainingArm() != null ? project.getTrainingArm() : TrainingArm.FULL,
                project.getModelName(),
                project.getTaskType(),
                dp,
                dp ? project.getDpTargetEpsilon() : null,
                dp ? project.getDpDelta() : null,
                dp ? project.getDpClipNorm() : null,
                tlsRequired,
                clientAuthRequired,
                roundTimeoutMs,
                dataSource);
    }

    /** Captures an intent without a round timeout, for callers to which the timeout is irrelevant. */
    public static RunIntent capture(Project project, boolean tlsRequired, boolean clientAuthRequired) {
        return capture(project, tlsRequired, clientAuthRequired, null);
    }

    /** Converts a round timeout in seconds, as the backend and the FL server configure it, to milliseconds. */
    public static long roundTimeoutMs(double seconds) {
        if (!(seconds > 0) || Double.isInfinite(seconds)) {
            throw new IllegalArgumentException("the FL round timeout must be a positive number of seconds, not "
                    + seconds);
        }
        return Math.round(seconds * 1000);
    }
}
