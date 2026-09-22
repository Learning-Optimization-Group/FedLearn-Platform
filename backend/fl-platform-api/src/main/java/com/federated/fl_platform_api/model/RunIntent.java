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
                        boolean clientAuthRequired) {

    /** The snapshot layout stored in runs.intent_version. */
    public static final int VERSION = 1;

    public RunIntent {
        Objects.requireNonNull(trainingArm, "trainingArm");
        Objects.requireNonNull(modelName, "modelName");
    }

    public static RunIntent capture(Project project, boolean tlsRequired, boolean clientAuthRequired) {
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
                clientAuthRequired);
    }
}
