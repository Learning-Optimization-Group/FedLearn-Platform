package com.federated.fl_platform_api.model;

/**
 * Where a run's training data comes from (V30), stated by its execution contract as {@code DataRequirement.source}.
 */
public enum TrainingDataSource {
    /** The recipe's committed fixture data, served to phones by the run's server: a test or demo run. */
    FIXTURE,
    /** Each participant's own dataset, imported on its device; the server serves no training data. */
    LOCAL_SNAPSHOT
}
