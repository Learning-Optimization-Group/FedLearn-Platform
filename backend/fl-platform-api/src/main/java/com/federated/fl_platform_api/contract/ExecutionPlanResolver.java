package com.federated.fl_platform_api.contract;

import com.fedlearn.contract.v1.ModelTraining;

import java.io.IOException;
import java.nio.file.Path;

/** Resolves the Python-owned part of a run's execution contract (fl-runtime/execution_plan.py). */
public interface ExecutionPlanResolver {

    /**
     * The resolved plan for a run configuration, with the digest of the initial model the FL server loads from
     * {@code initialState}.
     *
     * @throws NotRepresentableException the configuration has no v1 plan, or the initial model disagrees with it
     * @throws IOException               the resolver could not be run or its output could not be read
     */
    ModelTraining resolve(String recipeKey, String strategy, String trainingArm, Path initialState)
            throws NotRepresentableException, IOException;
}
