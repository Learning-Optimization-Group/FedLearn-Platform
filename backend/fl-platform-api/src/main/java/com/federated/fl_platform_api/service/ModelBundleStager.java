package com.federated.fl_platform_api.service;

import java.util.UUID;

/**
 * MO-15: seam for auto-staging a run's on-device model bundle when a run starts, so a mobile client
 * that joins finds its bundle at {@code GET /api/runs/{runId}/model-bundle} without an operator having
 * to run {@code scripts/stage_model_bundle.py} by hand (the manual step that blocked the live phone test).
 *
 * <p><b>Best-effort by contract:</b> implementations MUST NOT throw. A missing bundle is a graceful 404
 * the phone handles (it only affects mobile clients; desktop/laptop/Docker clients get their model from
 * the FL server over gRPC and never touch the bundle endpoint). Staging failure must therefore never
 * fail a project start. Implementations SHOULD be idempotent (skip if the run is already staged).</p>
 *
 * <p>The seam exists so {@link ProjectService}'s start path can be unit-tested with a fake (no real
 * child process), mirroring the DA-8 {@code FlServerProcessRunner} runner seam.</p>
 */
public interface ModelBundleStager {

    /**
     * Stage the on-device bundle for {@code runId}, whose model is the recipe {@code recipeKey}
     * (the project's model type). Never throws — failures are logged and swallowed.
     */
    void stageForRun(UUID runId, String recipeKey);
}
