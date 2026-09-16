package com.federated.fl_platform_api.dto;

import java.util.List;
import java.util.UUID;

/**
 * The on-device training bundle a mobile client fetches for a run: the ExecuTorch loss/infer graphs +
 * the on-device data partition + the trainable-param layout, each with a sha256 the client verifies
 * before loading. File fields are URLs served by GET /api/runs/{runId}/files/{filename}; the client
 * downloads + stages them locally. Built from the staged manifest.json (scripts/stage_model_bundle.py).
 */
public record ModelBundleDto(
        UUID runId,
        List<ParamSpec> paramLayout,   // trainable name -> shape, flat (named_parameters requires_grad) order
        long totalParamCount,          // incl. frozen params, for the ModelInfo tier
        String lossPteUrl,             // forward(flat,x,y) -> loss graph
        String lossSha256,
        String inferPteUrl,            // forward(flat,x)   -> logits graph
        String inferSha256,
        String inputsUrl,              // row-major float32 features
        String inputsSha256,
        List<Integer> inputShape,
        String targetsUrl,             // int64 labels
        String targetsSha256,
        // First-order (FedAvg) trainable graph: forward(x,y) -> (loss, prediction) with a captured backward,
        // loadable by ET's TrainingModule. null / empty when this run's bundle is DeComFL-only (no on-device
        // first-order) — the mobile client treats a null trainablePteUrl exactly as "DeComFL-only".
        String trainablePteUrl,
        String trainableSha256,
        List<String> trainableParamNames) {  // canonical base.<name> order the phone re-maps ET's map onto

    /** One trainable tensor's layout (mirrors the mobile ModelManifest.ParamSpec). */
    public record ParamSpec(String name, List<Integer> shape) {
    }
}
