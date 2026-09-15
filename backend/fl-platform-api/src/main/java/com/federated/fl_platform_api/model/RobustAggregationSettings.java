package com.federated.fl_platform_api.model;

import java.util.Optional;

/**
 * The Byzantine-robust rule a Robust run uses, and the settings actually sent to the server.
 *
 * <p>A {@code null} parameter means "not sent": {@code fl_server.py} then applies its own default (fraction
 * 0.0, trim ratio {@link RobustMethod#DEFAULT_TRIM_RATIO}, clipping radius 1.0). One instance is persisted on
 * the run and turned into argv, so the run record and the spawned server cannot disagree.
 *
 * @param method            the rule; never null
 * @param byzantineFraction estimated malicious share in [0, 0.5); sizes f for Krum, Multi-Krum and Bulyan
 * @param trimRatio         trimmed-mean only, in [0, 0.5)
 * @param centeredClipTau   centered clipping only, &gt; 0
 */
public record RobustAggregationSettings(RobustMethod method, Double byzantineFraction, Double trimRatio,
                                        Double centeredClipTau) {

    public RobustAggregationSettings {
        if (method == null) {
            throw new IllegalArgumentException("Robust aggregation settings require a method");
        }
    }

    /**
     * Why the server would refuse every round at {@code minClients}, or empty if it can run. Unset values are
     * evaluated at the server's defaults, because those are what the server will use.
     */
    public Optional<String> refusalReason(int minClients) {
        double fraction = byzantineFraction == null ? 0.0 : byzantineFraction;
        double trim = trimRatio == null ? RobustMethod.DEFAULT_TRIM_RATIO : trimRatio;
        return method.refusalReason(fraction, trim, minClients);
    }
}
