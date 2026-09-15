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
        return method.refusalReason(effectiveFraction(), effectiveTrim(), minClients);
    }

    /**
     * Why the server would refuse a round of some size this run allows, or empty if the rule runs at all of them. A
     * round completes once {@code clientsPerRound} updates arrive and can still finish with {@code minClients} at the
     * deadline, so the rule has to run at every size in between, and Bulyan's precondition is not monotone in n.
     * With the two equal this is exactly {@link #refusalReason(int)}, wording included.
     */
    public Optional<String> refusalReason(int minClients, int clientsPerRound) {
        if (clientsPerRound <= minClients) {
            return refusalReason(minClients);
        }
        double fraction = effectiveFraction();
        double trim = effectiveTrim();
        if (fraction > method.tolerance(trim)) {
            return method.refusalReason(fraction, trim, minClients);
        }
        for (int n = minClients; n <= clientsPerRound; n++) {
            int needed = method.updatesNeeded(fraction, n);
            if (n < needed) {
                return Optional.of(String.format(
                        "%s cannot aggregate a round of %d updates: with byzantineFraction=%s it needs at least %d. "
                                + "With minClients = %d and clientsPerRound = %d a round can end with anywhere from %d "
                                + "to %d updates, depending on how many clients drop out, so the server would refuse "
                                + "any round that ends with %d. Change minClients or clientsPerRound, lower "
                                + "byzantineFraction, or choose another rule.",
                        method.name(), n, fraction, needed, minClients, clientsPerRound, minClients, clientsPerRound,
                        n));
            }
        }
        return Optional.empty();
    }

    private double effectiveFraction() {
        return byzantineFraction == null ? 0.0 : byzantineFraction;
    }

    private double effectiveTrim() {
        return trimRatio == null ? RobustMethod.DEFAULT_TRIM_RATIO : trimRatio;
    }
}
