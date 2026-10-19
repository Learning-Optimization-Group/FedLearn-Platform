package com.federated.fl_platform_api.model;

/** Resolved server optimizer settings; these do not alter the client's training contract. */
public record FedOptSettings(String variant, Double serverLearningRate, Double beta1, Double beta2, Double tau) {
    public FedOptSettings {
        variant = variant == null ? "adam" : variant;
        serverLearningRate = serverLearningRate == null ? 1.0 : serverLearningRate;
        beta1 = beta1 == null ? .9 : beta1;
        beta2 = beta2 == null ? .99 : beta2;
        tau = tau == null ? .001 : tau;
        if (!java.util.Set.of("adam", "yogi").contains(variant)
                || !Double.isFinite(serverLearningRate) || serverLearningRate <= 0 || serverLearningRate > 10
                || !Double.isFinite(beta1) || beta1 < 0 || beta1 >= 1
                || !Double.isFinite(beta2) || beta2 < 0 || beta2 >= 1
                || !Double.isFinite(tau) || tau <= 0 || tau > 1) {
            throw new IllegalArgumentException("Invalid FedOpt settings: rate in (0,10], betas in [0,1), tau in (0,1], variant adam/yogi");
        }
    }
    public static FedOptSettings defaults() { return new FedOptSettings(null, null, null, null, null); }
}
