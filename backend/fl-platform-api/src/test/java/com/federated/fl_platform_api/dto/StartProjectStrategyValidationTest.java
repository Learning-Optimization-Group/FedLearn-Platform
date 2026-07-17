package com.federated.fl_platform_api.dto;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The StartProject strategy allowlist. Exposes the strategies that produce a WORKING run when
 * selected — FedOpt (server-side adaptive) and Robust (Byzantine-robust median) are now accepted
 * alongside FedAvg/DeComFL/FoT. FedProx stays rejected on purpose: the production client refuses a
 * proximal_mu it cannot honor (FR-20), so exposing it would be a broken option until FR-32.
 */
class StartProjectStrategyValidationTest {
    private final Validator v = Validation.buildDefaultValidatorFactory().getValidator();

    /** True iff the @Pattern on the strategy field rejects `strategy` (ignores other field violations). */
    private boolean strategyRejected(String strategy) {
        StartProject p = new StartProject();
        p.setStrategy(strategy);
        return v.validate(p).stream()
                .anyMatch(cv -> "strategy".equals(cv.getPropertyPath().toString()));
    }

    @Test void fedAvgAccepted()  { assertFalse(strategyRejected("FedAvg")); }
    @Test void deComflAccepted() { assertFalse(strategyRejected("DeComFL")); }
    @Test void fotAccepted()     { assertFalse(strategyRejected("FoT")); }

    // Newly exposed — both run end-to-end when selected.
    @Test void fedOptAccepted()  { assertFalse(strategyRejected("FedOpt")); }
    @Test void robustAccepted()  { assertFalse(strategyRejected("Robust")); }

    // Deliberately still rejected.
    @Test void fedProxRejected() { assertTrue(strategyRejected("FedProx")); }  // client refuses it (FR-20)
    @Test void unknownRejected() { assertTrue(strategyRejected("Nonsense")); }
}
