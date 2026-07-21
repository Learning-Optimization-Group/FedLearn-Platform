package com.federated.fl_platform_api.dto;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The StartProject strategy allowlist. Exposes the strategies that produce a WORKING run when
 * selected — FedOpt (server-side adaptive) and Robust (Byzantine-robust median) are now accepted
 * alongside FedAvg/DeComFL/FoT. FR-32: FedProx is now accepted too — the production client honors the
 * proximal term mu*(w - w_global), so it produces a real FedProx run rather than a mislabeled FedAvg.
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
    // FR-32: the production client now honors the proximal term, so FedProx is a working option.
    @Test void fedProxAccepted() { assertFalse(strategyRejected("FedProx")); }

    // Deliberately still rejected.
    @Test void unknownRejected() { assertTrue(strategyRejected("Nonsense")); }
}
