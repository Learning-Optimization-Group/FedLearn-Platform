package com.federated.fl_platform_api.model;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Feasibility of a Robust rule when a round can aggregate anywhere from minClients to clientsPerRound updates: it
 * completes at clientsPerRound, or at the deadline with fewer. The rule has to run at every size in between, and
 * Bulyan's precondition is not monotone in n.
 */
class RobustAggregationSettingsTest {

    private static RobustAggregationSettings rule(RobustMethod method, double fraction) {
        return new RobustAggregationSettings(method, fraction, null, null);
    }

    @Test
    void anEqualRoundSizeGivesTheSameAnswerAndWordingAsBefore() {
        RobustAggregationSettings s = rule(RobustMethod.BULYAN, 0.2);
        assertThat(s.refusalReason(10, 10)).isEqualTo(s.refusalReason(10));
        assertThat(s.refusalReason(10, 10).orElseThrow()).contains("minClients = 10");
    }

    @Test
    void aRuleFeasibleAtEveryRoundSizeInTheRangeRuns() {
        // Krum at 0.2: f = int(0.2 n) is 1 for n in 7..9 and 2 for n in 10..12, and n >= 2f + 3 holds throughout.
        assertThat(rule(RobustMethod.KRUM, 0.2).refusalReason(7, 12)).isEmpty();
    }

    @Test
    void aRoundSizeInsideTheRangeWhereTheRuleCannotRunIsRefused() {
        // Bulyan at 0.25 runs at n = 23 (f = 5, needs 23) but not at n = 24 (f = 6, needs 27), so a round that ends
        // with 24 updates would be refused even though minClients alone is fine.
        RobustAggregationSettings s = rule(RobustMethod.BULYAN, 0.25);
        assertThat(s.refusalReason(23)).isEmpty();
        assertThat(s.refusalReason(23, 24).orElseThrow())
                .contains("BULYAN").contains("24 updates").contains("minClients = 23").contains("clientsPerRound = 24");
    }

    @Test
    void theToleranceCheckDoesNotDependOnTheRoundSize() {
        assertThat(rule(RobustMethod.BULYAN, 0.3).refusalReason(40, 60).orElseThrow()).contains("tolerance");
    }

    @Test
    void aRuleWithoutACohortPreconditionRunsAtAnyRange() {
        assertThat(rule(RobustMethod.MEDIAN, 0.4).refusalReason(1, 50)).isEmpty();
    }
}
