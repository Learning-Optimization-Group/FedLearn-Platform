package com.federated.fl_platform_api.model;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * The start-time feasibility check for a Robust run must say exactly what the Python server will do.
 *
 * <p>These rules mirror {@code framework/src/fedlearn/server/robust_aggregation.py} and the n that
 * {@code fl_server.py} actually aggregates: {@code RobustAggregator} is built with only
 * {@code min_fit_clients}, so {@code clients_per_round} falls back to it and every round aggregates
 * exactly {@code minClients}. A rule that cannot run at that n does not fail loudly on the Python side -
 * {@code aggregate_fit} returns None every round, the global model never moves, and the run "completes"
 * at roughly random accuracy. The FR-12 breakdown benchmark hit exactly that and nearly recorded it as a
 * measured breakdown point. Refusing at start is the only place a user sees the reason.
 *
 * <p>Two runtime guards are mirrored, in the order Python applies them:
 * <ul>
 *   <li>tolerance: {@code byzantine_fraction > tolerance} (strict). For trimmed-mean the tolerance IS
 *       the trim ratio; otherwise the rule's breakdown point.</li>
 *   <li>cohort precondition, with {@code f = int(fraction * n)}: Krum / Multi-Krum need
 *       {@code n >= 2f + 3}, Bulyan needs {@code n >= 4f + 3}.</li>
 * </ul>
 */
class RobustMethodTest {

    private static final double DEFAULT_TRIM = RobustMethod.DEFAULT_TRIM_RATIO;

    @Test
    void wireNamesMatchTheFlServerChoices() {
        assertThat(RobustMethod.values()).extracting(RobustMethod::wireName)
                .containsExactly("median", "trimmed_mean", "krum", "multi_krum", "bulyan", "centered_clip");
        assertThat(RobustMethod.fromWire("multi_krum")).isEqualTo(RobustMethod.MULTI_KRUM);
    }

    @Test
    void bulyanAtTwentyClientsAndOneFifthAttackersRuns() {
        // f = int(0.2 * 20) = 4, needs 4*4+3 = 19 <= 20. Measured feasible in the n=20 sweep.
        assertThat(RobustMethod.BULYAN.refusalReason(0.2, DEFAULT_TRIM, 20)).isEmpty();
    }

    @Test
    void bulyanPreconditionIsNotMonotoneInCohortSize() {
        // f = int(0.2 * n): n=9 -> f=1 needs 7 (ok); n=10 -> f=2 needs 11 (refused); n=11 -> needs 11 (ok).
        // Checking "enough clients" as a single lower bound would get n=10 wrong.
        assertThat(RobustMethod.BULYAN.refusalReason(0.2, DEFAULT_TRIM, 9)).isEmpty();
        assertThat(RobustMethod.BULYAN.refusalReason(0.2, DEFAULT_TRIM, 10)).isPresent();
        assertThat(RobustMethod.BULYAN.refusalReason(0.2, DEFAULT_TRIM, 11)).isEmpty();
    }

    @ParameterizedTest
    @ValueSource(ints = {4, 10, 20, 40, 100, 1000})
    void bulyanAtAQuarterIsRefusedWhenTheCohortIsNotThreeModFour(int n) {
        // At fraction 0.25 the tolerance guard does not fire (strict >). f = int(0.25 n); for
        // n = 4k, 4k+1, 4k+2 that gives f = k and n >= 4k + 3 fails. The n=20 and n=40 sweeps refused
        // 0.25 for exactly this reason.
        assertThat(RobustMethod.BULYAN.refusalReason(0.25, DEFAULT_TRIM, n)).isPresent();
    }

    @ParameterizedTest
    @ValueSource(ints = {3, 7, 11, 39})
    void bulyanAtAQuarterRunsWhenTruncationLeavesRoom(int n) {
        // n = 4k + 3 gives f = k and exactly n = 4f + 3. Verified against the Python aggregator, which
        // runs at n = 3, 7, 11 ... 39 and refuses every other n in 1..40.
        assertThat(RobustMethod.BULYAN.refusalReason(0.25, DEFAULT_TRIM, n)).isEmpty();
    }

    @Test
    void bulyanNeverActuallyDefendsAQuarterOfTheCohort() {
        // The real finding behind "Bulyan cannot run at its quoted 0.25": whenever it does run, the
        // attacker share it is sized for, f/n = (n - 3) / 4n, stays strictly below a quarter.
        for (int n = 1; n <= 2000; n++) {
            if (RobustMethod.BULYAN.refusalReason(0.25, DEFAULT_TRIM, n).isEmpty()) {
                double defended = (double) RobustMethod.BULYAN.attackerCount(0.25, n) / n;
                assertThat(defended).as("defended share at n=%d", n).isLessThan(0.25);
            }
        }
    }

    @Test
    void bulyanAboveItsToleranceIsRefusedWhateverTheCohort() {
        assertThat(RobustMethod.BULYAN.refusalReason(0.3, DEFAULT_TRIM, 10_000))
                .hasValueSatisfying(r -> assertThat(r).contains("tolerance"));
    }

    @Test
    void krumAtFortyClientsRunsToNearlyHalfAndRefusesAtHalf() {
        // Matches the n=40 cross-attack sweep: Krum ran at f=0.45 and was refused at f=0.5.
        assertThat(RobustMethod.KRUM.refusalReason(0.45, DEFAULT_TRIM, 40)).isEmpty();
        assertThat(RobustMethod.KRUM.refusalReason(0.5, DEFAULT_TRIM, 40)).isPresent();
    }

    @Test
    void krumCannotRunWithTwoClientsEvenWithNoAttackers() {
        // f = 0 still needs n >= 3. Two clients is the start dialog's default.
        assertThat(RobustMethod.KRUM.refusalReason(0.0, DEFAULT_TRIM, 2)).isPresent();
        assertThat(RobustMethod.MULTI_KRUM.refusalReason(0.0, DEFAULT_TRIM, 2)).isPresent();
        assertThat(RobustMethod.KRUM.refusalReason(0.0, DEFAULT_TRIM, 3)).isEmpty();
    }

    @Test
    void trimmedMeanToleranceIsItsTrimRatio() {
        assertThat(RobustMethod.TRIMMED_MEAN.refusalReason(0.2, 0.1, 40)).isPresent();
        assertThat(RobustMethod.TRIMMED_MEAN.refusalReason(0.2, 0.2, 40)).isEmpty();   // strict >
    }

    @Test
    void medianAcceptsExactlyHalfAndRefusesMore() {
        assertThat(RobustMethod.MEDIAN.refusalReason(0.5, DEFAULT_TRIM, 1)).isEmpty();
        assertThat(RobustMethod.MEDIAN.refusalReason(0.51, DEFAULT_TRIM, 100)).isPresent();
    }

    @Test
    void attackerCountIsTruncatedExactlyAsPythonIntDoes() {
        // 0.29 * 100 = 28.999999999999996 in IEEE doubles; Python int() and Java (int) both give 28.
        // f=28 needs 2*28+3 = 59 <= 100. Rounding instead of truncating would give f=29 (needs 61).
        assertThat(RobustMethod.KRUM.attackerCount(0.29, 100)).isEqualTo(28);
    }

    @Test
    void centeredClipHasNoCohortPrecondition() {
        assertThat(RobustMethod.CENTERED_CLIP.refusalReason(0.5, DEFAULT_TRIM, 1)).isEmpty();
    }
}
