package com.federated.fl_platform_api.model;

import java.util.Optional;

/**
 * The Byzantine-robust aggregation rules the Python server implements, with the start-time
 * feasibility check that mirrors its runtime guards.
 *
 * <p>Why the check lives here as well as in Python: a rule that cannot run at the run's cohort size
 * does not fail loudly on the server. {@code RobustAggregator.aggregate_fit} returns None every round,
 * the global model never moves, and the run "completes" at roughly random accuracy. The FR-12
 * breakdown benchmark hit exactly that and nearly recorded it as a measured breakdown point. Refusing
 * at {@code /start} is the only place a user gets told why.
 *
 * <p>The rules mirror {@code framework/src/fedlearn/server/robust_aggregation.py} and must move with it:
 * <ul>
 *   <li><b>Tolerance</b>: the server refuses when {@code byzantine_fraction > tolerance} (strict).
 *       For trimmed-mean the tolerance is the trim ratio; otherwise it is the rule's breakdown point.</li>
 *   <li><b>Cohort precondition</b>, with {@code f = int(fraction * n)}: Krum and Multi-Krum need
 *       {@code n >= 2f + 3}, Bulyan needs {@code n >= 4f + 3}. This is not monotone in n.</li>
 * </ul>
 * The n to check is every round size a run allows: a round completes once {@code clientsPerRound} updates
 * arrive and can still finish with {@code minClients} at the deadline, so n ranges over both and everything
 * between (see {@link RobustAggregationSettings#refusalReason(int, int)}).
 */
public enum RobustMethod {

    MEDIAN("median", 0.5, 0),
    TRIMMED_MEAN("trimmed_mean", Double.NaN, 0),
    KRUM("krum", 0.5, 2),
    MULTI_KRUM("multi_krum", 0.5, 2),
    BULYAN("bulyan", 0.25, 4),
    CENTERED_CLIP("centered_clip", 0.5, 0);

    /** {@code fl_server.py}'s trim ratio when {@code --robust-trim-ratio} is omitted. */
    public static final double DEFAULT_TRIM_RATIO = 0.1;

    private final String wireName;
    /** The rule's breakdown point; NaN for trimmed-mean, whose tolerance is its trim ratio. */
    private final double breakdown;
    /** k in {@code n >= k*f + 3}; 0 means the rule has no cohort precondition. */
    private final int attackerMultiplier;

    RobustMethod(String wireName, double breakdown, int attackerMultiplier) {
        this.wireName = wireName;
        this.breakdown = breakdown;
        this.attackerMultiplier = attackerMultiplier;
    }

    /** The value {@code fl_server.py --robust-method} accepts. */
    public String wireName() {
        return wireName;
    }

    public static RobustMethod fromWire(String wire) {
        for (RobustMethod m : values()) {
            if (m.wireName.equals(wire)) {
                return m;
            }
        }
        throw new IllegalArgumentException("Unknown robust aggregation method: " + wire);
    }

    /** The value the server's guard compares the Byzantine fraction against. */
    public double tolerance(double trimRatio) {
        return this == TRIMMED_MEAN ? trimRatio : breakdown;
    }

    /**
     * The attacker count f exactly as {@code robust_aggregation.py} computes it: {@code int(fraction * n)},
     * which truncates. Rounding would disagree with the server on inputs like 0.29 * 100.
     */
    public int attackerCount(double byzantineFraction, int cohortSize) {
        return (int) (byzantineFraction * cohortSize);
    }

    /**
     * The updates a round of {@code cohortSize} needs for this rule to run ({@code k*f + 3}), or 0 when the rule has
     * no cohort precondition.
     */
    public int updatesNeeded(double byzantineFraction, int cohortSize) {
        return attackerMultiplier == 0 ? 0 : attackerMultiplier * attackerCount(byzantineFraction, cohortSize) + 3;
    }

    /**
     * Why the server would refuse every round at this configuration, or empty if it can run.
     *
     * @param byzantineFraction the operator's estimate of the malicious share, in [0, 1)
     * @param trimRatio         the trim ratio actually in effect (the default when unset)
     * @param cohortSize        updates aggregated in the round being checked
     */
    public Optional<String> refusalReason(double byzantineFraction, double trimRatio, int cohortSize) {
        double tolerance = tolerance(trimRatio);
        if (byzantineFraction > tolerance) {
            return Optional.of(String.format(
                    "%s cannot run with byzantineFraction=%s: it exceeds the rule's tolerance of %s, so "
                            + "the server would refuse to aggregate every round and the model would never train.",
                    name(), byzantineFraction, tolerance));
        }
        if (attackerMultiplier > 0) {
            int f = attackerCount(byzantineFraction, cohortSize);
            int needed = updatesNeeded(byzantineFraction, cohortSize);
            if (cohortSize < needed) {
                return Optional.of(String.format(
                        "%s needs at least %d clients per round (%d x %d expected attackers + 3), but each "
                                + "round aggregates minClients = %d, so the server would refuse every round. "
                                + "Raise minClients, lower byzantineFraction, or choose another rule.",
                        name(), needed, attackerMultiplier, f, cohortSize));
            }
        }
        return Optional.empty();
    }
}
