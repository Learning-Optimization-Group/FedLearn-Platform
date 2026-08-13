package com.federated.fl_platform_api.model;

/**
 * Which parameters a run trains and federates (P1).
 *
 * <ul>
 *   <li>{@code FULL}        — every parameter is trainable; the whole model rides the wire.</li>
 *   <li>{@code FROZEN_HEAD} — the backbone is frozen and only the head trains, so the wire carries
 *       the head alone. Dramatically cheaper per round; see {@code research/results/frozen-backbone/}.</li>
 * </ul>
 *
 * <p>Before P1 this was not a stored property at all. The Python client inferred it from the recipe
 * key ({@code USE_DERIVED = (mt == "FROZEN_DEMO")}), so every recipe had exactly one hard-coded arm,
 * one recipe could not be run under both arms as two halves of a comparison, and — the reason this
 * enum exists — a result could not say which arm produced it. Commit {@code 21699bc} ("frozen arm
 * silently mislabelled its backbone, risking cell overwrites") is that failure.
 *
 * <p>Which arms a given recipe actually supports is declared in {@code fl-runtime/recipes.py}
 * ({@code supported_arms} / {@code trainable_spec}), NOT here: this enum is the vocabulary, the
 * recipe catalog is the authority. Adding a constant here therefore requires widening the
 * {@code chk_projects_training_arm} CHECK in a new migration — {@code V22TrainingArmMigrationTest}
 * asserts every constant is accepted, so the split-brain fails loudly rather than at write time.
 */
public enum TrainingArm {
    /**
     * OvA-LP (arXiv:2511.05028): the same frozen encoder as FROZEN_HEAD, trained under C
     * independent one-vs-all binary classifiers instead of one softmax. It is a distinct ARM
     * rather than a strategy because it changes what a client trains and therefore what a result
     * means -- but note it federates the SAME parameters as FROZEN_HEAD, so only the objective
     * distinguishes their provenance. The paper's two-stage schedule is not implemented; see
     * fl-runtime/recipes.py arm_notes.
     */
    OVA_LP,
    FULL, FROZEN_HEAD
}
