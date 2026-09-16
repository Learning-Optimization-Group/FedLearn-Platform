package com.federated.fl_platform_api.model;

/**
 * The one-time model-initialisation phase of a project (BA-1), stored on {@code projects.init_status}.
 *
 * <p>Model init runs a Python process to materialise the initial weights file. It happens at
 * <em>create</em> time — before any {@link Run} exists — so it cannot be expressed through the
 * run-derived {@link ProjectStatus} (BA-4): a project mid-init has no active run and would otherwise
 * read as the idle {@code CREATED}. This enum is the project-level init phase that
 * {@link com.federated.fl_platform_api.service.ProjectStatusService} consults <em>before</em> falling
 * through to run-derivation.</p>
 *
 * <ul>
 *   <li>{@link #INITIALIZING} — the async init worker is (or is about to be) running; the project is
 *       not yet trainable.</li>
 *   <li>{@link #DONE} — init finished successfully; status now defers entirely to the active run.
 *       This is the backfilled default for every pre-BA-1 row (they were created synchronously).</li>
 *   <li>{@link #FAILED} — init timed out or errored; the project row persists in a failed state the
 *       owner can see and delete/retry (the request no longer rolls the row back).</li>
 * </ul>
 */
public enum ProjectInitStatus {
    INITIALIZING,
    DONE,
    FAILED
}
