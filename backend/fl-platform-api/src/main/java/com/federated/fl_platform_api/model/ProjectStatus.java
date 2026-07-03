package com.federated.fl_platform_api.model;

/**
 * The lifecycle status of a project, derived from its active {@link Run} (BA-4).
 *
 * <p>Historically a project carried a free-form {@code String} status that was written eagerly and
 * independently of the run it describes, so it could drift from reality — most visibly, a run that
 * ended in {@link RunStatus#FAILED} left the project stuck showing {@code RUNNING}. This enum makes
 * the mapping from run state to project state the single source of truth.</p>
 *
 * <p>The constant names are deliberately identical to the status string tokens the web SPA and
 * desktop client already consume ({@code CREATED/RUNNING/STOPPED/COMPLETED/FAILED}); Jackson
 * serializes an enum by {@link #name()} by default, so exposing this type over the wire is
 * byte-compatible with the previous {@code String} and needs no Flyway migration.</p>
 */
public enum ProjectStatus {
    /**
     * The one-time model-init worker is still preparing the initial weights (BA-1). Reached only
     * before any run exists, via {@link ProjectInitStatus#INITIALIZING}; the SPA renders it as a
     * "Preparing" pill and polls until it resolves to {@link #CREATED} or {@link #FAILED}.
     */
    INITIALIZING,
    /** No active run yet (or the active run was deleted). The SPA renders this as an idle "Ready" pill. */
    CREATED,
    /** An active run is starting up or training. */
    RUNNING,
    /** The active run was stopped by an operator. */
    STOPPED,
    /** The active run finished successfully. */
    COMPLETED,
    /** The active run failed. Previously unreachable from the project status; the SPA already maps it to "Error". */
    FAILED;

    /**
     * Derive the project status from its active run's status. A {@code null} run status means the
     * project has no active run, which is {@link #CREATED}.
     */
    public static ProjectStatus fromActiveRunStatus(RunStatus runStatus) {
        if (runStatus == null) {
            return CREATED;   // no active run (or it was deleted, nulling active_run_id)
        }
        return switch (runStatus) {
            // A run is created in STARTING with active_run_id set at once; PENDING is currently dead
            // but means the same "spinning up" intent. Both read as RUNNING so the project never
            // shows idle while a server is actually being started (matches today's start UX).
            case PENDING, STARTING, RUNNING -> RUNNING;
            case COMPLETED -> COMPLETED;
            case STOPPED -> STOPPED;
            case FAILED -> FAILED;
        };
    }
}
