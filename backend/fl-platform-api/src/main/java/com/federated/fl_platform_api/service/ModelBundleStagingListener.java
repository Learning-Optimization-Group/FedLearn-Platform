package com.federated.fl_platform_api.service;

import java.util.UUID;

/**
 * Hears how each run's bundle staging ended. The execution-contract publisher uses it to publish or refuse a
 * contract once the programs a contract binds are on disk. Implementations are called on the staging worker and
 * must not assume the caller survives their exceptions -- the stager contains them.
 */
public interface ModelBundleStagingListener {

    /** The run's bundle is staged (now, or by an earlier attempt). */
    void onStaged(UUID runId);

    /** The run's bundle will not be staged; {@code timedOut} is true when staging exceeded its deadline. */
    void onStagingFailed(UUID runId, boolean timedOut, String detail);
}
