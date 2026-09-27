package com.federated.fl_platform_api.orchestration;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Terminates a spawned FL server together with everything it started.
 *
 * <p>The backend spawns a wrapper script ({@code run_fl_server.sh}), which runs {@code fl_server.py | tee} as its
 * children. Killing only the wrapper orphans them: the FL server keeps running and keeps its port. So the whole
 * tree is killed. The descendants are captured before anything is killed, because once the wrapper dies its
 * children are re-parented and no longer appear as its descendants.</p>
 */
final class ProcessTrees {

    private ProcessTrees() {
    }

    /** Forcibly terminate {@code root} and every descendant; returns all the processes it signalled. */
    static List<ProcessHandle> destroyForcibly(ProcessHandle root) {
        List<ProcessHandle> tree = new ArrayList<>(root.descendants().toList());
        tree.forEach(ProcessHandle::destroyForcibly);
        root.destroyForcibly();
        tree.add(root);
        return tree;
    }

    /** Wait up to {@code timeoutSeconds} for every process in {@code tree} to exit. */
    static void awaitExit(List<ProcessHandle> tree, long timeoutSeconds)
            throws InterruptedException, ExecutionException, TimeoutException {
        CompletableFuture.allOf(tree.stream().map(ProcessHandle::onExit).toArray(CompletableFuture[]::new))
                .get(timeoutSeconds, TimeUnit.SECONDS);
    }
}
